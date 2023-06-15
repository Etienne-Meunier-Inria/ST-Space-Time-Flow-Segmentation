from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from argparse import ArgumentParser
import flowiz
from PIL import Image
import numpy as np
from ipdb import set_trace
import pytorch_lightning as pl
import os
from pathlib import Path
from torchvision import transforms, io
from torch.utils.data.dataloader import default_collate
from .Transforms.Transforms import TransformsComposer
from .utils.KeyTypes import parse_request
import torchvision
from tqdm import tqdm
import psutil
import pickle
import re
from pathlib import Path

class FilesLoaders() :
    def __init__(self) :
        self.loaders =  {'Flow' :  self.load_flow,
                         'Image' :  self.load_image,
                         'GtMask' :  self.load_mask}

    def load_file(self, path, typekey, img_size=None) :
        file = self.loaders[typekey](path)
        if img_size is not None:
            if type(img_size[0]) == float :
                img_size = (int(img_size[0]*file.shape[-2]), int(img_size[1]*file.shape[-1]))
            if not list(file.shape[-2:]) == img_size:
                resize = transforms.Resize(img_size)
                file = resize(file[None])[0] # You need to add and remove a leading dimension because resize doesn't accept (W,H) inputs
        assert torch.isnan(file).sum() == 0, f'Nan in file {path}'
        return file

    @staticmethod
    def load_flow(flow_path) :
        flow = torch.tensor(flowiz.read_flow(flow_path)).permute(2, 0, 1) # 2, i, j
        assert flow.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return flow

    @staticmethod
    def load_image(image_path) :
        #im = (io.read_image(image_path)/255.) - 0.5 # c, i, j : [-0.5; 0.5]
        im = (torch.tensor(np.array(Image.open(image_path))).permute(2,0,1)/255.) # c, i, j : [0; 1]
        assert im.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return im

    @staticmethod
    def load_mask(mask_path) :
        im = torch.tensor(np.array(Image.open(mask_path))/255., dtype=torch.long) # i, j, c
        if im.ndim == 3 :
            im = im[:,:,0]
        assert im.ndim == 2, f'Wrong Number of dimension : {mask_path}'
        return im

class CSVFile() :
    def __init__(self, csv_path, boundaries='Ignore', request=None, subsample=1) :
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        if subsample < 1 :
            self.df = self.df.sample(frac=subsample, random_state=123)
        self.db = self.df.set_index(keys=['Sequence', 'Sequence_Index'], verify_integrity=True)
        self.idx_map = dict(zip(range(len(self.db.index)), self.db.index))
        self.boundaries = boundaries
        print(f'Boundaries : {boundaries}')
        if self.boundaries == 'Strict' :
            assert request is not None, 'Scrict Boundaries : need to provide request to filter dataset.'
            self.filter_index(request)

    def __len__(self) :
        return len(self.idx_map)

    def get_item(self, sequence, sequence_idx, request) :
        """
        Get a requested element in the csv.
        Args :
            sequence (str) : name of the sequence to get the item
            sequence_idx (int) : index of the element in the sequence
            request (str) : field to get
        """
        try :
            r = self.db.loc[sequence, sequence_idx]
            return r[request]
        except KeyError :
            return False

    def get_sequence(self, idx) :
        """
        From a given index ( in the database ) retrieve a sequence and sequence index
        This function goes by idx_map incase there is rows we can't directly access
        Args :
            idx (int) : Item to get in the database
        Return :
            sequence (str) : name of the sequence to get the item
            sequence_idx (int) : index of the element in the sequence
        """
        sequence,  sequence_idx = self.idx_map[idx]
        return sequence,  sequence_idx


    def get_requests(self, idx, requests) :
        """
        Get and return the paths for a list of requests.
        Args :
                idx (int) : Item to get in the database
                requests ([str]) : List of requests to get in the database for this index.
        Returns :
                dict ({str:str}) : dictionnary with the path for all request
        """
        sequence, sequence_idx = self.get_sequence(idx)
        r = {}
        for request in requests :
            colname, gap = self.parse_request(request)
            fetch_index = self.fetch_index(sequence_idx+gap, sequence)
            item = self.get_item(sequence, fetch_index, colname)
            r[request] = item
        return r

    def fetch_index(self, index, sequence) :
        """
        Return index to fetch. Depending if boundaries is Periodic
        """
        if self.boundaries == 'Periodic' :
            idxs = self.db.loc[sequence].index
            return self.bounce(index, idxs.min(), idxs.max())
        else :
            return index

    @staticmethod
    def bounce(index, idmin, idmax) :
        """
        Return the index after "bouncing" on boundaries (idmax, idxmin)
        Args :
            index (int) : index to retrieve
            idmin (int) : min available index
            idmax (int) : max available index
        Return :
            index (int) : index after bouncing (possibly several times) on boundaries
        """
        assert idmax > idmin, f'Problem in boundaries {idmax} {idmin}'
        if index > idmax :
         return CSVFile.bounce(idmax - (index - idmax), idmin, idmax)
        if index < idmin :
         return CSVFile.bounce(idmin + (idmin-index), idmin, idmax)
        return index

    def parse_request(self, request) :
        """
        parse a request constraining the colnames to the list of available columns
        Args :
            request (str) : string with '{ColName}{opt:+/-}{opt:offset}'
        Returns :
            colname (str) : string with the name of the column
            gap (signed int) : offset from the reference index
        """
        return parse_request(request, self.df.columns)


    def filter_index(self, requests) :
        """
        Filter idx_map to only keep rows that can respond to the requests.
        Only use for the val loader to avoid bug during validation
        1. Get all rows that can fulfill requests
        2. Filter idx_map to only map to those rows
        """
        print(f'Filter index in request : {requests}')
        flt = [idx for i, idx in enumerate(self.db.index) if all(self.get_requests(i, requests).values())]
        self.idx_map = dict(zip(range(len(flt)), flt))


class CsvDataset(Dataset) :
    def __init__(self,
                 data_path: str,
                 base_dir: str,
                 img_size: tuple, # i, j
                 request: list, # List of fields you want in the folder
                 subsample = 1,# percemtage of the dataset available ( if this is under 1 we subsample randomly )
                 transform=None,
                 preload_cache=False,
                 boundaries='Strict'):
        super().__init__()

        self.img_size = img_size
        self.transform = transform
        self.base_dir = base_dir
        self.data_path = Path(data_path)
        self.fs = CSVFile(data_path, boundaries=boundaries, request=request)
        print('request : ', request)
        self.request = request.copy()
        self.fl = FilesLoaders()
        self.cache = {}
        if preload_cache :
         self.preload_cache()


    def __len__(self) :
        return len(self.fs)

    def getter(self, idx, function) :
        dict_paths = self.fs.get_requests(idx, self.request)
        if not all(dict_paths.values()) : return None
        ret = {}
        for key, value in dict_paths.items() :
            try :
                if isinstance(value, (np.integer, np.float)) :
                    assert value is not np.nan, f'Error in the Datasplit in {key}'
                    ret[key] = value
                elif isinstance(value, (str)) :
                    ret[f'{key}Path'] =  value
                    ret[key] = function(value, self.fs.parse_request(key)[0], self.img_size)
                else :
                    raise Exception(f'Data type of {key} not handled')
            except Exception as e :
                 #set_trace()
                 print(e, value) # File does not exist
                 return None
        return ret

    def __getitem__(self, idx):
        ret =  self.getter(idx, self.load)
        if ret is None : return None
        return self.transform(ret)

    #def __getitems__(self, possibly_batched_index) :
    #    set_trace()

    @staticmethod
    def cache_key(filename, img_size) :
        return (filename, str(img_size)) # Key for dict

    def preload_cache(self) :
        # First check if we have a pickle file corresponding to load as base
        with self.data_path.parent / 'Pickles' / self.data_path.with_suffix('.pickle').name as p :
            if p.exists() :
                print(f'Using pickle file : {p} as base')
                with open(p, 'rb') as handle:
                    self.cache = pickle.load(handle)

        for idx in tqdm(range(len(self.fs)), desc='Preload Cache') :
             self.getter(idx, self.preload)
        print(f'Dataset preloaded\n\t Size : {len(self.cache)} RAM usage : {psutil.Process().memory_info().rss / (1024 * 1024 * 1000):.2f} Gb')

    def preload(self, filename, typekey, img_size) :
        #set_trace()
        key = self.cache_key(filename, img_size)
        if key not in self.cache :
            print(f'Not in cache : {key}')
            self.cache[key] = self.fl.load_file(os.path.join(self.base_dir, filename), typekey, self.img_size)
        else :
            pass
            #print(f'In cache : {key}')

    def load(self, filename, typekey, img_size) :
        key = self.cache_key(filename, img_size)
        if key in self.cache :
            return self.cache[key].clone()
        else :
            #print(f'Not in cache {key}')
            return self.fl.load_file(os.path.join(self.base_dir, filename), typekey, self.img_size)



class CsvDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str,
                       base_dir: str,
                       batch_size: int,
                       request: list, # List of fields you want in the folder
                       img_size : tuple,
                       subsample_train=1, # percentage of the train data to use for training.
                       shuffle_fit=True,
                       preload_cache=False,
                       num_workers = 0,
                       boundaries='Strict',
                       **kwargs) :
        super().__init__()
        print(f'num workers : {num_workers}')
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.base_dir = base_dir
        self.request = request
        self.subsample_train = subsample_train
        self.shuffle_fit = shuffle_fit
        self.preload_cache = preload_cache
        self.boundaries = boundaries
        self.kwargs_dataloader = {'batch_size':self.batch_size,
                                  'collate_fn' : self.collate_fn,
                                  'persistent_workers':num_workers>0,
                                  'num_workers': num_workers,
                                  'pin_memory':num_workers>0,
                                  'drop_last': False}
        self.set_transformations(**kwargs)


    def set_transformations(self, flow_augmentation, val_augment=False, **kwargs) :
        self.transforms = {}
        self.transforms['train'] = TransformsComposer(flow_augmentation)
        if val_augment :
            print('Enabling Data Augmentation on validation set')
            self.transforms['val'] = TransformsComposer(flow_augmentation)
        else :
            self.transforms['val'] = TransformsComposer('')
        self.transforms['test'] = TransformsComposer('')

    def setup(self, stage=None):
        print(f'Loading data in : {self.data_path} ------ Stage : {stage}')
        if stage == 'fit' or stage is None:
             self.dtrain = CsvDataset(self.data_path.format('train'), self.base_dir, self.img_size, self.request,
                                      subsample=self.subsample_train, transform=self.transforms['train'],
                                      preload_cache=self.preload_cache, boundaries=self.boundaries)
             self.dval = CsvDataset(self.data_path.format('val'), self.base_dir, self.img_size, self.request,
                                    subsample=self.subsample_train, transform=self.transforms['val'],
                                    preload_cache=self.preload_cache, boundaries=self.boundaries)
        if stage == 'test' or stage is None: # For now the val and test are the same
             self.dtest = CsvDataset(self.data_path.format('test'), self.base_dir, self.img_size, self.request,
                                     transform=self.transforms['test'],
                                     preload_cache=self.preload_cache, boundaries=self.boundaries)
        self.size(stage)

    def size(self, stage=None) :
        print('Size of dataset :')
        if stage == 'fit' or stage is None:
            print(f'\tTrain : {self.dtrain.__len__()} \t Val : {self.dval.__len__()}')
        if stage == 'test' or stage is None:
            print(f'\t Test : {self.dtest.__len__()}')

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))

        if len(batch) == 0 :
            return None
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(self.dtrain, **self.kwargs_dataloader ,shuffle=self.shuffle_fit)

    def val_dataloader(self):
        return DataLoader(self.dval, **self.kwargs_dataloader, shuffle=self.shuffle_fit)

    def test_dataloader(self):
        return DataLoader(self.dtest, **self.kwargs_dataloader, shuffle=False)

    def get_sample(self, set=None) :
        if set == "train"  : return next(iter(self.train_dataloader()))
        elif set == "val"  : return next(iter(self.val_dataloader()))
        elif set == "test" : return next(iter(self.test_dataloader()))

    def get_dataloader(self, set=None) :
        if set == "train"  : return self.train_dataloader()
        elif set == "val"  : return self.val_dataloader()
        elif set == "test" : return self.test_dataloader()

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TransformsComposer.add_specific_args(parser)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--subsample_train', default=1, type=float)
        parser.add_argument('--img_size', nargs='+', type=int, default=[128, 224])
        parser.add_argument('--base_dir', type=str, required=True)
        parser.add_argument('--data_file', type=str, required=True)
        parser.add_argument('--boundaries', type=str,
                            choices=['Strict','Periodic', 'Ignore'], default='Ignore',
                            help='''Setup how to deal with sequence boundaries and access outside them .
                                    \n Ignore : Datamodule don't deal with boundaries.
                                    \n Strict : Prevent request outside boundaries by filtering the index.
                                    \n Periodic : No filtering, requested index are converted using bounce.''')
        parser.add_argument('--val_augment', action='store_true', help='Enable data Augmentation in validation')
        parser.add_argument('--preload_cache', action='store_true', help='Enable data Augmentation in validation')
        return parser
