from csvflowdatamodule.CsvDataset import CsvDataset
from csvflowdatamodule.Transforms.Transforms import TransformsComposer
from torchvision import transforms
import os, torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from ipdb import set_trace

class DataLoadingModule(Dataset) :
    def __init__(self, data_file, data_base_dir, pred_base_dir, request, pred_suffix,  **k) :
        """DataLoadingModule mix the usual CsvDataloader with the predictions.

        Parameters
        ----------
        data_file : str
            String of the DataSplit to use from DataSplit/
        data_base_dir : str
            Base dir for the data field
        pred_base_dir : str
            Path of the base path of the model you want to evaluate.
        request list(str) :
            List of fields we want to extract from our data
        pred_suffix (str) :
            Suffix to add to the pred path
        """

        # Load Data
        self.data_path  = f'../DataSplit/{data_file}.csv'
        self.data_base_dir = data_base_dir
        self.request = request.add('Image')
        # No transform or flow normalisation ( anyway predictions are uncomputed)
        transform = TransformsComposer(flow_augmentation='')

        # No Resizing : We keep the original GTMasks Size for evaluation
        self.dst = CsvDataset(self.data_path, self.data_base_dir, img_size=None,
                              request=request, transform=transform)


        # Load Prediction
        self.pred_base_dir = pred_base_dir
        self.pred_suffix = pred_suffix

    def __len__(self) :
        return self.dst.__len__()

    def __getitem__(self, idx) :
        """Returns the item and prediction at a given index.

        Parameters
        ----------
        idx : int
            Index of the item to get in the dataloader

        Returns
        -------
        dict
            Dictionnary with on key for each field in request and Pred key
            for the prediction of the model
        """
        d = self.dst.__getitem__(idx)
        try :
            # Add Leading dimension to each tensor in the batch
            img_size = d['GtMask'].shape
            predpath = d['FlowPath'].replace(Path(d['FlowPath']).suffix, f'{self.pred_suffix}.npy')
            d['PredPath'] = predpath
            pred = torch.tensor(np.load(self.pred_base_dir+'/'+predpath))
            resize = transforms.Resize(img_size)
            pred = resize(pred[None])[0] # (b ,nAffineMasks, i, j)
            d['Pred'] = pred
        except Exception as e :
            print(e)
            d = None
        return d

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0 :
            return None
        return default_collate(batch)

if __name__=='__main__' :
    data_file = 'DAVIS_D16Split_val'
    data_base_dir = os.environ['Dataria']
    pred_base_dir = f'{os.environ["Dataria"]}/Models/SegGrOptFlow/vir/m4mmn3jp/'
    request = set(['Image', 'Flow', 'GtMask'])
    dlm = DataLoadingModule(data_file, data_base_dir, pred_base_dir, request)
    d = dlm.__getitem__(4)

    dld = DataLoader(dlm, collate_fn=dlm.collate_fn, batch_size=5)
    r = iter(dld).next()
    r['ImagePath']
