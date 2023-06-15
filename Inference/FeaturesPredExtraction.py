import sys, os, einops, torch.nn as nn, torch, numpy as np
from copy import deepcopy
from ipdb import set_trace
sys.path.append('../')
sys.path.append('../utils')
from csvflowdatamodule.CsvDataset import CsvDataModule
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from script_utils import prepare_model
from tqdm import tqdm

def get_model(model_dir, **kwargs) :
    model = prepare_model(model_dir).eval();
    model.hparams["flow_augmentation"] = ''
    model.cuda();
    print(f'Model Inputs : {model.backbone_model.model.inputs}')
    final_layer = deepcopy(model.backbone_model.model.model.layers.final_layer)
    print('Identity in Get Model')
    model.backbone_model.model.model.layers.final_layer = nn.Identity()
    return model, final_layer


def get_data(data_file, hparams, batch_size=20, preload_cache=False, **kwargs) :
    hparams['batch_size'] = batch_size
    hparams['preload_cache'] = False
    hparams['data_file'] = data_file
    data_path  = f'../DataSplit/{data_file}_'+'{}.csv'
    dm = CsvDataModule(data_path =data_path, request=set(hparams['inputs']),
                       num_workers=0, shuffle_fit=False, **hparams)
    dm.setup('fit')
    return dm

def compute_features_preds(model, final_layer, batch, save_dir, extract_features=False, device='cuda') :
    flowv = einops.rearrange([batch[k] for k in model.inputs],
                                 't b c h w -> b c t h w')

    with torch.no_grad() :
        features, _ = model.model(flowv.to(device))
        pred = torch.softmax(final_layer(features), dim=1)

    for i in range(len(batch['FlowPath'])) :
        if extract_features :
            name = save_dir+batch['FlowPath'][i].replace('.flo', '_features.npy')
            Path(name).parent.mkdir(parents=True, exist_ok=True)
            np.save(name, features[i].cpu().numpy())

        name = save_dir+batch['FlowPath'][i].replace('.flo', '_proba.npy')
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        np.save(name, pred[i].cpu().numpy())


def extract_predfeatures(model_dir, data_file, extract_features=False) :
    model, final_layer = get_model(model_dir)
    dm = get_data(data_file, model.hparams)
    for split in ['val', 'train'] :
        for batch in tqdm(dm.get_dataloader(split), desc='predfeaturesextraction') :
            compute_features_preds(model.backbone_model.model, final_layer, batch, model_dir +'/')


if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str, required=True)
    parser.add_argument('--base_dir', '-b', type=str, required=True)
    parser.add_argument('--data_file', type=str, choices=['DAVIS_D16Split','DAVIS17_D17Split', 'FBMSclean_FBMSSplit',
                                                          'SegTrackv2_EvalSplit'], nargs='+', required=True)
    parser.add_argument('--extract_features', action='store_true')

    args = parser.parse_args()
    for data_file in args.data_file :
        extract_predfeatures(args.model_dir, args.data_file, args.extract_features)
