import torch, os, sys
sys.path.append(f'..')
sys.path.append(f'../utils')
from evaluationmodule.DataLoadingModule import DataLoadingModule
from evaluationmodule.BinarisationModule import BinarisationModule
from evaluationmodule.ScoreModule import ScoreModule
from evaluationmodule.SaveModule import SaveModule
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from ipdb import set_trace
import time

def main(scoring_method, binary_method, model_id, data_file, pred_suffix,
         eval_session_name, data_base_dir, pred_base_dir) :
    if scoring_method == None :
        if 'Moca' in data_file :
            scoring_method = 'bbox_jacc'
        else :
            scoring_method = 'db_eval_iou'

    hparams = {}

    hparams['eval_session_name'] = eval_session_name
    hparams['data_file'] = data_file
    hparams['pred_suffix'] = pred_suffix
    hparams['data_base_dir'] = data_base_dir
    hparams['model_id'] = model_id
    hparams['pred_base_dir'] = pred_base_dir + f'/{hparams["model_id"]}/'
    hparams['scoring_method'] = scoring_method
    hparams['binary_method'] = binary_method

    scm = ScoreModule(hparams['scoring_method'])
    bnm = BinarisationModule(hparams['binary_method'])

    bnm.request.add('Flow')
    request = scm.request | bnm.request
    dlm = DataLoadingModule(**hparams, request=request)

    dld = DataLoader(dlm, collate_fn=dlm.collate_fn, batch_size=1, num_workers=4) # Dataloader
    svm = SaveModule(save_dir=hparams['pred_base_dir']+hparams['eval_session_name']+'/')


    for i, d in enumerate(tqdm(dld)) :
        try :
            bnm.binarise(d)
            scm.score(d)
            scm.stat_masks(d)
            svm.write_result(d)
            svm.save_binary(d)
            if i % 10 == 0 :
                svm.generate_fig(d)
        except Exception as e :
            print(e)
            pass
    hparams.update(svm.summarise_csv(dlm.dst.fs.df))
    svm.save_config(hparams)



if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--scoring_method', '-sm', type=str, choices=['db_eval_iou', 'bbox_jacc'], default=None)
    parser.add_argument('--binary_method', '-bm', type=str, choices=['optimax', 'exceptbiggest', 'heuristiclustering', 'fair', 'binarise_argmax'])
    parser.add_argument('--model_id', '-mi', type=str)
    parser.add_argument('--data_file', '-df', type=str)
    parser.add_argument('--data_base_dir', '-dbd', type=str, default=os.environ['Dataria'])
    parser.add_argument('--pred_base_dir', '-dbp', type=str, default=f'{os.environ["Dataria"]}/Models/SegGrOptFlow/vir-tempo/')
    parser.add_argument('--pred_suffix', '-ps', type=str)
    parser.add_argument('--eval_session_name', '-esn', type=str, default='')
    args = parser.parse_args()

    main(**vars(args))
