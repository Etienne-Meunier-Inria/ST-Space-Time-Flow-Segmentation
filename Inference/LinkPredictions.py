import os, sys, torch, numpy as np, pandas as pd
sys.path.append(f'../utils')
from script_utils import load_sequence, list_seq, cut
from pathlib import Path
from argparse import ArgumentParser
from ipdb import set_trace
from tqdm import tqdm


def link_predictions_naive(predf, L) :
    """
    Link predictions label between batches in predf and return a new segmentation
    map with consistent predictions.
    Args :
        predf (b t i j) : segmentation label per batches
        L int : max number of classes
    Returns :
        labels ( b t i j) : segmentation labels temporally consistent
    """
    iou = lambda x, y : (x & y).sum() / (x | y).sum() if  (x | y).sum() else 1.0
    labels = predf.copy()
    for t in range(1, predf.shape[0]) :

        corres = np.zeros((L, L))
        for k in range(L) :
            for l in range(L) :
                corres[k,l] = iou((labels[t-1,1] == k), (labels[t,0] == l)) + iou((labels[t-1,2] == k), (labels[t,1] == l))
        new_label = corres.argmax(axis=0)
        for i in range(L) :
            labels[t,:][predf[t,:] == i] = new_label[i]
    return labels

def link_predictions_naive_cuts(predm, L, cuts_size=None) :
    cuts_size = cuts_size if cuts_size is not None else len(predm)
    predm_cut = cut(predm, I=cuts_size)
    labels = [link_predictions_naive(pfc, L) for pfc in predm_cut]
    labels = np.concatenate(labels)
    return labels

def save_labels_seq(labels_seq, path, cuts_size) :
    np.save(str(Path(path).parent) + f'/{Path(path).parent.stem}_cs{cuts_size}_labelseq.npy', labels_seq)

def linkpredictions(model_dir, data_file, cuts_size=None, base_dir=os.environ['Dataria']) :
    seqs = list_seq(data_file)
    for seq in tqdm(seqs, desc='linkpredictions') :
        r = load_sequence(model_dir=model_dir, data_file=data_file, sequence=seq, base_dir=base_dir)
        preds, paths = zip(*[(ri['PredModel'], ri['PredModelPath']) for ri in r])
        L, t = preds[1].shape[:2]
        predm = torch.stack(preds[t//2:-(t//2)]).numpy().argmax(1)
        labels_seq = link_predictions_naive_cuts(predm, L=L, cuts_size=cuts_size)
        save_labels_seq(labels_seq, paths[1 + t//2], cuts_size)


if __name__ == '__main__'  :
    parser = ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str)
    parser.add_argument('--cuts_size', '-cs', type=int, default=None)
    parser.add_argument('--base_dir', '-b', type=str, default=os.environ['Dataria'])
    parser.add_argument('--data_file', type=str, choices=['DAVIS_D16Split_train', 'DAVIS_D16Split_val',
                                                          'SegTrackv2_EvalSplit_val',
                                                          'FBMSclean_FBMSSplit_val', 'DAVIS17_D17Split_val', 'DAVIS17_D17Split_train'])

    args = parser.parse_args()
    if args.cuts_size == -1 :
        args.cuts_size = None
    linkpredictions(args.model_dir, args.data_file, args.cuts_size, base_dir=args.base_dir)
