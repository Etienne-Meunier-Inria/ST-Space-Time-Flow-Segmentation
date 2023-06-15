import os, sys, torch, numpy as np, pandas as pd
sys.path.append(f'../utils')
from script_utils import load_sequence, list_seq, cut, unravel_sequence, custom_fl
from ShapeChecker import ShapeCheck
from evaluations import db_eval_iou
from pathlib import Path
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from ipdb import set_trace

def optimax_argmax(argmax, GtMask, L, max_segs=100) :
    """
    Choose the best labels based on the comparison with GtMask based on time t only ( central time)
    Params :
        argmax (b i j) argmax map with times [t-k... t ... t+k] with coherent segmentation labels
        GtMask (b i j) gt map for time t
    Returns
        optimax_masks_vol (b i j) binary labels with the best selection
    """
    sc = ShapeCheck([L], 'L')
    sc.update(argmax.shape, 'b i j')
    sc.update(GtMask.shape, 'b i j')

    binmax = sc.repeat(torch.zeros(1, device=argmax.device), '1 -> b L i j').clone()
    binmax.scatter_(1, argmax[:, None], 1)
    s_channels = []
    for itr in range(min(L, max_segs)) :
        bench=[]
        for k in range(L) :
            pdm = sum([binmax[torch.arange(sc.get('b')['b']),s,:,:] == 1.0
                                     for s in s_channels + [k]]).to(torch.bool)
            bench.append(db_eval_iou(GtMask, pdm))
        bench = torch.stack(bench)
        s_channels.append(bench.argmax(axis=0))
    optimax_masks = sum([argmax == sc.repeat(s, 'b -> b i j') for s in s_channels]).to(bool)
    return optimax_masks, s_channels


def optimax_sequence(labels, gts, L, max_segs=100) :
    """
    Extract the best masks to evaluate the sequence depending on Ground Thruth
    labels (b i j) : labels consistent across full sequence
    gts (b i j) : binary GT mask for the sequence
    L : number of possible segments
    max_segs : maximum number of segments selectable for aggregation
    return :
        binary_mask_seq (b i j) : binary mask with the best selection
        idx_seq (b, min(max_segs, L)) : index selection for each element in the batch ( should be the same for all sequence)
    """
    sc = ShapeCheck([min(L, max_segs)], 'k')
    sc.update(labels.shape, 'b i j')
    sc.update(gts.shape, 'b i j')
    argmax = sc.rearrange(labels, 'b i j -> 1 i (b j)')
    gts = sc.rearrange(gts, 'b i j -> 1 i (b j)')
    binary_mask_seq, idx = optimax_argmax(argmax, gts, L=L)
    binary_mask_seq = sc.rearrange(binary_mask_seq, '1 i (b j) -> b i j')
    idx_seq =  sc.repeat(sc.rearrange(idx, 'k 1-> k'), 'k -> b k')
    return binary_mask_seq, idx_seq

def optimax_sequence_cuts(labels, gts, L, cuts_size=None) :
    """
    Call optimax on sequences of size cs and return the optimal masks
    Params :
        labels (b i j) : consistent labels by group of size cs for the sequence
        gts (b i j) : gt labels for the sequence
        L : max number of masks
        cut_size : size of the cut for the choice of the masks in evaluation
    Returns :
        binary_mask_seqs ( b i j) : binary output after selection
        idx (b): index of selected masks for each element of the batch
    """
    cuts_size = cuts_size if cuts_size is not None else len(labels)
    sc = ShapeCheck([L],'L')
    sc.update(labels.shape, 'b i j')
    gts_cut = cut(gts, I=cuts_size, shortcut=True)
    labels_cut = cut(labels, I=cuts_size, shortcut=True)
    sc.update([len(gts), sum([g is not None for g in gts]), len(gts_cut), len(labels_cut)], 'b b_effective n_cuts n_cuts')

    binary_mask_seqs = []
    binary_mask_seqs_idx = []
    for i in range(sc.get('n_cuts')['n_cuts']):
        flte = [g is not None for g in gts_cut[i]]
        if sum(flte) > 0 :
            l = labels_cut[i][flte]
            gs = torch.stack([g for g in gts_cut[i] if g is not None])#.numpy()
            bmsq, bidx = optimax_sequence(l, gs, L=L)
            binary_mask_seqs.append(bmsq)
            binary_mask_seqs_idx.append(bidx)

    sc.update([len(binary_mask_seqs), len(binary_mask_seqs_idx)], 'n_cuts_effective n_cuts_effective')
    binary_mask_seqs = torch.cat(binary_mask_seqs)
    binary_mask_seqs_idx = torch.cat(binary_mask_seqs_idx)
    sc.update(binary_mask_seqs.shape, 'b_effective i j')
    sc.update(binary_mask_seqs_idx.shape, 'b_effective L')
    return binary_mask_seqs, binary_mask_seqs_idx


def save_binaries(binary_seq, paths, cuts_size) :
    assert len(binary_seq) == len(paths), f'Error in lenghts for save {len(binary_seq)} != {len(paths)}'
    for i in range(len(paths)) :
        prob = torch.stack([binary_seq[i]==False, binary_seq[i]]).to(float)
        np.save(paths[i].replace('.png', f'_seq_cs{cuts_size}binary_eval.npy'), prob)

def maskselectionoptimax(model_dir, data_file, cuts_size=None, base_dir=os.environ['Dataria']) :
    seqs = list_seq(data_file)
    for seq in tqdm(seqs, desc='maskselectionoptimax')  :
        r = load_sequence(data_file, seq, model_dir=model_dir, cuts_size=cuts_size, keys=['GtMask', 'PredModelSeq'], base_dir=base_dir)
        gts, gtp = zip(*[(ri['GtMask'], ri['GtMaskPath']) for ri in r])
        gts = list(gts)
        labels_seq = torch.tensor(r[0]['PredModelSeq'], dtype=torch.int64)
        flatlabel_seq = unravel_sequence(labels_seq)

        binary_mask_seq, binary_mask_seqs_idx = optimax_sequence_cuts(flatlabel_seq, gts, len(np.unique(flatlabel_seq)), cuts_size=cuts_size)

        save_dir = Path(r[0]['PredModelSeqPath']).parent
        save_binaries(binary_mask_seq, [str(save_dir  / Path(gtpi).name) for gtpi in gtp if gtpi is not None], cuts_size)


if __name__ == '__main__' :
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
    maskselectionoptimax(args.model_dir, args.data_file, args.cuts_size, base_dir=args.base_dir)
