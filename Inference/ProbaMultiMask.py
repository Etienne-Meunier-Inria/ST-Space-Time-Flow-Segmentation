import os, sys, torch, numpy as np, pandas as pd
sys.path.append(f'../utils')
from script_utils import load_sequence, list_seq, cut, unravel_sequence, custom_fl
from ShapeChecker import ShapeCheck
from pathlib import Path
from ipdb import set_trace
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from PIL import Image


def save_argmax(mat, path) :
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    mat = np.array(mat).astype('uint8')
    save_indexed(path, mat)

# From https://github.com/Jyxarthur/OCLR_model/blob/89ad2339107368ae5e4e23479a1605088170354d/utils.py#L120
def imwrite_indexed(filename, array, colour_palette):
    # Save indexed png for DAVIS
    im = Image.fromarray(array)
    im.putpalette(colour_palette.ravel())
    im.save(filename, format='PNG')

def save_indexed(filename, img, colours = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]):
    colour_palette = np.array([[0,0,0]] + colours).astype(np.uint8)
    imwrite_indexed(filename, img, colour_palette)

def probatomultimask(model_dir, data_file, cuts_size=None, base_dir=os.environ['Dataria']) :
    seqs = list_seq(data_file)
    for seq in tqdm(seqs, desc='probatomultimask') :
        r = load_sequence(data_file, seq, img_size=None, model_dir=model_dir, cuts_size=cuts_size, keys=['GtMask', 'PredModelSeq'], base_dir=base_dir)
        gt, gtp = zip(*[(ri['GtMask'], ri['GtMaskPath']) for ri in r])
        gtp = list(gtp)
        gt = list(gt)
        labels_seq = torch.tensor(r[0]['PredModelSeq'], dtype=torch.int64)
        flatlabel_seq = unravel_sequence(labels_seq)
        #Because we don't predict a mask for the last frame ( there is no forward flow) we duplicate the
        #mask for the frame before for evaluation in D17.
        flatlabel_seq = torch.cat([flatlabel_seq, flatlabel_seq[-1:]])
        gt.append(gt[-1])
        gtp.append(f'{Path(gtp[-1]).parent}/{int(Path(gtp[-1]).stem)+1:05d}{Path(gtp[-1]).suffix}')

        for i in range(flatlabel_seq.shape[0]) :
            pbig =  resize(flatlabel_seq[i][None, None], gt[i].shape, interpolation=InterpolationMode.NEAREST)[0,0] + 1
            seq = Path(r[0]['PredModelSeqPath']).parent.name
            dir = Path(r[0]['PredModelSeqPath']).parent.parent.parent
            output_path = dir  / Path(f'cs{cuts_size}') / seq / Path(gtp[i]).name
            save_argmax(pbig, output_path)

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
    probatomultimask(args.model_dir, args.data_file, args.cuts_size, base_dir=args.base_dir)
