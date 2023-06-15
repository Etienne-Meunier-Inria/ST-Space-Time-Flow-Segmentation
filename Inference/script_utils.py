import os, sys, torch, numpy as np, pandas as pd, yaml
from pathlib import Path
sys.path.append('..')
from csvflowdatamodule.CsvDataset import FilesLoaders
from Models.CoherenceNets.MethodeB import MethodeB
from ipdb import set_trace
from glob import glob

def load_numpy(numpy_path) :
    return torch.tensor(np.load(numpy_path))

def custom_fl() :
    fl = FilesLoaders()
    fl.load_numpy = load_numpy
    fl.loaders['PredModel'] = fl.load_numpy
    fl.loaders['FeatureModel'] = fl.load_numpy
    fl.loaders['Dino'] = fl.load_numpy
    return fl

def load_sequence(data_file, sequence, img_size=[128, 224], keys=['PredModel'], model_dir=None, cuts_size='NA', base_dir = os.environ['Dataria']) :
    data_path  = f'../DataSplit/{data_file}.csv'
    df = pd.read_csv(data_path)
    df = df[df['Sequence'] == sequence]
    if 'PredModel' in keys :
        assert model_dir is not None, 'Model dir is None. Impossible to fetch preds.'
        df['PredModel'] = model_dir + '/' +df['Flow'].str.replace('.flo','_proba.npy', regex=False)
    fl = custom_fl()
    r =[]
    for i, d in df.iterrows() :
        g = {}
        for k in keys :
            try :
                g[f'{k}Path'] = base_dir + d[k] if d[k] is not np.nan else None
                g[k] = fl.load_file(g[f'{k}Path'], k, img_size=img_size)
                if g[k].dtype == torch.float32 :
                    g[k] = g[k].to(torch.float16)
            except Exception as e :
                g[k] = None
        r.append(g)
    if 'PredModelSeq' in keys :
        assert model_dir is not None, 'Model dir is None. Impossible to fetch preds sequence.'
        assert cuts_size != 'NA', 'Cut size is NA Impossible to fetch preds sequence.'
        r[0]['PredModelSeqPath'] =  base_dir + model_dir + '/' + str(Path(df['Flow'].iloc[0]).parent) + f'/{sequence}_cs{cuts_size}_labelseq.npy'
        r[0]['PredModelSeq'] = np.load(r[0]['PredModelSeqPath'])
    return r

def list_seq(data_file) :
    data_path  = f'../DataSplit/{data_file}.csv'
    df = pd.read_csv(data_path)
    return df['Sequence'].unique().tolist()


def cut(m, I, shortcut=False) :
    """
    Cut the matrix (b, *) in a list of mini matrix [(I, *) ... (b%I, *)
    """
    if shortcut :
        rf = m[0]
        rl = m[-1]
        m = m[1:-1]
    cm = [m[i*I:min(i*I+I, len(m))] for i in range(0, len(m)//I + (1 if (len(m)%I > 0) else 0))]
    if shortcut :
        if type(cm[0]) == torch.Tensor :
            cm[0] = torch.cat([rf[None], cm[0]])
            cm[-1] = torch.cat([cm[-1], rl[None]])
        if type(cm[0]) == list :
            cm[0] = [rf] + list(cm[0])
            cm[-1] = list(cm[-1]) + [rl]
    return cm


def unravel_sequence(seq) :
    """
    Unravel temporal sequence to a flat one
    seq (b t i j) : temporal sequence
    return
        flatseq (b+(t-1) i j) unraveled sequence
    """
    b, t, i, j = seq.shape
    k = t//2
    flatseq = torch.full((b+k*2, i, j), 100, dtype=seq.dtype)
    flatseq[k:-k] = seq[:,k]
    flatseq[:k] = seq[0,:k]
    flatseq[-k:] = seq[-1,-k:]
    return flatseq

def prepare_model(model_dir, img_size_gen=None,  binary_method_gen=None, len_seq_gen=None) :
    """
    Load the best checkpoint of this model ( based on the validation loss)
    Args :
        model_dir (str) : path of the model to load containing ckpt files
        img_size_gen (list int) : size of the image to generate
        len_seq (int) : lenght of the sequence ( temporal dimension)
    Returns :
        net (nn.Module) : return loaded pytorch models
    """

    lpm = glob(os.path.join(model_dir,'checkpoints/*epoch_val_loss*.ckpt'))
    print(model_dir)
    pm = sorted(lpm, key = lambda x : float(x.split(':')[-1].strip('.ckpt')))[0]
    print(f'Using Checkpoint : {pm}')

    t = torch.load(pm, map_location='cpu')
    print(f'Model type : {t["hyper_parameters"]["model_type"]} Backbone :{t["hyper_parameters"]["backbone"]}')
    assert t['hyper_parameters']['model_type'] == 'coherence_B', 'Error in model type'

    if binary_method_gen is None :
        binary_method_gen = t['hyper_parameters']['binary_method']

    net = MethodeB.load_from_checkpoint(pm, strict=False, binary_method=binary_method_gen)
    if img_size_gen is not None and len_seq_gen is not None:
        net.init_param_model(img_size_gen, len_seq_gen, net.hparams['param_model'])
        net.hparams['img_size'] = img_size_gen

    print(f'Hyperparameters : {net.hparams}')
    with open(pm.replace('ckpt', 'hyp'), 'w') as f : yaml.dump(net.hparams, f)
    net.data_path  = f'DataSplit/{net.hparams.data_file}_'+'{}.csv'
    return net
