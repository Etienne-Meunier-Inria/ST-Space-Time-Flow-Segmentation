import flowiz, torch, wandb, os, pandas as pd, numpy as np,\
       matplotlib.pyplot as plt, pytorch_lightning as pl
from ipdb import set_trace
from pathlib import Path
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResultsLogger(pl.Callback) :
    def __init__(self, filepath=None, keys=['losses', 'masks_usages', 'jaccs']):
        super().__init__()
        self.fp = filepath
        self.keys = keys

    def setup(self, trainer, pl_module, stage) :
        if self.fp is None :
            self.fp = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.experiment.id, 'results.csv')
        print(f'Save results in {self.fp}')
        with open(self.fp, 'w') as f :
            f.write(f'epoch,step_label,file_name,'+','.join(self.keys)+'\n')

    @torch.no_grad()
    def write_results(self, imps, outputs, epoch, step_label) :
        with open(self.fp, 'a') as f :
            for i, imn in enumerate(imps) :
                f.write(f'{epoch},{step_label},{imn.strip(os.environ["Dataria"])},'+','.join([f'{outputs[j][i].item():.3f}' for j in self.keys if j in outputs.keys()])+'\n')

    def batch_end(self, trainer, outputs, batch, step_label):
        if batch is None : return None
        key_path = 'ImagePath' if 'ImagePath' in batch.keys() else 'FlowPath'
        self.write_results(batch[key_path], outputs, trainer.current_epoch, step_label)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'test')

    def on_test_end(self, trainer, pl_module) :
        self.summary_path = self.fp.replace('.csv', '_summary.tsv')
        dfr = pd.read_csv(self.fp)
        dfr['sequence'] = dfr['file_name'].apply(lambda x : x.split('/')[-2])
        dfr['dataset'] = dfr['file_name'].apply(lambda x : x.split('/')[0])
        dfr.groupby('sequence').mean().to_csv(self.summary_path, sep='\t')
        self.summary_log = dfr.mean()
        self.summary_log.to_csv(self.summary_path, sep='\t', mode='a', header=False)
        print(f'Summary saved at : {self.summary_path}')
