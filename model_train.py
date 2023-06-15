from Models.CoherenceNet import CoherenceNet
from Models.CoherenceNets.MethodeB import MethodeB
from Models.LitSegmentationModel import LitSegmentationModel
from csvflowdatamodule.CsvDataset import CsvDataModule
from utils.Callbacks import ResultsLogger
from utils.ExperimentalFlag import ExperimentalFlag as Ef

import sys, torch, os, yaml
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
from datetime import datetime

exp_flags = yaml.safe_load(open('utils/ExperimentalFlags.yaml'))
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser = CoherenceNet.add_specific_args(parser)
parser = LitSegmentationModel.add_specific_args(parser)
parser = CsvDataModule.add_specific_args(parser)
parser.add_argument('--path_save_model', type=str, default=None)
parser.add_argument('--experimental_flag', default = ['PerturbInputFlowNoise'],
                    choices=list(exp_flags.keys()), nargs='*')
args = parser.parse_args()
Ef.set(args.experimental_flag)

pl.seed_everything(123)

#########################
##      Load Model     ##
#########################
model = MethodeB(**vars(args))


#########################
##      Load Data      ##
#########################
args.data_path  = args.data_file+'_{}.csv'
dm = CsvDataModule(request=model.request, **vars(args))


#########################
##   Model Checkpoint  ##
#########################
if args.path_save_model :
    path = Path(args.path_save_model)
    path.mkdir(exist_ok=True)
    # We save the model with the lowest validation loss.
    args.callbacks = [pl.callbacks.ModelCheckpoint(args.path_save_model+'/checkpoints/',
                                                   monitor='val/loss',
                                                   filename='{epoch}-epoch_val_loss:{val/loss:.5f}',
                                                   mode='min',
                                                   auto_insert_metric_name=False,
                                                   save_top_k=1),
                      ResultsLogger(args.path_save_model+'/results.csv')]


#########################
##        Logger       ##
#########################

# Configure the pytorch lightning logger of your choice here.
logger = pl.loggers.CSVLogger(args.path_save_model)
#logger.log_hyperparams(args)

#########################
##      Run Training   ##
#########################
trainer = pl.Trainer.from_argparse_args(args,  logger=logger)
trainer.fit(model, dm)
