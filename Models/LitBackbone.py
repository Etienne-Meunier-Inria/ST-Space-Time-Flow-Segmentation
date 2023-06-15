import torch, torch.nn as nn, pytorch_lightning as pl
from ipdb import set_trace
from argparse import ArgumentParser
from torchvision import models
from ipdb import set_trace

from Models.Backbones.FlowUnet3D import FlowUnet3D
from Models.Backbones.GeneralUnet.u3D_3D import Unet3D_3D
from Models.Backbones.GeneralUnet.unet_clean import UNetClean



class LitBackbone(pl.LightningModule):

    def __init__(self, inputs, backbone, num_classes, hparams, **kwargs) :
        super().__init__()
        self.inputs = inputs
        self.model = self.init_model(backbone = backbone,
                                     inputs = inputs,
                                     num_classes = num_classes, hparams=hparams)

    def init_model(self, *, backbone, num_classes, inputs, hparams) :
        if backbone == 'FlowUnet3D' :
            return FlowUnet3D(Unet3D_3D, inputs, num_classes, hparams)
        else :
            print(f'Backbone {backbone} not available')

    def forward(self, batch) :
        return self.model(batch)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = UNetClean.add_specific_args(parser)
        parser.add_argument('--backbone', '-bb', type=str, choices=['FlowUnet3D'], default='FlowUnet3D')
        parser.add_argument('--inputs', nargs='+', type=str, default=['Flow-1', 'Flow', 'Flow+1'])
        return parser
