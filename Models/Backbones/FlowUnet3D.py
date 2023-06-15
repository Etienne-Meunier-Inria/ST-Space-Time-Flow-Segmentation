from Models.Backbones.GeneralUnet.unet_clean import UNetClean
from Models.Backbones.GeneralUnet.u3D_3D import Unet3D_3D
import torch.nn as nn
import einops
import torch
from utils.ExperimentalFlag import ExperimentalFlag as Ef
from ipdb import set_trace
from torchvision.transforms import RandomErasing


class FlowUnet3D(nn.Module) :
    def __init__(self, base_net, inputs, num_classes, hparams) :
        super().__init__()
        self.inputs = inputs
        self.model = base_net(input_channels=2, num_classes=num_classes, num_layers=hparams['unet.num_layers'],
                               features_start=hparams['unet.features_start'], padding_mode=hparams['unet.padding_mode'],
                               inner_normalisation=hparams['unet.inner_normalisation'], train_bn=hparams['unet.train_bn'])

    def forward(self, batch) :
        """
        Args :
            batch dict 'Flow' (b c h w): dictionnary with input keys ( Flow at different timesteps )
        Returns :
            predv ( b l t h w) : Preds on which the loss is applied
            target_index : index of the target frame in the temporal dimension
            Add to batch :
                HiddenV (unet.num_layers b ftd td hd wd) : List hidden representation at the bottleneck

        """
        input = einops.rearrange([batch[k] for k in self.inputs], 't b c h w -> b c t h w')
        if Ef.check('PerturbInputFlowNoise') :
            #print('PerturbInputFlowNoise')
            input = perturb_flow_noise(input)
        pred, auxs = self.model(input)
        batch.update(auxs)
        pred = torch.softmax(pred, axis=1)
        batch['InputV'] = input.detach()
        return pred, self.inputs.index('Flow')

def perturb_flow_noise(flowv) :
    if torch.rand(1)[0] > 0.8:
        idx = torch.randint(flowv.shape[2], (1,))[0]
        flowv[:,:,idx] = torch.randn_like(flowv[:,:,idx]) * flowv[:,:,idx].var(axis=(2,3), keepdims=True) +\
                         flowv[:,:,idx].mean(axis=(2,3), keepdims=True)
    return flowv
