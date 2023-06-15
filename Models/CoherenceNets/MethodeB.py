import cv2, sys, einops, torch, os, numpy as np
from argparse import ArgumentParser
from ipdb import set_trace
from functools import partial

from Models.CoherenceNet import CoherenceNet
from Models.LitSegmentationModel import LitSegmentationModel
from ShapeChecker import ShapeCheck

class MethodeB(CoherenceNet) :
    """
    Model using as criterion the coherence of the optical flow in segmented regions.
    """
    def __init__(self, **kwargs) :
        super().__init__(**kwargs) # Build Coherence Net

    def ComputeTheta(self, pred, flow):
        """
        General Method to compute theta

        Params :
            pred (b, L, T, I, J) : Mask proba predictions with optional time dimension
            flow ( b, 2, T, I, J) : Flow Map with optional time dimension
        Returns :
            Theta : parameters set for each layers and sample : (b, L, ft)
        """
        fctn = partial(self.CoherenceLoss, name=self.coherence_loss, vdist=self.vdist)
        return self.ps.computetheta_optim(self.grid, flow, pred, fctn)


    def ComputeParametricFlow(self, batch) :
        """
        For a given batch compute the parametric flow with the appropriate technique.
        Gradients on PredV are detached before Theta estimation.
        Params :
            Batch containing at least :
                pred (b, L, T(opt), I, J) : Mask proba predictions
                flow ( b, 2, T(opt), I, J) : Original Flow Map ( before data augmentation )
        Returns :
            Add to batch 'Theta' ( b, l, ft) to the batch with the parametric motion parameters.
            param_flos (b, l, c, T(opt), i, j) : parametric flow for each layer
        """
        batch['Theta'] = self.ComputeTheta(batch['PredV'].detach(), batch['FlowV']).detach()

        param_flos = self.ps.parametric_flows(self.grid, batch['Theta'])
        return param_flos
