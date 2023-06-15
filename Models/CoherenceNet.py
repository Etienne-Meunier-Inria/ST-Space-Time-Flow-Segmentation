import warnings, einops, flowiz, sys, torch, math, os,\
       json, numpy as np, matplotlib.pyplot as plt
from argparse import ArgumentParser
from ipdb import set_trace
from Models.LitSegmentationModel import LitSegmentationModel
sys.path.append('..')
from utils.evaluations import *
from utils.distance_metrics import VectorDistance
from ShapeChecker import ShapeCheck
from utils.ParamShell import ParamShell
from Models.Regularisation import Regularisation

class CoherenceNet(LitSegmentationModel) :

    def __init__(self, img_size, len_seq, binary_method,
                 coherence_loss, v_distance,
                 param_model, flows_volume,
                 regularisation={},**kwargs) :
        super().__init__(**kwargs) # Build Lit Segmentatin Model
        self.request.add('Flow')
        self.init_param_model(img_size, len_seq, param_model)
        self.regularisation = regularisation
        self.flows_volume = flows_volume
        self.hparams.update({'img_size':img_size,
                             'len_seq': len_seq,
                             'flows_volume' : flows_volume,
                             'binary_method': binary_method,
                             'coherence_loss': coherence_loss,
                             'v_distance':v_distance,
                             'regularisation':regularisation,
                             'param_model':param_model})
        self.setup_binary_method(binary_method)
        self.coherence_loss = coherence_loss
        self.vdist = VectorDistance(v_distance)
        self.request |= set(self.flows_volume)

    def init_param_model(self, img_size, len_seq, param_model) :
        """Initialise ParamShell with the given model and the Grid for training

        Parameters
        ----------
            dims (int) : dimension input flow (T, I, J) or (I, J)
            param_model (str): Type of parametric model to generate (Affine /  Quadratic )
        Returns
        -------
            grid : Features for regression (T, I, J, 2, ft) depending on parametric model
        """
        print(f'Dimensions parametric model : {img_size} {len_seq}')
        T, I, J = len_seq, *img_size
        self.ps = ParamShell(param_model)
        self.register_buffer('grid', self.ps.build_grid((T, I, J)))
        self.ft = self.grid.shape[-1]

    def _flow_volume(self, batch) :
        """
        Based on the attribute self.flow_volume stack the flow fields in
        temporal dimension to get a volume representation of the flow
        for evaluation
        Args :
            batch with 'Flow' keys
        Returns :
            flowv( b c t h w) : Flows on which the loss is applied
        """
        flowv = einops.rearrange([batch[k] for k in self.flows_volume], 't b c h w -> b c t h w')
        return flowv

    def Criterion(self, batch) :
        """
        Compute the coherence loss of the masks / preds depending on the given flow
        Params :
            Batch containing at least :
                pred (b, L, T(opt), I, J) : Mask proba predictions
                flow ( b, 2, T(opt), I, J) : Original Flow Map ( before data augmentation )
        Returns :
            evals dictionnary with :
                losses (b) : loss for each prediction of the batch ( coherence + entropy )
            parametric_flow (b 2, T(opt), I J) :  Piecewise parametric flow as a composition of layers
        """
        sc = ShapeCheck(batch['PredV'].shape, 'b l t i j')
        assert  sc.reduce(batch['PredV'], 'b l t i j -> b t i j', 'sum').allclose(torch.tensor([1.0],\
                          device=batch['PredV'].device)),'Prediction must sum to 1 for the masks'
        # Optimzation in Theta
        batch['FlowV'] = self._flow_volume(batch)

        param_flos = self.ComputeParametricFlow(batch)

        # Compute Loss for optimisation of the network
        coherence_losses = self.CoherenceLoss(self.coherence_loss, self.vdist, param_flos,
                                              batch['PredV'], batch['FlowV']) # [b]
        regularisation_losses, reg_dict = self.RegularisationLoss(batch) # [b]
        losses = coherence_losses + regularisation_losses # [b]

        parametric_flow = self.ps.reconstruct_flow(batch['PredV'], param_flos)

        # Reconstruction for plotting
        return {'losses' : losses,
                'regularisation_losses' : regularisation_losses,
                'coherence_losses':coherence_losses, **reg_dict}, parametric_flow

    @staticmethod
    def CoherenceLoss(name, vdist, param_flow, pred, flow) :
        """
        Compute and return the coherence loss of the model given theta and the batch.
        Params :
            param_flo (b, l, 2, T(opt), I, J) : parametric flow for each layer
            pred (b, L, T(opt), I, J) : Mask proba predictions
            flow (b, 2, T(opt), I, J) : Original Flow Map
        Return  :
            coherence_losses (b): coherence loss for each sample of the batch
        """
        if pred.ndim == 4 and flow.ndim == 4 :
            param_flow = einops.repeat(param_flow, 'b l c i j -> b l c t i j', t = 1)
            pred =  einops.repeat(pred, 'b l i j -> b l t i j', t = 1)
            flow =  einops.repeat(flow, 'b c i j -> b c t i j', t = 1)

        sc = ShapeCheck([2], 'c')
        sc.update(param_flow.shape, 'b l c t i j')
        sc.update(pred.shape, 'b l t i j')
        sc.update(flow.shape, 'b c t i j')

        if 'pieceFit'  in name :
            flr = sc.repeat(flow, 'b c t i j -> b l t i j c')
            rsc = sc.rearrange(param_flow, 'b l c t i j -> b l t i j c')
            coherence_losses = sc.reduce(pred * vdist(rsc, flr), 'b l t i j -> b t i j', 'sum')
            if 'pieceFitScale' == name :
                fl = sc.repeat(flow, 'b c t i j -> b t i j c')
                mu_norm = sc.reduce(vdist(fl, torch.zeros_like(fl)), 'b t i j -> b t 1 1', 'mean')
                coherence_losses /= mu_norm
            coherence_losses = sc.reduce(coherence_losses, 'b t i j -> b', 'mean') #Original
        return coherence_losses

    def RegularisationLoss(self, batch) :
        """
        Compute and return the coherence loss of the model given theta and the batch.
        Params :
            Batch containing at least :
                theta : parameters set for each layers and sample : (b, L, ft)
                pred (b, L, T, I, J) : Mask proba predictions
                flow (b, 2, T, I, J) : Original Flow Map
        Return  :
            regularisation_losses (b): coherence loss for each sample of the batch
        """
        sc = ShapeCheck([2], 'c')
        sc.update(batch['PredV'].shape, 'b l t i j')
        sc.update(batch['FlowV'].shape, 'b c t i j')

        regularisator =  Regularisation()
        reg_dict = {}
        regularisation_losses = torch.zeros(sc.get('b')['b'], device=batch['PredV'].device)
        for k, v in self.regularisation.items() :
            l = regularisator.loss(name=k, batch=batch)
            if v != 0 :
                regularisation_losses += v*l
            reg_dict[k] = l.detach()
        return regularisation_losses, reg_dict

    def setup_binary_method(self, binary_method) :
        """
        Add to the request necessary inputs for the requested binary methods
        """
        if binary_method in ['fair'] : self.request.add('GtMask')
        self.binary_method = binary_method


    def generate_binary_mask(self, batch):
        """
        Takes as input a batch of propability masks and return a binary mask of the foreground pixels for evaluation
            - 'smallest' : choose the smallest mask as foreground
            - 'optimal' : segment mask using p > 0.5, select the segments to increase Jac score
            - 'optimax' : segments masks using argmax(p), select the segments to increase Jac score
            - 'fair' : segment masks using argmax(p), select segments that overlap with foreground for more than half their pixels
        Args :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions with sofmax over dim=1
        Returns : None but add key 'PredMask' binary (b, I, J) to batch
        """
        b, L, I, J = batch['Pred'].shape
        if self.binary_method == 'exceptbiggest' :
            b, L, I, J = batch['Pred'].shape
            argmax = batch['Pred'].argmax(1)
            idxmax = argmax.flatten(1).mode().values
            batch['PredMask'] = (argmax != idxmax[:,None, None].repeat(1,argmax.shape[1], argmax.shape[2]))
        elif self.binary_method == 'fair' :
            assert len(torch.unique(batch['GtMask'])) == 2, 'GtMask have to be binary fair'
            argmax = batch['Pred'].argmax(1, keepdims=True)
            binmax = torch.zeros_like(batch['Pred'])
            binmax.scatter_(1, argmax, 1)
            spgtmask = batch['GtMask'][:,None].repeat_interleave(L, dim=1)
            si = ((binmax *spgtmask).sum(axis=(2,3)) / binmax.sum(axis=(2,3))) > 0.5
            batch['PredMask'] = (binmax * si[:,:,None,None]).sum(axis=1)

    def custom_figs(self, ax, batch, evals) :
        """
        Generate figures depending on the model
        ax : list of axes to add the figure to
        batch : images and modalities
        evals : evaluation metrics dict
        """
        sh = lambda x : flowiz.convert_from_flow(x.detach().cpu().permute(1,2,0).numpy())
        ax[0,2].set_title('ParametricFlow')
        ax[0,2].imshow(sh(batch['ParametricFlow']))
        return ax

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--binary_method', help='Method to use to produce binary masks', type=str,
                            choices=['fair', 'exceptbiggest'], default='exceptbiggest')
        parser.add_argument('--coherence_loss', help='Computation of coherence loss', type=str,
                            choices=['pieceFit', 'pieceFitScale'], default='pieceFitScale')
        parser.add_argument('--regularisation', type=json.loads,
                            help='Regularisation to use, provide as "{reg_name:reg_weight, reg2_name:reg2_weight, ... }',
                            default={'EulerPottsl1R3':1})
        parser.add_argument('--v_distance', help='Vector distance metric to use in the computation', type=str,
                            choices=['squared', 'l2', 'l1'],
                            default='l1')
        parser.add_argument('--flows_volume', nargs='+', type=str, default=['Flow-1', 'Flow', 'Flow+1'],
                            help='Flow fields to build the flow volume for coherence loss')
        parser.add_argument('--param_model', help='Model to use to fit and compute the parametric flow',
                            type=str, default='QuadraticFullPT')
        parser.add_argument('--len_seq', help='Temporal length of the sequence to use for criterion',
                            type=int, default=3)
        return parser
