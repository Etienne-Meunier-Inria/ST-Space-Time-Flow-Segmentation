import torch, einops
from torchvision import transforms as T
from ipdb import set_trace
from argparse import ArgumentParser

from ..utils.KeyTypes import extract_ascending_list
from utils.ParamShell import ParamShell

class TrAugmentFlow() :
    """
    Data augmentation techniques for optical flow fields
    Augments all 'Flow' fields in the sequence ( with )

    Args :
        Name flow_augmentation (list str) : data augmentation to return
    """
    def __init__(self, flow_augmentation) :
        self.augs = []
        for flow_augmentation_name in flow_augmentation :
            self.interpret_name(flow_augmentation_name)
        self.declare()

    def interpret_name(self, name) :
        if 'globalmotion' in name:
            self.augs.append(GlobalMotion(name))
        elif (name == 'none') or (name=='') :
            pass
        else :
            raise Exception(f'Flow augmentation {name} is unknown')

    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        ret : dict with keys 'Flow-1' ( 2 I J)
         1. We assemble all ret into a flow volumet ( 2 T I J) so we can treat them all at once
         2. Apply augmentation on the flow volume
         3. Split the volume into chunks and redistribute the flow
         BEWARE : in this function we use a "default" order for the time given in the key dict
         above. So the volume will be ordered following this. This is important for cases where
         the flow augmentation vary with time.
        """
        if len(self.augs) > 0 :
            # Get all keys both in ret and with type flow in the order defined in KEY_TYPES
            flows_keys = extract_ascending_list(ret.keys(), 'Flow')
            if len(flows_keys) > 0 :
                flow_volume = einops.rearrange([ret[k] for k in flows_keys], 't c i j -> c t i j')

                # Apply augmentations to the flow volume
                for aug in self.augs :
                    flow_volume = aug(flow_volume)

                # Redistribute flow to their respective spot in the dict
                for i in range(len(flows_keys)) :
                     ret[flows_keys[i]] = flow_volume[:, i]
        return ret

    def declare(self):
        print(f'Flow Transformations : {[aug for aug in self.augs]}')


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--flow_augmentation', type=str,
                            default=['globalmotion.QuadraticFullPT.light'], nargs='*')
        return parser

class GlobalMotion :
    def __init__(self, desc) :
        self.desc = desc

        _, name_model, alpha = desc.split('.')
        self.ps  = ParamShell(name_model)

        alphas = {'light' : 2, 'medium' : 4, 'strong' : 8}
        self.alpha = alphas[alpha]

        self.declare()

    def declare(self) :
        print(f'Augmentation {self.desc} model : {self.ps} alpha : {self.alpha}')

    def __call__(self, flow_volume) :
        """"
        Apply the transformation to the flow
        flow_volume (2 T I J)

        return :
            flow_volume + param_flo : ( 2 T I J)
        """
        param_flo = self.get_param_flow(flow_volume) # Take the first one to get the norm
        m =  torch.rand(1)
        return flow_volume + m*param_flo

    def get_param_flow(self, flow_volume) :
        """
        Get Parametric Flow with same average magnitude as input flow
        """
        param_flo = self.globalmotion(flow_volume.shape, self.ps)
        mf = torch.sqrt((flow_volume ** 2).sum(axis=0)).mean().to(param_flo.device)
        mp = torch.sqrt((param_flo  ** 2).sum(axis=0)).mean()
        return param_flo * (mf / (mp + 1e-6)) * self.alpha

    @staticmethod
    def globalmotion(shape, ps) :
        """
        Add a global motion to the flow field
        Args :
          shape [] : (2, T, I, J) of the parametric flow to produce
          ps : ParameterShell instance
        Return :
          param_flo : (2, T, I, J) parametric flow to add
        """
        _, T, I , J = shape
        grid = ps.build_grid((T, I, J))
        theta = torch.rand(1, 1, grid.shape[-1]) - 0.5
        param_flo = ps.parametric_flows(grid, theta)[0,0]
        return param_flo
