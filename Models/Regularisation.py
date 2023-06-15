import torch, math
from ipdb import set_trace
from ShapeChecker import ShapeCheck

class MetaRegFunction :
    @classmethod
    def get_requests(cls) :
        return {}

class EulerPottsl1R3(MetaRegFunction) :
    @classmethod
    def __call__(cls, sc, PredV, FlowV, **kwargs) :
        eu_loss = sc.reduce(torch.abs(PredV[:,:,:-1] - PredV[:,:,1:]), 'b l tn i j -> b tn i j', 'mean')
        tempo_flow = sc.reduce(torch.abs(FlowV[:,:,:-1] - FlowV[:,:,1:]), 'b c tn i j -> b tn i j', 'sum')
        qt = torch.quantile(sc.rearrange(tempo_flow, 'b tn i j -> b (tn i j)'), 0.99, dim=1)[:, None, None, None]
        return sc.reduce(eu_loss * (tempo_flow < qt), 'b tn i j -> b', 'mean')

class Regularisation :
    def __init__(self) :
        self.regs = {'EulerPottsl1R3':EulerPottsl1R3()}
        self.sc = ShapeCheck([2], 'c')

    def loss(self, name, batch) :
        self.sc.update(batch['Theta'].shape, 'b l ft')
        self.sc.update(batch['PredV'].shape, 'b l t i j')
        self.sc.update(batch['FlowV'].shape, 'b c t i j')
        return self.regs[name](self.sc, **batch)
