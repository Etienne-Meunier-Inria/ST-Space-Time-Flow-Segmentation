import torch, einops, re, math
from ipdb import set_trace
from ShapeChecker import ShapeCheck
from .ExperimentalFlag import ExperimentalFlag as Ef


class ParamShell :
    models = {}
    models['Affine'] =  ['i j 1'+' 0'*3,
                         '0 '*3 +'i j 1']

    models['AffinePT'] = ['t it jt i j 1'+' 0'*6,
                          '0 '*6 +'t it jt i j 1']

    models['QuadraticPlanar'] =  ['ij jj 0 0 0 i j 1',
                                  'ii ij i j 1 0 0 0']

    models['QuadraticPlanarPT'] =  ['ij ijt jj jjt 0 0  0 i  it j jt 1 t',
                                    'ii iit ij ijt i it j jt 1  t 0  0 0']
    models['QuadraticFull'] = ['ij ii jj i j 1'+' 0'*6,
                               '0 '*6 +'ij ii jj i j 1']

    models['QuadraticFullPT'] = ['t tij tii tjj ti tj ij  ii  jj  i j 1'+' 0'*12,
                                 '0 '*12 +'t tij tii tjj ti tj ij  ii  jj  i j 1']

    def __init__(self, model_name) :
        """
        Translate the model name into an argument sequence
        """
        self.model_name = model_name
        self.model_sequence = self.models[self.model_name]


    def __str__(self):
        return self.model_name

    def build_grid(self, shape) :
        """
        shape : shape of the grid to build ( t, i, j)
        Return a flat grid with the parameters :
            FlatGrid : ( c, t, i, j, ft), c=2 and ft depends on the model
        """
        T, I , J = shape

        t, i, j = torch.meshgrid(torch.linspace(-1, 1, T),
                                 torch.linspace(-1, 1, I),
                                 torch.linspace(-1, 1, J), indexing='ij')

        d = {'i' :i, 'j':j, 't':t,
             '0': torch.zeros_like(i),
             '1': torch.ones_like(j)}

        pm = [re.sub(' +', ' ', m).split(' ') for m in self.model_sequence]
        ft = len(pm[0])
        assert ft == len(pm[1]), f'Error in model sequence : {self.model_sequence}'

        Xn = torch.zeros((2, T, I, J, ft)) # N, C, ft

        for c in range(ft) :
            Xn[0, ..., c] = math.prod([d[k] for k in pm[0][c]])
            Xn[1, ..., c] = math.prod([d[k] for k in pm[1][c]])
        return Xn

    def parametric_flows(self, grid, theta) :
        """
        Computes a parametric flow of the requested shape with the given theta
        Args :
            grid : ( c, t, i, j, ft), c=2 and ft depends on the model
            theta : (b, l, ft) here we can have 2 leading dimension wich is a batch size and layers
        Return :
            param_flos (b, l, c, t, i, j) : parametric flow for each layer
        """
        sc = ShapeCheck(theta.shape, 'b l ft')
        sc.update(grid.shape, 'c t i j ft')

        theta = sc.rearrange(theta, 'b l ft -> (b l) ft 1')
        grid = sc.repeat(grid, 'c t i j ft -> (b l) (c t i j) ft')
        param_flos = sc.rearrange(torch.bmm(grid, theta), '(b l) (c t i j) 1 -> b l c t i j')
        return param_flos

    def reconstruct_flow(self, pred, param_flo) :
        """
        using parametric flow and prediction assemble a flow field by combining both
        Args :
            pred : weightning of different layers (b, l, t, i, j)
            param_flo (b, l, c, t, i, j) : parametric flow for each layer
        Return :
            reconstructed_flow (b, c, t, i, j) : flow combining all layers
        """
        sc = ShapeCheck(param_flo.shape, 'b l c t i j')
        sc.update(pred.shape, 'b l t i j')
        reconstructed_flow = sc.reduce(sc.repeat(pred, 'b l t i j -> b l c t i j') * param_flo, 'b l c t i j -> b c t i j', 'sum')
        return reconstructed_flow

    def get_random_theta(self) :
        """
        Return a random theta vector to generate a fake flow ( for data augmentation)
        Return :
            theta : (ft) parameters for each layers and element of the batch
        """
        theta = torch.rand(sl)

    def computetheta_ols(self, grid, flow, pred) :
        """
        Computes the parameter theta for this flow and weight map with vdist given
        Args :
            grid : (c, t, i, j, ft), c=2 and ft depends on the model
            flow : flow field (b, c, t, i, j)
            pred : weightning of different layers (b, l, t, i, j)
        Return :
            theta : ( b, l, ft) parameters for each layers and element of the batch
        """
        sc = ShapeCheck(pred.shape, 'b l t i j')
        sc.update(flow.shape, 'b c t i j')
        sc.update(grid.shape, 'c t i j ft')

        # Formulate preds to Weights
        pred = sc.repeat(pred, 'b l t i j -> (b l) (c t i j) 1')

        grid = sc.repeat(grid, 'c t i j ft -> (b l) (c t i j) ft')

        # Formulate Flow
        flow = sc.repeat(flow, 'b c t i j -> (b l) (c t i j) 1')

        try :
            wsq =  torch.sqrt(pred)
            Theta = torch.linalg.lstsq(grid*wsq, flow*wsq).solution

            Theta = sc.repeat(Theta, '(b l) ft 1 -> b l ft')

        except Exception as e:
           Theta = torch.zeros(tuple(sc.get('b l ft').values()),
                   device=pred.device, requires_grad=True)
           print(f'Inversion Error : {e} batch')
           set_trace()
        if torch.isnan(Theta).sum() > 0 :
            print('Nan in Theta')
        return Theta.nan_to_num(0) # Avoid nan in theta estimation

    def computetheta_optim(self, grid, flow, pred, functional) :
        """
        Compute Theta using optimisation and the Coherence Loss
        Args :
            grid : (c, t, i, j, ft), c=2 and ft depends on the model
            flow : flow field (b, c, t, i, j)
            pred : weightning of different layers (b, l, t, i, j)
            functional handle : function to minimize taking as input ( param_flow, pred, flow )
        Return :
            theta : ( b, l, ft) parameters for each layers and element of the batch
        """
        #print('Solve Optim')
        sc = ShapeCheck(pred.shape, 'b l t i j')
        sc.update(flow.shape, 'b c t i j')
        sc.update(grid.shape, 'c t i j ft')
        theta = self.computetheta_ols(grid, flow, pred).detach() # Init with OLS
        sc.update(theta.shape,'b l ft')
        theta.requires_grad_(True)
        lbgfs = torch.optim.LBFGS([theta], line_search_fn='strong_wolfe')
        def closure() :
            lbgfs.zero_grad()
            param_flow = self.parametric_flows(grid, theta)
            loss = functional(param_flow=param_flow, pred=pred.detach(), flow=flow).mean()
            loss.backward()
            return loss
        lbgfs.step(closure)
        theta = theta.detach() # Our theta is not supposed to have gradients after LBFGS

        return theta
