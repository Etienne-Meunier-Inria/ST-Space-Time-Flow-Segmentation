import torch

class VectorDistance() :

    def __init__(self, name) :
        """
        name : name of the distance metric to use
        """
        self.distances = {'squared' : self.squared_dist,
                          'l2' : self.l2_dist,
                          'l1' : self.l1_dist,
                          'charb' : self.charb_dist,
                          'l2_normed':self.l2_normed_dist,
                          'l2_normed_alt':self.l2_normed_alt_dist,
                          'squared_normed_alt':self.squared_normed_alt_dist}
        assert name in list(self.distances.keys()), f"No distance named {name}"
        self.name = name

    def __call__(self, f1, f2):
        """
        Compute the distance between two matrix of flow vectors using the metric requested
        f1, f2 (torch.tensor) : (..., 2) any size you want but the last have to be (u,v)
        returns :
            dist : (torch.tensor) : (...) the distance between each vectors in the input shape
        """
        assert f1.shape[-1] == f2.shape[-1], "Need to have the same number of components"
        return self.distances[self.name](f1, f2)

    @staticmethod
    def squared_dist(f1, f2) :
        return ((f1-f2)**2).sum(axis=-1)

    @staticmethod
    def l2_dist(f1, f2) :
        return torch.norm(f1-f2, dim=-1)

    @staticmethod
    def l1_dist(f1, f2) :
        return torch.norm(f1-f2, dim=-1, p=1)

    @staticmethod
    def charb_dist(f1, f2) :
        return (((f1-f2)**2 + 1e-8)**(0.5)).sum(axis=-1)

    @staticmethod
    def l2_normed_dist(f1, f2) :
        return torch.norm(f1-f2, dim=-1) / (torch.norm(f1, dim=-1) + torch.norm(f2, dim=-1) + 1e-10)

    @staticmethod
    def l2_normed_alt_dist(f1, f2) :
        """
        Divergence where we only divide by the norm of the second vector
        div(f1, f2) = norm(f1-f2) / norm(f1)
        """
        return torch.norm(f1-f2, dim=-1) / (torch.norm(f2, dim=-1) + 1e-10)

    @staticmethod
    def squared_normed_alt_dist(f1, f2) :
        """
        Divergence where we only divide by the norm of the second vector
        div(f1, f2) = (f1-f2)**2 / norml2(f2)
        """
        return ((f1-f2)**2).sum(axis=-1) / (torch.norm(f2, dim=-1) + 1e-10)
