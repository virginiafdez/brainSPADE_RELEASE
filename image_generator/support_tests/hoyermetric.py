import torch
import numpy as np

'''
Calculates the sparsity metric for a set of vectors. 
The sparsest the vector, the closest to 1 the Hoyer Metric.
Source: Niall P. Hurley and Scott T. Rickard. Comparing measuresof sparsity.IEEE Transactions on Information Theory,55:4723–4741, 2008
'''
class HoyerMetric:
    def __init__(self, dimension):
        '''
        Definition of an Hoyer Metric dimension
        Args:
            dimension: Dimension of the latent space
        '''
        self.dimension = dimension

    def __call__(self, y_set, reduce = True, normalize = False):
        '''
        Computes the average Hoyer Metric over a dataset
        Args:
            y_set: Tensor N x D where D is the dimension of the latent space and N the dataset length
        Returns: Average Hoyer Metric
        '''

        if normalize:
            y_set = self.normalize(y_set, 1)
        h = (torch.sqrt(torch.tensor([self.dimension]).type(torch.float32)) -
             self.l1_norm(y_set, 1)/self.l2_norm(y_set, 1)) / (torch.sqrt(
            torch.tensor([self.dimension]).type(torch.float32)) - 1)

        if reduce:
            return h.mean()
        else:
            return h

    def normalize(self, y_set, *args):
        if args:
            dim_ = args[0]
        else:
            dim_ = 0
        maxs = torch.max(y_set, dim_)[0].repeat(y_set.shape[1], 1).transpose(0,1)
        mins = torch.min(y_set, dim_)[0].repeat(y_set.shape[1], 1).transpose(0,1)
        out =  (y_set - mins)/(maxs-mins)
        out[torch.isnan(out)] = 0.0
        return out


    def l1_norm(self, y, *args):
        if args:
            dim_ = args[0]
        else:
            dim_ = 0
        return abs(y.sum(dim_))

    def l2_norm(self, y, *args):
        if args:
            dim_ = args[0]
        else:
            dim_ = 0

        return torch.sqrt((y**2).sum(dim_))
