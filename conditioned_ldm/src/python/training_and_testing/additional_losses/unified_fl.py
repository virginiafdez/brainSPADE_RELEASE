import torch
from monai.losses import TverskyLoss, FocalLoss

class Unified_FL(torch.nn.Module):

    def __init__(self, n_classes, delta = 0.6, gamma = 2, lambda_ = 0.75):
        super(Unified_FL, self).__init__()
        self.N = n_classes
        self.delta = delta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tversky = TverskyLoss(include_background=True, alpha = self.delta, beta = (1-self.delta),
                                   reduction="mean", softmax=True)
        self.focal = FocalLoss(include_background=True, gamma = self.gamma, reduction="mean")


    def forward(self, x, y):

        return self.lambda_ * self.delta * self.focal(x, y) + (1 - self.lambda_) * self.tversky(x, y)



