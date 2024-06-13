import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

@weighted_loss
def sparsity_loss(masks, sparsity_target):
    loss = torch.tensor(.0).cuda()
    for i in range(len(masks)):
        for j in range(len(masks[0])):
            for k in range(len(masks[0][0])):
                if masks[i][j][k] is not None:
                    loss += masks[i][j][k]
    loss = loss / (len(masks)*(len(masks[0])-1)*len(masks[0][0]))
    return loss

@LOSS_REGISTRY.register()
class SparsityLoss(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound. 
    This loss is annealed.
    '''

    def __init__(self,sparsity_target):
        super(SparsityLoss, self).__init__()
        self.sparsity_target = sparsity_target
    def forward(self, masks):
        theta = 0.0001
        return theta*sparsity_loss(masks, self.sparsity_target) 