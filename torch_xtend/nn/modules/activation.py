from typing import Union
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module

from .. import functional as _F


class Swish(Module):

    __constants__ = ["beta", "trainable"]

    def __init__(self, beta: float = 1.0, trainable: bool = False, inplace: bool = False):
        super().__init__()
        self.beta = torch.tensor(beta)
        self.trainable = trainable
        self.inplace = inplace
        if self.trainable:
            self.beta = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def forward(self, input):
        return _F.swish(input, self.beta)
