from typing import Union
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module

from .. import functional as _F


class Swish(Module):
    r"""Apply the Swish unit function element-wise.

    :math:`\text{Swish}(x) = x \sigma (\beta x)`
    where :`\sigma(x)`: is the sigmoid function.

    Args:
        beta (float): the :math`\beta`: in Swish paper. Default: 1.0
        trainable (bool): whether beta is fixed or could be learned by model.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples:

        >>> m = Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

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

    def extra_repr(self):
        return f'beta={self.beta}, trainable={self.trainable}'

