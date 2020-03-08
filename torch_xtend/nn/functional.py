from typing import Union

from torch import Tensor
from torch.nn import Parameter
from ._functional import SwishFunction


def swish(data: Tensor, beta: Union[float, Tensor]) -> Tensor:
    z = data * (beta * data).sigmoid()
    return z

# swish = SwishFunction.apply
