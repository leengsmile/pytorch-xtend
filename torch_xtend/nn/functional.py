from typing import Union

from torch import Tensor
from torch.nn import Parameter


def swish(data: Tensor, beta: Union[float, Parameter]) -> Tensor:
    z = data * (beta * data).sigmoid()
    return z
