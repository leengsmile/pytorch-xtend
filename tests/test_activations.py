import unittest
import numpy as np
import torch
from torch import Tensor
from torch_xtend.nn import Swish


class TestActivation(unittest.TestCase):

    def test_swish(self):

        def d_swish(input: Tensor, beta: float = 1.0):
            if input.requires_grad:
                input = input.clone().detach()
            s = (input * beta).sigmoid()
            z = s + input * s * (1. - s) * beta
            return z

        input = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        swish = Swish()
        output: Tensor = swish(input)
        output.sum().backward()

        auto_grad = input.grad.data.numpy()
        manual_grad = d_swish(input).numpy()
        print("grad [auto]: ", auto_grad)
        print("grad [manual]: ", manual_grad)

        np.testing.assert_allclose(auto_grad, manual_grad)

