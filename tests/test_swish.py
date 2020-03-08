import unittest
import torch
from torch import Tensor
from torch.autograd import gradcheck
from torch_xtend.nn import Swish


class TestSwish(unittest.TestCase):

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

        auto_grad = input.grad.data
        manual_grad = d_swish(input)
        print("grad [auto]: ", auto_grad)
        print("grad [manual]: ", manual_grad)

        assert torch.allclose(auto_grad, manual_grad)

    def test_swish_trainable(self):

        def d_swish(input: Tensor, beta: float = 1.0):
            if input.requires_grad:
                input = input.clone().detach()
            s = (input * beta).sigmoid()
            z = s + input * s * (1. - s) * beta
            return z

        input = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        swish = Swish(trainable=True)
        output: Tensor = swish(input)
        output.sum().backward()

        auto_grad = input.grad.data
        manual_grad = d_swish(input)
        print("grad [auto]: ", auto_grad)
        print("grad [manual]: ", manual_grad)
        print("swish grad: ", swish.beta.grad)

        assert torch.allclose(auto_grad, manual_grad)

    def test_swish_internal_function(self):

        from torch_xtend.nn._functional import SwishFunction
        swish = SwishFunction.apply
        torch.manual_seed(1)
        input = (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.randn(10, dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.randn(10, dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double, requires_grad=True))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double, requires_grad=True))
        gradcheck(swish, inputs=input)

    def test_swish_function(self):

        from torch_xtend.nn.functional import swish
        torch.manual_seed(1)
        input = (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.randn(10, dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.randn(10, dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double, requires_grad=True))
        gradcheck(swish, inputs=input)

        torch.manual_seed(1)
        input = (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double, requires_grad=True),
                 torch.tensor(1.0, dtype=torch.double, requires_grad=True))
        gradcheck(swish, inputs=input)

    def test_swish_extra_repr(self):

        swish = Swish()
        print(swish)