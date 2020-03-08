import torch
from torch import Tensor

from torch.autograd.function import Function


class SwishFunction(Function):

    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input, beta)
        output = input * (input * beta).sigmoid()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, beta = ctx.saved_tensors
        s = (input * beta).sigmoid()
        grad = s + input * s * (1. - s) * beta

        d_beta = None
        if ctx.needs_input_grad[1]:
            d_beta = input ** 2 * s * (1 - s)
            d_beta *= grad_output
        return grad * grad_output, d_beta

