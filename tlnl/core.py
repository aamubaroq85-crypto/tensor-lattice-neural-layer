import math
import torch
import torch.nn as nn

PI_EFF_BASE = 3.141592653589793

class LatticeTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, pi_eff=PI_EFF_BASE):
        ctx.save_for_backward(input_tensor)
        scaled_tensor = input_tensor * (pi_eff / math.pi)
        transformed = torch.sin(scaled_tensor) * torch.tanh(input_tensor)
        return transformed

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone() * torch.cos(input_tensor)
        return grad_input, None

def apply_lattice_transform(tensor: torch.Tensor, pi_eff: float = PI_EFF_BASE) -> torch.Tensor:
    return LatticeTransform.apply(tensor, pi_eff)
