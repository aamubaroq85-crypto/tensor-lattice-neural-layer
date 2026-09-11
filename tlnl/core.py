import math
import torch
import torch.nn as nn

PI_EFF_BASE = 3.141592653589793

@torch.jit.script
def _fast_lattice_transform(input_tensor: torch.Tensor, pi_eff: float) -> torch.Tensor:
    scaled = input_tensor * (pi_eff / 3.141592653589793)
    return torch.sin(scaled) * torch.tanh(input_tensor)

def apply_lattice_transform(tensor: torch.Tensor, pi_eff: float = PI_EFF_BASE) -> torch.Tensor:
    return _fast_lattice_transform(tensor, pi_eff)
