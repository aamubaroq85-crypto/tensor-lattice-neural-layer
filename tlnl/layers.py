import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import PI_EFF_BASE

class LatticeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, pi_eff: float = PI_EFF_BASE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pi_eff = pi_eff
        
        # Inisialisasi parameter
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lakukan perkalian linier dulu, baru transformasi pada ruang output yang lebih kecil
        out = F.linear(x, self.weight, self.bias)
        scaled = out * (self.pi_eff / 3.141592653589793)
        return torch.sin(scaled) * torch.tanh(out)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, pi_eff={self.pi_eff}'
