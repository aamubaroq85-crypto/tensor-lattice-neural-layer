# Tensor Lattice Neural Layer (TLNL)

A lightweight and high-performance lattice-based neural network layer for PyTorch, designed to optimize VRAM footprint and accelerate inference latency in Deep Learning architectures.

## 🚀 Instalasi

```bash
pip install tensor-lattice-neural-layer
import torch
from tlnl import LatticeLinear

# Menggantikan nn.Linear standar
layer = LatticeLinear(in_features=512, out_features=128)
x = torch.randn(32, 512)
output = layer(x)
