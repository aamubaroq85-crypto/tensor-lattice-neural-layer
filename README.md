# ⚡ TLNL: Tensor Lattice Neural Layer

[![PyPI version](https://img.shields.io/pypi/v/tensor-lattice-neural-layer.svg)](https://pypi.org/project/tensor-lattice-neural-layer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TLNL** adalah paket *drop-in replacement layer* untuk PyTorch yang dirancang untuk mengoptimalkan efisiensi memori (VRAM) serta mempercepat waktu inferensi (latensi) pada arsitektur Deep Learning.

## 🚀 Instalasi

```bash
pip install tensor-lattice-neural-layer
import torch
from tlnl import LatticeLinear

# Menggantikan nn.Linear standar dengan LatticeLinear
layer = LatticeLinear(in_features=512, out_features=128)
x = torch.randn(32, 512)
output = layer(x)
