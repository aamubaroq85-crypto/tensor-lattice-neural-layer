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
## 📊 Performance & Benchmarks

Benchmarked on **NVIDIA T4 GPU** (Google Colab Environment) with matrix dimensions of `in_features=4096`, `out_features=4096`, and `batch_size=64` over 100 inference passes.

| Metric | `torch.nn.Linear` | `TLNL (LatticeLinear)` | Difference |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | 0.904 ms | **0.788 ms** | **⚡ 12.90% Faster** |
| **Peak VRAM** | 208.17 MB | 212.17 MB | ~1.92% Overhead (v0.1.0) |

> **Key Takeaway:** `LatticeLinear` achieves a **~13% speedup in inference latency** out of the box due to optimized tensor mapping. VRAM optimization techniques (in-place operations and custom CUDA kernels) are planned for the `v0.2.0` release.

### Reproduce Benchmark
You can run the benchmark script directly in your CUDA-enabled environment:

```bash
python benchmark.py
