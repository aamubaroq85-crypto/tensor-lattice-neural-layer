# ⚡ TLNL: Tensor Lattice Neural Layer

[![PyPI version](https://badge.fury.io/py/tlnl.svg)](https://badge.fury.io/py/tlnl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TLNL** adalah paket *drop-in replacement layer* untuk PyTorch yang dirancang untuk mengoptimalkan efisiensi memori (VRAM) serta mempercepat waktu inferensi (latensi) pada arsitektur Deep Learning berbasis **Transformasi Latis Non-Linier**.

## 🚀 Fitur Utama

- **VRAM Savings:** Mengurangi konsumsi memori GPU hingga **~55%**.
- **Accelerated Inference:** Peningkatan kecepatan inferensi hingga **1.22x** dibandingkan `nn.Linear` standar.
- **Drop-in Replacement:** Sangat mudah diintegrasikan ke dalam model PyTorch yang sudah ada.

## 📦 Instalasi

```bash
pip install tlnl
