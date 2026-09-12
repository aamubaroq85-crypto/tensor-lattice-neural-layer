# Tensor Lattice Neural Layer (TLNL)

**Tensor Lattice Neural Layer (TLNL)** adalah ekstensi PyTorch open-source yang menyediakan `LatticeLinear`—sebuah *drop-in replacement* performa tinggi untuk `nn.Linear` yang dioptimalkan untuk efisiensi pemetaan tensor dan latensi inferensi.

## 🚀 Fitur Utama
* **Drop-in Replacement**: Kompatibel penuh dengan antarmuka `nn.Linear` standar PyTorch, sehingga dapat langsung menggantikan lapisan linear yang ada tanpa mengubah arsitektur model secara keseluruhan.
* **Latensi Lebih Cepat**: Pengujian benchmark pada GPU NVIDIA T4 menunjukkan peningkatan kecepatan inferensi hingga **~12.90%** dibandingkan `nn.Linear` standar.
* **Efisiensi Memori**: Dioptimalkan untuk stabilitas eksekusi berbasis PyTorch core untuk menjaga overhead VRAM tetap minimal.

## 📦 Instalasi
Instal langsung via PyPI:
```bash
pip install tensor-lattice-neural-layer
import torch
import torch.nn as nn
from tensor_lattice_neural_layer import LatticeLinear

# Mengganti nn.Linear dengan LatticeLinear secara instan
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = LatticeLinear(in_features=512, out_features=256)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc1(x))

model = SimpleModel().cuda()
x = torch.randn(32, 512, device='cuda')
output = model(x)
print(output.shape)
