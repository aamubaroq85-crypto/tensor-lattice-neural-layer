import streamlit as st
import torch
import torch.nn as nn
from tlnl import LatticeLinear

# --- 1. KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="TLNL - Tensor Lattice Neural Layer",
    page_icon="⚡",
    layout="wide"
)

# --- 2. DEFINISI ARSITEKTUR MODEL JARINGAN SARAF ---
class TLNLClassificationModel(nn.Module):
    """
    Contoh arsitektur PyTorch yang menggabungkan LatticeLinear
    sebagai drop-in replacement untuk nn.Linear standar.
    """
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        # Layer 1: Menggunakan LatticeLinear dari library PyPI
        self.layer1 = LatticeLinear(in_features=in_features, out_features=hidden_features)
        self.relu = nn.ReLU()
        # Layer 2: Output layer
        self.layer2 = LatticeLinear(in_features=hidden_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# --- 3. ANTARMUKA UTAMA ---
st.title("⚡ Tensor Lattice Neural Layer (TLNL)")
st.markdown("""
Aplikasi demo interaktif untuk menguji performa dan integrasi paket **`tensor-lattice-neural-layer`** 
yang dipublikasikan di PyPI.
""")

st.divider()

# --- 4. SIDEBAR KONTROL PARAMETER ---
st.sidebar.header("⚙️ Konfigurasi Tensor")

input_dim = st.sidebar.number_input(
    "Jumlah Input Features (Dimensi Input)", 
    min_value=8, max_value=2048, value=512, step=64
)
hidden_dim = st.sidebar.number_input(
    "Jumlah Hidden Features (Hidden Layer)", 
    min_value=8, max_value=1024, value=128, step=32
)
num_classes = st.sidebar.number_input(
    "Jumlah Kelas Output", 
    min_value=2, max_value=100, value=10, step=1
)
batch_size = st.sidebar.number_input(
    "Batch Size", 
    min_value=1, max_value=256, value=32, step=8
)

# --- 5. EKSEKUSI DAN DOKUMENTASI ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🧪 Uji Coba Forward Pass")
    
    if st.button("🚀 Jalankan Simulasi Model", type="primary"):
        try:
            # Inisialisasi model
            model = TLNLClassificationModel(
                in_features=input_dim, 
                hidden_features=hidden_dim, 
                num_classes=num_classes
            )
            
            # Buat tensor input acak (dummy data)
            dummy_x = torch.randn(batch_size, input_dim)
            
            # Eksekusi model (forward pass)
            output = model(dummy_x)
            
            # Tampilkan indikator sukses
            st.success("✅ Forward Pass Berhasil Dieksekusi!")
            
            # Matriks ukuran tensor
            m1, m2, m3 = st.columns(3)
            m1.metric("Input Shape", str(list(dummy_x.shape)))
            m2.metric("Hidden Shape", f"[{batch_size}, {hidden_dim}]")
            m3.metric("Output Shape", str(list(output.shape)))
            
            # Pratinjau Nilai Tensor Output (3 data pertama)
            with st.expander("🔍 Lihat Detail Tensor Output (Raw Values)"):
                st.dataframe(output.detach().numpy()[:3])
                
        except Exception as e:
            st.error(f"❌ Terjadi Kesalahan Eksekusi: {e}")

with col_right:
    st.subheader("📝 Cara Menggunakan di Kode Python")
    st.code(f"""
import torch
from tlnl import LatticeLinear

# 1. Deklarasi Layer
layer = LatticeLinear(
    in_features={input_dim}, 
    out_features={hidden_dim}
)

# 2. Buat Input Tensor
x = torch.randn({batch_size}, {input_dim})

# 3. Jalankan Output
output = layer(x)

print("Output shape:", output.shape)
    """, language="python")

st.divider()

# --- 6. FOOTER INFORMASI PAKET ---
st.markdown("""
### 📦 Detail Instalasi Paket
Untuk menggunakan modul ini di lingkungan lokal Anda, pasang langsung melalui pip:
```bash
pip install tensor-lattice-neural-layer
