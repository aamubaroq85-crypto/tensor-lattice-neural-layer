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
    def __init__(self, in_features: int, hidden_features: int, num_classes: int):
        super().__init__()
        self.layer1 = LatticeLinear(in_features=in_features, out_features=hidden_features)
        self.relu = nn.ReLU()
        self.layer2 = LatticeLinear(in_features=hidden_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# --- 3. ANTARMUKA UTAMA ---
st.title("⚡ Tensor Lattice Neural Layer (TLNL)")
st.write("Aplikasi demo interaktif untuk menguji performa dan integrasi paket **tensor-lattice-neural-layer** yang dipublikasikan di PyPI.")

st.divider()

# --- 4. SIDEBAR KONTROL PARAMETER ---
st.sidebar.header("⚙️ Konfigurasi Tensor")

input_dim = st.sidebar.number_input(
    "Jumlah Input Features", 
    min_value=8, max_value=2048, value=512, step=64
)
hidden_dim = st.sidebar.number_input(
    "Jumlah Hidden Features", 
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
            model = TLNLClassificationModel(
                in_features=input_dim, 
                hidden_features=hidden_dim, 
                num_classes=num_classes
            )
            
            dummy_x = torch.randn(batch_size, input_dim)
            output = model(dummy_x)
            
            st.success("✅ Forward Pass Berhasil Dieksekusi!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Input Shape", str(list(dummy_x.shape)))
            m2.metric("Hidden Shape", f"[{batch_size}, {hidden_dim}]")
            m3.metric("Output Shape", str(list(output.shape)))
            
            with st.expander("🔍 Lihat Detail Tensor Output"):
                st.dataframe(output.detach().numpy()[:3])
                
        except Exception as e:
            st.error(f"❌ Terjadi Kesalahan Eksekusi: {e}")

with col_right:
    st.subheader("📝 Cara Menggunakan di Kode Python")
    code_example = (
        "import torch\n"
        "from tlnl import LatticeLinear\n\n"
        f"# 1. Deklarasi Layer\n"
        f"layer = LatticeLinear(in_features={input_dim}, out_features={hidden_dim})\n\n"
        f"# 2. Buat Input Tensor\n"
        f"x = torch.randn({batch_size}, {input_dim})\n\n"
        f"# 3. Jalankan Output\n"
        f"output = layer(x)\n\n"
        'print("Output shape:", output.shape)'
    )
    st.code(code_example, language="python")

st.divider()

# --- 6. FOOTER INFORMASI PAKET ---
st.subheader("📦 Detail Instalasi Paket")
st.write("Untuk menggunakan modul ini di lingkungan lokal, pasang langsung melalui pip:")
st.code("pip install tensor-lattice-neural-layer", language="bash")
