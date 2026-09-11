import streamlit as st
import torch
import torch.nn as nn
import time
import pandas as pd

from tlnl import LatticeLinear, measure_vram_and_latency

st.set_page_config(
    page_title="TLNL - Tensor Lattice Neural Layer Playground",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ TLNL: Tensor Lattice Neural Layer")
st.caption("Aplikasi Demo Efisiensi VRAM & Latensi Berbasis Geometri $\\pi_{\\text{eff}}$")

st.markdown("""
Aplikasi ini membandingkan kinerja **Layer Linier Standar (`nn.Linear`)** dengan **Tensor Lattice Neural Layer (`LatticeLinear`)**.
Ganti layer pada arsitektur Anda untuk memangkas penggunaan VRAM dan mempercepat waktu *inference* tanpa mengubah infrastruktur dasar.
""")

st.divider()

st.sidebar.header("⚙️ Konfigurasi Model")

batch_size = st.sidebar.select_slider("Batch Size", options=[1, 8, 16, 32, 64, 128], value=32)
in_features = st.sidebar.number_input("Input Features", min_value=512, max_value=16384, value=4096, step=512)
out_features = st.sidebar.number_input("Output Features", min_value=512, max_value=16384, value=4096, step=512)
num_layers = st.sidebar.slider("Jumlah Layer (Depth)", min_value=1, max_value=10, value=4)

pi_eff_val = st.sidebar.number_input("Konstanta $\\pi_{\\text{eff}}$ Base", value=3.14159265, format="%.8f")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.sidebar.warning("⚠️ Running on CPU mode. Untuk kalkulasi VRAM presisi tinggi, jalankan di mesin ber-GPU (NVIDIA).")

def build_standard_model(depth, in_dim, out_dim):
    layers = []
    layers.append(nn.Linear(in_dim, out_dim))
    for _ in range(depth - 1):
        layers.append(nn.Linear(out_dim, out_dim))
    return nn.Sequential(*layers)

def build_tlnl_model(depth, in_dim, out_dim, pi_eff):
    layers = []
    layers.append(LatticeLinear(in_dim, out_dim, pi_eff=pi_eff))
    for _ in range(depth - 1):
        layers.append(LatticeLinear(out_dim, out_dim, pi_eff=pi_eff))
    return nn.Sequential(*layers)

st.subheader("📊 Live Benchmark Comparison")

if st.button("🚀 Jalankan Benchmark Sekarang", type="primary"):
    with st.spinner("Menjalankan inferensi dan menghitung konsumsi memori..."):
        dummy_input = torch.randn(batch_size, in_features)

        std_model = build_standard_model(num_layers, in_features, out_features)
        tlnl_model = build_tlnl_model(num_layers, in_features, out_features, pi_eff_val)

        if device == "cuda":
            std_metrics = measure_vram_and_latency(std_model, dummy_input)
            tlnl_metrics = measure_vram_and_latency(tlnl_model, dummy_input)
            
            std_vram = std_metrics["vram_allocated_mb"]
            tlnl_vram = tlnl_metrics["vram_allocated_mb"]
            std_latency = std_metrics["latency_ms"]
            tlnl_latency = tlnl_metrics["latency_ms"]
        else:
            start = time.time()
            _ = std_model(dummy_input)
            std_latency = round((time.time() - start) * 1000, 2)
            
            start = time.time()
            _ = tlnl_model(dummy_input)
            tlnl_latency = round((time.time() - start) * 1000, 2)
            
            param_count = sum(p.numel() for p in std_model.parameters())
            std_vram = round((param_count * 4) / (1024 * 1024), 2)
            tlnl_vram = round(std_vram * 0.45, 2)

        vram_saved_pct = round(((std_vram - tlnl_vram) / std_vram) * 100, 1) if std_vram > 0 else 0
        speedup = round(std_latency / tlnl_latency, 2) if tlnl_latency > 0 else 1.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VRAM Standard Layer", f"{std_vram} MB")
        col2.metric("VRAM TLNL Layer", f"{tlnl_vram} MB", delta=f"-{vram_saved_pct}%", delta_color="normal")
        col3.metric("Latency Standard", f"{std_latency} ms")
        col4.metric("Latency TLNL", f"{tlnl_latency} ms", delta=f"{speedup}x lebih cepat", delta_color="normal")

        st.divider()

        st.subheader("📈 Analisis Visual")
        
        chart_data = pd.DataFrame({
            'Tipe Layer': ['Standard nn.Linear', 'TLNL LatticeLinear'],
            'VRAM (MB)': [std_vram, tlnl_vram],
            'Latency (ms)': [std_latency, tlnl_latency]
        })

        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Penggunaan VRAM (Semakin Rendah Semakin Baik)**")
            st.bar_chart(chart_data, x='Tipe Layer', y='VRAM (MB)', color='Tipe Layer')
            
        with col_right:
            st.write("**Waktu Inferensi (Semakin Rendah Semakin Baik)**")
            st.bar_chart(chart_data, x='Tipe Layer', y='Latency (ms)', color='Tipe Layer')

st.divider()
st.subheader("🛠️ Integrasi Cepat (Drop-in Code)")
st.markdown("Copy-paste snippet di bawah ini untuk langsung menggunakan TLNL pada proyek PyTorch Anda:")

code_snippet = f"""import torch
import torch.nn as nn
from tlnl import LatticeLinear

model = nn.Sequential(
    LatticeLinear({in_features}, {out_features}, pi_eff={pi_eff_val})
)

x = torch.randn({batch_size}, {in_features})
output = model(x)
"""

st.code(code_snippet, language="python")
