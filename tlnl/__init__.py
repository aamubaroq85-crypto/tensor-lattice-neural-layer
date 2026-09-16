"""
TLNL - Tensor Lattice Neural Layer
Package pemodelan efisien VRAM berbasis geometri pi_eff.
"""

from .layers import LatticeLinear
from .core import apply_lattice_transform
from .utils import measure_vram_and_latency

__version__ = "0.1.0"
__author__ = "Baroq / Zuhri Formalism Team"

# Penambahan aman untuk menangani modul enterprise opsional (Open-Core)
try:
    from .enterprise_core import run_enterprise_computation
    __all__ = [
        "LatticeLinear",
        "apply_lattice_transform",
        "measure_vram_and_latency",
        "run_enterprise_computation",
    ]
except ImportError:
    __all__ = [
        "LatticeLinear",
        "apply_lattice_transform",
        "measure_vram_and_latency",
    ]
