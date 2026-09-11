import time
import torch

def measure_vram_and_latency(model: torch.nn.Module, input_tensor: torch.Tensor):
    if not torch.cuda.is_available():
        return {"error": "CUDA tidak tersedia"}

    model = model.cuda()
    input_tensor = input_tensor.cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    latency_ms = (time.time() - start_time) * 1000

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "vram_allocated_mb": round(peak_vram_mb, 2),
        "latency_ms": round(latency_ms, 2)
    }
