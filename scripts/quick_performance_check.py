#!/usr/bin/env python3
"""Quick performance check of gPAC."""

import time
import torch
import numpy as np
from gpac import calculate_pac

# Test parameters
fs = 1000
duration = 5
n_samples = int(fs * duration)
signal = np.random.randn(n_samples)

# Prepare signal
signal_4d = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

print("gPAC Performance Quick Check")
print("=" * 50)
print(f"Signal: {duration}s @ {fs}Hz = {n_samples} samples")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print()

# Move to GPU if available
if device.type == "cuda":
    signal_gpu = signal_4d.cuda()
    
    # Warmup
    _ = calculate_pac(signal_gpu, fs=fs, pha_n_bands=10, amp_n_bands=10)
    
    # Time GPU computation
    torch.cuda.synchronize()
    start = time.time()
    pac_gpu, _, _ = calculate_pac(
        signal_gpu, 
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=50,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=30
    )
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"GPU computation time: {gpu_time:.3f}s")
else:
    print("No GPU available")

# CPU computation
start = time.time()
pac_cpu, _, _ = calculate_pac(
    signal_4d, 
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=50,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=30,
    device="cpu"
)
cpu_time = time.time() - start

print(f"CPU computation time: {cpu_time:.3f}s")

if device.type == "cuda":
    print(f"\nGPU speedup: {cpu_time/gpu_time:.1f}x faster")

# Batch processing test
print("\nBatch Processing Test:")
batch_size = 16
batch_signal = torch.randn(batch_size, 1, 1, n_samples)

start = time.time()
pac_batch, _, _ = calculate_pac(
    batch_signal,
    fs=fs,
    pha_n_bands=20,
    amp_n_bands=20,
    device=device
)
batch_time = time.time() - start

print(f"Batch ({batch_size} signals) time: {batch_time:.3f}s")
print(f"Per-signal time: {batch_time/batch_size:.3f}s")
print(f"Batch efficiency: {(cpu_time * batch_size) / batch_time:.1f}x")

print("\nNote: TensorPAC comparison requires TensorPAC installation")
print("According to README: gPAC is 28-63x faster with GPU")
print("According to feature request: gPAC was 8x slower (before optimizations?)")