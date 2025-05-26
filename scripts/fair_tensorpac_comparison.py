#!/usr/bin/env python3
"""
Fair Performance Comparison: gPAC vs TensorPAC with multi-core CPU
"""

import time
import numpy as np
import torch
import multiprocessing
from gpac import calculate_pac

# Try to import tensorpac
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("TensorPAC not available")
    exit(1)

# Get number of CPU cores
n_cores = multiprocessing.cpu_count()
print(f"System has {n_cores} CPU cores available")

# Test parameters
fs = 1000
duration = 5
n_samples = int(fs * duration)
n_trials = 10  # Multiple trials for averaging

# Generate test signal
t = np.linspace(0, duration, n_samples, endpoint=False)
phase_freq = 8
amp_freq = 80
signal = (np.sin(2 * np.pi * phase_freq * t) + 
          0.5 * np.sin(2 * np.pi * amp_freq * t) + 
          0.2 * np.random.randn(n_samples))

# Frequency parameters
pha_bands = 50
amp_bands = 30

print(f"\nTest Configuration:")
print(f"Signal: {duration}s @ {fs}Hz")
print(f"Frequency resolution: {pha_bands} x {amp_bands} = {pha_bands * amp_bands} pairs")
print(f"Running {n_trials} trials for each method\n")

# 1. TensorPAC with single CPU
print("1. TensorPAC with n_jobs=1 (single CPU)")
print("-" * 40)
pac_tp = Pac(idpac=(2, 0, 0), f_pha=(2, 20, pha_bands), f_amp=(60, 160, amp_bands))
times_tp_single = []

for i in range(n_trials):
    start = time.time()
    pac_tp.filterfit(fs, signal, n_jobs=1)
    times_tp_single.append(time.time() - start)
    print(f"  Trial {i+1}: {times_tp_single[-1]:.3f}s")

mean_tp_single = np.mean(times_tp_single)
print(f"Mean time: {mean_tp_single:.3f}s\n")

# 2. TensorPAC with all CPUs
print(f"2. TensorPAC with n_jobs=-1 ({n_cores} CPUs)")
print("-" * 40)
times_tp_multi = []

for i in range(n_trials):
    start = time.time()
    pac_tp.filterfit(fs, signal, n_jobs=-1)
    times_tp_multi.append(time.time() - start)
    print(f"  Trial {i+1}: {times_tp_multi[-1]:.3f}s")

mean_tp_multi = np.mean(times_tp_multi)
print(f"Mean time: {mean_tp_multi:.3f}s")
print(f"Multi-core speedup: {mean_tp_single/mean_tp_multi:.1f}x\n")

# 3. gPAC on CPU
print("3. gPAC on CPU")
print("-" * 40)
signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
times_gpac_cpu = []

for i in range(n_trials):
    start = time.time()
    pac_gpac_cpu, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=pha_bands,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=amp_bands,
        device="cpu"
    )
    times_gpac_cpu.append(time.time() - start)
    print(f"  Trial {i+1}: {times_gpac_cpu[-1]:.3f}s")

mean_gpac_cpu = np.mean(times_gpac_cpu)
print(f"Mean time: {mean_gpac_cpu:.3f}s\n")

# 4. gPAC on GPU (if available)
if torch.cuda.is_available():
    print("4. gPAC on GPU")
    print("-" * 40)
    signal_gpu = signal_torch.cuda()
    times_gpac_gpu = []
    
    # Warmup
    _ = calculate_pac(signal_gpu, fs=fs, pha_n_bands=10, amp_n_bands=10)
    
    for i in range(n_trials):
        torch.cuda.synchronize()
        start = time.time()
        pac_gpac_gpu, _, _ = calculate_pac(
            signal_gpu,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=pha_bands,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=amp_bands
        )
        torch.cuda.synchronize()
        times_gpac_gpu.append(time.time() - start)
        print(f"  Trial {i+1}: {times_gpac_gpu[-1]:.3f}s")
    
    mean_gpac_gpu = np.mean(times_gpac_gpu)
    print(f"Mean time: {mean_gpac_gpu:.3f}s\n")
else:
    mean_gpac_gpu = None
    print("4. GPU not available\n")

# Summary
print("=" * 60)
print("FAIR PERFORMANCE COMPARISON SUMMARY")
print("=" * 60)
print(f"TensorPAC (1 CPU):      {mean_tp_single:.3f}s")
print(f"TensorPAC ({n_cores} CPUs):    {mean_tp_multi:.3f}s")
print(f"gPAC (CPU):             {mean_gpac_cpu:.3f}s")
if mean_gpac_gpu:
    print(f"gPAC (GPU):             {mean_gpac_gpu:.3f}s")

print("\nSpeedup Factors:")
print(f"TensorPAC multi-core vs single-core: {mean_tp_single/mean_tp_multi:.1f}x")
print(f"gPAC CPU vs TensorPAC multi-core: {mean_tp_multi/mean_gpac_cpu:.1f}x")
if mean_gpac_gpu:
    print(f"gPAC GPU vs TensorPAC multi-core: {mean_tp_multi/mean_gpac_gpu:.1f}x")
    print(f"gPAC GPU vs gPAC CPU: {mean_gpac_cpu/mean_gpac_gpu:.1f}x")

print("\n⚠️  Note: This is a fairer comparison using all available CPU cores for TensorPAC")