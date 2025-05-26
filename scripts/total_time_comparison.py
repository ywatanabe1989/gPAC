#!/usr/bin/env python3
"""
Total time comparison including initialization.
Shows actual time user would experience for a single PAC computation.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from tensorpac import Pac
from src.gpac import calculate_pac

# Realistic parameters from gPAC defaults
n_channels = 1
n_segments = 1  
seq_len = 5000  # 5s at 1000Hz
fs = 1000

# Use TensorPAC's 'hres' config: 50 bands for both phase and amplitude
# This matches gPAC's default phase bands and is more realistic
pha_n_bands = 50
amp_n_bands = 50
pha_range = (2.0, 20.0)
amp_range = (60.0, 160.0)

print(f"Configuration:")
print(f"Signal: {seq_len/fs}s @ {fs}Hz")
print(f"Frequency resolution: {pha_n_bands} x {amp_n_bands} = {pha_n_bands * amp_n_bands} pairs")
print(f"Phase range: {pha_range}, Amplitude range: {amp_range}")
print()

# Generate test signal
torch.manual_seed(42)
np.random.seed(42)
signal_torch = torch.randn(1, n_channels, n_segments, seq_len)
signal_numpy = signal_torch.numpy()

print("=" * 70)
print("TOTAL TIME (INITIALIZATION + COMPUTATION)")
print("=" * 70)

# 1. TensorPAC total time
print("\n1. TensorPAC")
print("-" * 40)
start_total = time.time()

# Initialize using TensorPAC's 'hres' preset
p = Pac(
    idpac=(2, 0, 0),  # Tort MI
    f_pha='hres',  # 50 bands from 2-20 Hz
    f_amp='hres',  # 50 bands from 60-160 Hz
    n_bins=18,
)

# Compute
pac_tensorpac = p.filterfit(fs, signal_numpy[0, 0, 0], n_jobs=-1)

total_time_tensorpac = time.time() - start_total
print(f"  Total time: {total_time_tensorpac:.4f}s")

# 2. gPAC CPU total time
print("\n2. gPAC CPU")
print("-" * 40)
start_total = time.time()

# Initialize and compute (calculate_pac does both)
pac_gpac_cpu, _, _ = calculate_pac(
    signal_torch,
    fs=fs,
    pha_start_hz=pha_range[0],
    pha_end_hz=pha_range[1],
    pha_n_bands=pha_n_bands,
    amp_start_hz=amp_range[0],
    amp_end_hz=amp_range[1],
    amp_n_bands=amp_n_bands,
    device='cpu',
    mi_n_bins=18,
    use_optimized_filter=True,
)

total_time_gpac_cpu = time.time() - start_total
print(f"  Total time: {total_time_gpac_cpu:.4f}s")

# 3. gPAC GPU total time
if torch.cuda.is_available():
    print("\n3. gPAC GPU")
    print("-" * 40)
    start_total = time.time()
    
    # Initialize and compute
    pac_gpac_gpu, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=pha_range[0],
        pha_end_hz=pha_range[1],
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_range[0],
        amp_end_hz=amp_range[1],
        amp_n_bands=amp_n_bands,
        device='cuda',
        mi_n_bins=18,
        use_optimized_filter=True,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time_gpac_gpu = time.time() - start_total
    print(f"  Total time: {total_time_gpac_gpu:.4f}s")

print("\n" + "=" * 70)
print("TOTAL TIME SUMMARY (INCLUDING INITIALIZATION)")
print("=" * 70)
print(f"\nMethod                Total Time    Speedup vs TensorPAC")
print("-" * 60)
print(f"TensorPAC            {total_time_tensorpac:.4f}s      1.0x")
print(f"gPAC CPU             {total_time_gpac_cpu:.4f}s      {total_time_tensorpac/total_time_gpac_cpu:.1f}x")
if torch.cuda.is_available():
    print(f"gPAC GPU             {total_time_gpac_gpu:.4f}s      {total_time_tensorpac/total_time_gpac_gpu:.1f}x")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. For a single computation, gPAC's initialization overhead matters")
print("2. TensorPAC creates filters on-demand, gPAC pre-creates all filters")
print("3. For batch processing, gPAC's speedup would be more pronounced")
print("4. Consider your use case: single shot vs repeated computations")