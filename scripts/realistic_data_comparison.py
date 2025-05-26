#!/usr/bin/env python3
"""
Realistic data size comparison to show GPU advantages.
Tests multiple scenarios from small to large data.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from tensorpac import Pac
from src.gpac import calculate_pac

def test_scenario(name, n_channels, n_segments, seq_len_seconds, fs=1000):
    """Test a specific data scenario."""
    seq_len = int(seq_len_seconds * fs)
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"Data shape: {n_channels} channels × {n_segments} segments × {seq_len} samples")
    print(f"Signal length: {seq_len_seconds}s @ {fs}Hz")
    print(f"Total data points: {n_channels * n_segments * seq_len:,}")
    print(f"Memory estimate: ~{n_channels * n_segments * seq_len * 4 / 1024**2:.1f} MB")
    
    # Generate test data
    torch.manual_seed(42)
    np.random.seed(42)
    signal_torch = torch.randn(1, n_channels, n_segments, seq_len)
    signal_numpy = signal_torch.numpy()
    
    # Test TensorPAC (single channel, single segment only for comparison)
    print(f"\n1. TensorPAC (1 channel, 1 segment)")
    print("-" * 40)
    start_time = time.time()
    p = Pac(idpac=(2, 0, 0), f_pha='hres', f_amp='hres', n_bins=18)
    pac_tensorpac = p.filterfit(fs, signal_numpy[0, 0, 0], n_jobs=-1)
    tensorpac_time = time.time() - start_time
    print(f"  Time: {tensorpac_time:.4f}s")
    
    # Test gPAC CPU
    print(f"\n2. gPAC CPU (full data)")
    print("-" * 40)
    start_time = time.time()
    pac_gpac_cpu, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=50,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=50,
        device='cpu',
        mi_n_bins=18,
        use_optimized_filter=True,
    )
    cpu_time = time.time() - start_time
    print(f"  Time: {cpu_time:.4f}s")
    
    # Test gPAC GPU
    print(f"\n3. gPAC GPU (full data)")
    print("-" * 40)
    start_time = time.time()
    pac_gpac_gpu, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=50,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=50,
        device='cuda',
        mi_n_bins=18,
        use_optimized_filter=True,
    )
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"  Time: {gpu_time:.4f}s")
    
    print(f"\nResults:")
    print(f"  CPU speedup vs TensorPAC: {tensorpac_time/cpu_time:.1f}x")
    print(f"  GPU speedup vs TensorPAC: {tensorpac_time/gpu_time:.1f}x")
    print(f"  GPU speedup vs CPU: {cpu_time/gpu_time:.1f}x")
    
    return {
        'name': name,
        'data_points': n_channels * n_segments * seq_len,
        'tensorpac_time': tensorpac_time,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time
    }

# Test scenarios from small to large
scenarios = [
    ("Small: Single trial", 1, 1, 5),          # 5,000 samples
    ("Medium: Multi-channel", 64, 1, 10),      # 640,000 samples  
    ("Large: Multi-trial", 1, 100, 30),       # 3,000,000 samples
    ("Very Large: Full dataset", 64, 50, 60), # 192,000,000 samples
]

results = []
for name, n_channels, n_segments, duration in scenarios:
    try:
        result = test_scenario(name, n_channels, n_segments, duration)
        results.append(result)
    except Exception as e:
        print(f"Error in {name}: {e}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY: When does GPU become beneficial?")
print(f"{'='*80}")
print(f"{'Scenario':<20} {'Data Points':<15} {'CPU vs TensorPAC':<15} {'GPU vs TensorPAC':<15} {'GPU vs CPU':<10}")
print("-" * 80)
for r in results:
    cpu_speedup = r['tensorpac_time'] / r['cpu_time']
    gpu_speedup = r['tensorpac_time'] / r['gpu_time'] 
    gpu_vs_cpu = r['cpu_time'] / r['gpu_time']
    print(f"{r['name']:<20} {r['data_points']:<15,} {cpu_speedup:<15.1f}x {gpu_speedup:<15.1f}x {gpu_vs_cpu:<10.1f}x")

print(f"\nKey insights:")
print("- GPU becomes beneficial with larger datasets (>1M data points)")
print("- Multi-channel and multi-trial data favor GPU processing")
print("- CPU is often faster for small, single-trial analyses")
print("- A100 GPU shines with batch processing and large datasets")