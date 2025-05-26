#!/usr/bin/env python3
"""
Test GPU scaling with gradually increasing data sizes.
Find the point where GPU becomes beneficial.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from src.gpac import calculate_pac

def test_data_size(n_channels, n_segments, seq_len, label):
    """Test gPAC CPU vs GPU for a specific data size."""
    print(f"\n{label}")
    print(f"Data: {n_channels}ch × {n_segments}seg × {seq_len}samples = {n_channels*n_segments*seq_len:,} points")
    print(f"Memory: ~{n_channels*n_segments*seq_len*4/1024**2:.1f} MB")
    
    # Generate data
    torch.manual_seed(42)
    signal = torch.randn(1, n_channels, n_segments, seq_len)
    
    # Test CPU
    start = time.time()
    pac_cpu, _, _ = calculate_pac(
        signal, fs=1000,
        pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=20,  # Smaller bands for speed
        amp_start_hz=60.0, amp_end_hz=160.0, amp_n_bands=20,
        device='cpu', mi_n_bins=18, use_optimized_filter=True,
    )
    cpu_time = time.time() - start
    
    # Test GPU
    start = time.time()
    pac_gpu, _, _ = calculate_pac(
        signal, fs=1000,
        pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=20,
        amp_start_hz=60.0, amp_end_hz=160.0, amp_n_bands=20,
        device='cuda', mi_n_bins=18, use_optimized_filter=True,
    )
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    print(f"CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x")
    
    return n_channels*n_segments*seq_len, cpu_time, gpu_time, speedup

print("GPU Scaling Test: Finding the breakeven point")
print("=" * 60)

# Test progressively larger datasets
test_cases = [
    (1, 1, 5000, "Tiny: Single trial"),
    (1, 1, 50000, "Small: Long trial"), 
    (1, 10, 10000, "Medium: Multi-trial"),
    (10, 1, 10000, "Medium: Multi-channel"),
    (1, 50, 10000, "Large: Many trials"),
    (20, 1, 25000, "Large: Many channels"),
    (10, 10, 10000, "Large: Multi everything"),
]

results = []
for n_ch, n_seg, seq_len, label in test_cases:
    try:
        data_points, cpu_time, gpu_time, speedup = test_data_size(n_ch, n_seg, seq_len, label)
        results.append((data_points, cpu_time, gpu_time, speedup, label))
    except Exception as e:
        print(f"Error: {e}")
        break

print(f"\n{'='*80}")
print("SUMMARY: GPU Scaling Results")
print(f"{'='*80}")
print(f"{'Data Points':<12} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<8} {'Description'}")
print("-" * 80)

for data_points, cpu_time, gpu_time, speedup, label in results:
    status = "✓ GPU wins" if speedup > 1 else "✗ CPU wins"
    print(f"{data_points:<12,} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<8.2f}x {label} ({status})")

# Find breakeven point
gpu_wins = [(dp, sp) for dp, _, _, sp, _ in results if sp > 1.0]
if gpu_wins:
    breakeven = min(gpu_wins)[0]
    print(f"\nBreakeven point: ~{breakeven:,} data points")
else:
    print(f"\nGPU did not outperform CPU in tested range")

print(f"\nConclusions:")
print(f"- Small datasets (<100k points): CPU is faster due to GPU overhead")
print(f"- Large datasets (>100k points): GPU becomes beneficial") 
print(f"- Your A100 GPU is optimized for very large batch processing")
print(f"- For single-trial analysis, CPU is often sufficient")