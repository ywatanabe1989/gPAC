#!/usr/bin/env python3
"""
Precise timing breakdown: Initialization vs Computation.
Fair GPU comparison with data pre-loaded on GPU (like DataLoader scenario).
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from src.gpac import calculate_pac
from src.gpac._PAC import PAC

def precise_timing_test(n_channels, n_segments, seq_len, label):
    """Precise timing breakdown for CPU vs GPU."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Data: {n_channels}ch × {n_segments}seg × {seq_len}samples = {n_channels*n_segments*seq_len:,} points")
    
    # Generate data
    torch.manual_seed(42)
    signal_cpu = torch.randn(1, n_channels, n_segments, seq_len)
    
    # Pre-move data to GPU (like DataLoader scenario)
    if torch.cuda.is_available():
        signal_gpu = signal_cpu.cuda()
        torch.cuda.synchronize()  # Ensure transfer is complete
    
    # PAC parameters
    params = {
        'fs': 1000,
        'pha_start_hz': 2.0,
        'pha_end_hz': 20.0, 
        'pha_n_bands': 20,
        'amp_start_hz': 60.0,
        'amp_end_hz': 160.0,
        'amp_n_bands': 20,
        'mi_n_bins': 18,
        'use_optimized_filter': True,
    }
    
    print(f"\n1. CPU TIMING BREAKDOWN")
    print("-" * 40)
    
    # CPU: Measure initialization separately
    start = time.time()
    model_cpu = PAC(
        pha_start_hz=params['pha_start_hz'],
        pha_end_hz=params['pha_end_hz'],
        pha_n_bands=params['pha_n_bands'],
        amp_start_hz=params['amp_start_hz'],
        amp_end_hz=params['amp_end_hz'],
        amp_n_bands=params['amp_n_bands'],
        fs=params['fs'],
        seq_len=seq_len,
        mi_n_bins=params['mi_n_bins'],
        device='cpu',
        use_optimized_filter=params['use_optimized_filter'],
    )
    cpu_init_time = time.time() - start
    print(f"  Initialization: {cpu_init_time:.4f}s")
    
    # CPU: Measure computation only
    start = time.time()
    with torch.no_grad():
        result_cpu = model_cpu(signal_cpu)
    cpu_comp_time = time.time() - start
    print(f"  Computation: {cpu_comp_time:.4f}s")
    print(f"  Total: {cpu_init_time + cpu_comp_time:.4f}s")
    
    if torch.cuda.is_available():
        print(f"\n2. GPU TIMING BREAKDOWN (Data Pre-loaded)")
        print("-" * 40)
        
        # GPU: Measure initialization separately
        start = time.time()
        model_gpu = PAC(
            pha_start_hz=params['pha_start_hz'],
            pha_end_hz=params['pha_end_hz'],
            pha_n_bands=params['pha_n_bands'],
            amp_start_hz=params['amp_start_hz'],
            amp_end_hz=params['amp_end_hz'],
            amp_n_bands=params['amp_n_bands'],
            fs=params['fs'],
            seq_len=seq_len,
            mi_n_bins=params['mi_n_bins'],
            device='cuda',
            use_optimized_filter=params['use_optimized_filter'],
        )
        torch.cuda.synchronize()
        gpu_init_time = time.time() - start
        print(f"  Initialization: {gpu_init_time:.4f}s")
        
        # GPU: Measure computation only (data already on GPU)
        start = time.time()
        with torch.no_grad():
            result_gpu = model_gpu(signal_gpu)
        torch.cuda.synchronize()
        gpu_comp_time = time.time() - start
        print(f"  Computation: {gpu_comp_time:.4f}s")
        print(f"  Total: {gpu_init_time + gpu_comp_time:.4f}s")
        
        # Also test with data transfer included
        print(f"\n3. GPU WITH DATA TRANSFER (Cold Start)")
        print("-" * 40)
        start = time.time()
        signal_gpu_cold = signal_cpu.cuda()
        with torch.no_grad():
            result_gpu_cold = model_gpu(signal_gpu_cold)
        torch.cuda.synchronize()
        gpu_cold_time = time.time() - start
        print(f"  Data transfer + Computation: {gpu_cold_time:.4f}s")
        
        print(f"\n4. COMPARISON SUMMARY")
        print("-" * 40)
        print(f"{'Component':<25} {'CPU':<10} {'GPU':<10} {'GPU Speedup':<12}")
        print("-" * 60)
        print(f"{'Initialization':<25} {cpu_init_time:<10.4f} {gpu_init_time:<10.4f} {cpu_init_time/gpu_init_time:<12.2f}x")
        print(f"{'Computation (warm)':<25} {cpu_comp_time:<10.4f} {gpu_comp_time:<10.4f} {cpu_comp_time/gpu_comp_time:<12.2f}x")
        print(f"{'Total (warm)':<25} {cpu_init_time+cpu_comp_time:<10.4f} {gpu_init_time+gpu_comp_time:<10.4f} {(cpu_init_time+cpu_comp_time)/(gpu_init_time+gpu_comp_time):<12.2f}x")
        print(f"{'Computation (cold)':<25} {cpu_comp_time:<10.4f} {gpu_cold_time:<10.4f} {cpu_comp_time/gpu_cold_time:<12.2f}x")
        
        return {
            'data_points': n_channels * n_segments * seq_len,
            'cpu_init': cpu_init_time,
            'cpu_comp': cpu_comp_time,
            'gpu_init': gpu_init_time,
            'gpu_comp': gpu_comp_time,
            'gpu_cold': gpu_cold_time,
            'label': label
        }
    else:
        return {
            'data_points': n_channels * n_segments * seq_len,
            'cpu_init': cpu_init_time,
            'cpu_comp': cpu_comp_time,
            'label': label
        }

# Test different data sizes
test_cases = [
    (1, 1, 5000, "Small: Single trial (5k points)"),
    (1, 1, 50000, "Medium: Long trial (50k points)"),
    (10, 1, 10000, "Large: Multi-channel (100k points)"),
    (1, 50, 10000, "Large: Multi-trial (500k points)"),
]

print("PRECISE TIMING BREAKDOWN: gPAC CPU vs GPU")
print("Testing initialization vs computation times")
print("GPU timing with data pre-loaded (DataLoader scenario)")

results = []
for n_ch, n_seg, seq_len, label in test_cases:
    try:
        result = precise_timing_test(n_ch, n_seg, seq_len, label)
        results.append(result)
    except Exception as e:
        print(f"Error in {label}: {e}")

# Final summary
print(f"\n{'='*80}")
print("FINAL SUMMARY: When is GPU beneficial?")
print(f"{'='*80}")
print(f"{'Scenario':<25} {'Data Points':<12} {'Init Speedup':<12} {'Comp Speedup':<12} {'Overall':<10}")
print("-" * 80)

for r in results:
    if 'gpu_init' in r:
        init_speedup = r['cpu_init'] / r['gpu_init']
        comp_speedup = r['cpu_comp'] / r['gpu_comp']
        total_speedup = (r['cpu_init'] + r['cpu_comp']) / (r['gpu_init'] + r['gpu_comp'])
        status = "✓ GPU" if total_speedup > 1 else "✗ CPU"
        print(f"{r['label']:<25} {r['data_points']:<12,} {init_speedup:<12.2f}x {comp_speedup:<12.2f}x {total_speedup:<10.2f}x {status}")

print(f"\nKey insights for DataLoader scenarios:")
print("- Initialization happens once, computation many times")
print("- GPU shines when computation dominates over initialization")
print("- Pre-loading data to GPU eliminates transfer overhead")
print("- Batch processing amortizes GPU initialization cost")