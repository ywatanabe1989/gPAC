#!/usr/bin/env python3
"""
Separate initialization and computation timing.
Fair GPU comparison with data pre-loaded.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

def time_gpac_components(signal, device, label):
    """Time gPAC initialization and computation separately."""
    from src.gpac._PAC import PAC
    
    print(f"\n{label}")
    print("-" * 40)
    
    # Move signal to device first (like DataLoader)
    if device == 'cuda' and torch.cuda.is_available():
        signal = signal.cuda()
        torch.cuda.synchronize()
    
    seq_len = signal.shape[-1]
    
    # 1. Time initialization only
    start = time.time()
    model = PAC(
        seq_len=seq_len,
        fs=1000,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=20,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=20,
        mi_n_bins=18,
        use_optimized_filter=True,
    )
    
    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.synchronize()
    
    init_time = time.time() - start
    print(f"  Initialization: {init_time:.4f}s")
    
    # 2. Time computation only (data already on device)
    start = time.time()
    with torch.no_grad():
        result = model(signal)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    comp_time = time.time() - start
    print(f"  Computation: {comp_time:.4f}s")
    print(f"  Total: {init_time + comp_time:.4f}s")
    
    return init_time, comp_time

def test_data_size(n_channels, n_segments, seq_len, scenario_name):
    """Test different data sizes."""
    print(f"\n{'='*60}")
    print(f"{scenario_name}")
    print(f"{'='*60}")
    
    data_points = n_channels * n_segments * seq_len
    print(f"Data: {n_channels}ch × {n_segments}seg × {seq_len}samples = {data_points:,} points")
    print(f"Memory: ~{data_points * 4 / 1024**2:.1f} MB")
    
    # Generate test data
    torch.manual_seed(42)
    signal = torch.randn(1, n_channels, n_segments, seq_len)
    
    # Test CPU
    cpu_init, cpu_comp = time_gpac_components(signal.clone(), 'cpu', "CPU Timing")
    
    # Test GPU if available
    if torch.cuda.is_available():
        gpu_init, gpu_comp = time_gpac_components(signal.clone(), 'cuda', "GPU Timing (Data Pre-loaded)")
        
        # Also test cold start (includes data transfer)
        print(f"\nGPU Cold Start (includes data transfer)")
        print("-" * 40)
        signal_cpu = signal.clone()  # Start with CPU data
        
        start = time.time()
        signal_gpu = signal_cpu.cuda()  # Transfer to GPU
        
        # Use existing model on GPU for fair comparison
        model = PAC(
            seq_len=seq_len,
            fs=1000,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=20,
            amp_start_hz=60.0,
            amp_end_hz=160.0,
            amp_n_bands=20,
            mi_n_bins=18,
            use_optimized_filter=True,
        ).cuda()
        
        with torch.no_grad():
            result = model(signal_gpu)
        torch.cuda.synchronize()
        
        cold_time = time.time() - start
        print(f"  Transfer + Computation: {cold_time:.4f}s")
        
        # Summary table
        print(f"\nSummary:")
        print(f"{'Component':<20} {'CPU':<10} {'GPU':<10} {'Speedup':<10}")
        print("-" * 50)
        print(f"{'Initialization':<20} {cpu_init:<10.4f} {gpu_init:<10.4f} {cpu_init/gpu_init:<10.2f}x")
        print(f"{'Computation (warm)':<20} {cpu_comp:<10.4f} {gpu_comp:<10.4f} {cpu_comp/gpu_comp:<10.2f}x")
        print(f"{'Total (warm)':<20} {cpu_init+cpu_comp:<10.4f} {gpu_init+gpu_comp:<10.4f} {(cpu_init+cpu_comp)/(gpu_init+gpu_comp):<10.2f}x")
        print(f"{'Computation (cold)':<20} {cpu_comp:<10.4f} {cold_time:<10.4f} {cpu_comp/cold_time:<10.2f}x")
        
        return {
            'data_points': data_points,
            'cpu_init': cpu_init,
            'cpu_comp': cpu_comp,
            'gpu_init': gpu_init,
            'gpu_comp': gpu_comp,
            'gpu_cold': cold_time,
            'scenario': scenario_name
        }
    else:
        return {
            'data_points': data_points,
            'cpu_init': cpu_init,
            'cpu_comp': cpu_comp,
            'scenario': scenario_name
        }

# Test scenarios
test_cases = [
    (1, 1, 5000, "Small: Single trial"),
    (1, 1, 50000, "Medium: Long trial"),
    (10, 1, 10000, "Large: Multi-channel"),
    (1, 50, 10000, "Large: Multi-trial"),
]

print("SEPARATED TIMING ANALYSIS")
print("Measuring initialization vs computation separately")
print("GPU timing with pre-loaded data (DataLoader scenario)")

results = []
for n_ch, n_seg, seq_len, scenario in test_cases:
    try:
        result = test_data_size(n_ch, n_seg, seq_len, scenario)
        results.append(result)
    except Exception as e:
        print(f"Error in {scenario}: {e}")
        import traceback
        traceback.print_exc()

# Final comparison
print(f"\n{'='*80}")
print("OVERALL COMPARISON: DataLoader Use Case")
print(f"{'='*80}")
print("Scenario: Initialize once, compute many times (like training loop)")
print()

if len(results) > 0 and 'gpu_init' in results[0]:
    print(f"{'Scenario':<20} {'Data Points':<12} {'Init Ratio':<10} {'Comp Ratio':<10} {'GPU Better?':<12}")
    print("-" * 70)
    
    for r in results:
        if 'gpu_init' in r:
            init_ratio = r['cpu_init'] / r['gpu_init']
            comp_ratio = r['cpu_comp'] / r['gpu_comp']
            gpu_better = "Yes" if comp_ratio > 1 else "No"
            print(f"{r['scenario']:<20} {r['data_points']:<12,} {init_ratio:<10.2f}x {comp_ratio:<10.2f}x {gpu_better:<12}")

print(f"\nDataLoader scenario insights:")
print("- Initialization happens once per training session")
print("- Computation happens many times per epoch")
print("- GPU excels when computation >> initialization")
print("- Pre-loading data eliminates transfer overhead")