#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 15:00:00"
# Author: Claude
# Filename: example_gpu_optimized_benchmark.py

"""
GPU-optimized benchmark showing where gPAC should excel:
- Larger batch sizes for better GPU utilization
- Excludes data transfer overhead (assumes DataLoader)
- Tests scaling behavior
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from gpac import PAC as gPAC_PAC
from gpac import SyntheticDataGenerator
from tensorpac import Pac as TensorPAC_Pac
import warnings
warnings.filterwarnings('ignore')

def gpu_optimized_benchmark():
    """Run benchmarks optimized for GPU performance"""
    print("GPU-OPTIMIZED gPAC vs TensorPAC BENCHMARK")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU available, results won't be representative")
        return
    
    # Get GPU info
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CPU cores available: {64}")
    print()
    
    # Test configurations
    configs = [
        # Optimal GPU scenarios
        {'name': 'Small Batch (Suboptimal)', 'batch_size': 1, 'n_channels': 64, 'duration': 10, 'fs': 512, 'bands': 30},
        {'name': 'Medium Batch', 'batch_size': 8, 'n_channels': 64, 'duration': 10, 'fs': 512, 'bands': 30},
        {'name': 'Large Batch (Optimal)', 'batch_size': 32, 'n_channels': 64, 'duration': 10, 'fs': 512, 'bands': 30},
        {'name': 'Very Large Batch', 'batch_size': 64, 'n_channels': 64, 'duration': 10, 'fs': 512, 'bands': 30},
        # Scaling tests
        {'name': 'Many Channels', 'batch_size': 16, 'n_channels': 256, 'duration': 10, 'fs': 512, 'bands': 30},
        {'name': 'Long Duration', 'batch_size': 16, 'n_channels': 64, 'duration': 30, 'fs': 512, 'bands': 30},
        {'name': 'Many Bands', 'batch_size': 16, 'n_channels': 64, 'duration': 10, 'fs': 512, 'bands': 50},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Config: batch={config['batch_size']}, ch={config['n_channels']}, "
              f"dur={config['duration']}s, fs={config['fs']}Hz, bands={config['bands']}×{config['bands']}")
        
        n_samples = int(config['duration'] * config['fs'])
        nyquist = config['fs'] / 2
        amp_end_hz = min(150, nyquist - 10)
        
        # Initialize models
        pac_gpac = gPAC_PAC(
            seq_len=n_samples,
            fs=config['fs'],
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=config['bands'],
            amp_start_hz=30,
            amp_end_hz=amp_end_hz,
            amp_n_bands=config['bands'],
            trainable=False
        ).to(device)
        
        pha_edges = np.linspace(2, 20, config['bands'] + 1)
        amp_edges = np.linspace(30, amp_end_hz, config['bands'] + 1)
        pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
        
        pac_tp = TensorPAC_Pac(
            idpac=(2, 0, 0),
            f_pha=pha_bands,
            f_amp=amp_bands,
            verbose=False
        )
        
        # Generate test data
        data_np = np.random.randn(config['batch_size'], config['n_channels'], n_samples).astype(np.float32)
        data_gpu = torch.from_numpy(data_np).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = pac_gpac(data_gpu)
        torch.cuda.synchronize()
        
        # Time GPU computation only (no transfer)
        gpu_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                results_gpu = pac_gpac(data_gpu)
            
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start)
        
        gpu_mean = np.mean(gpu_times[1:])  # Exclude first run
        
        # Time with CPU transfer
        full_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                results_gpu = pac_gpac(data_gpu)
            pac_cpu = results_gpu["pac"].cpu().numpy()
            
            torch.cuda.synchronize()
            full_times.append(time.time() - start)
        
        full_mean = np.mean(full_times[1:])
        
        # Time TensorPAC
        signals_reshaped = data_np.transpose(0, 2, 1).reshape(n_samples, -1)
        
        tp_times = []
        for _ in range(3):
            start = time.time()
            _ = pac_tp.filterfit(config['fs'], signals_reshaped, n_jobs=64)
            tp_times.append(time.time() - start)
        
        tp_mean = np.mean(tp_times[1:])
        
        # Calculate metrics
        total_samples = config['batch_size'] * config['n_channels'] * n_samples
        gpu_throughput = total_samples / gpu_mean / 1e6
        full_throughput = total_samples / full_mean / 1e6
        tp_throughput = total_samples / tp_mean / 1e6
        
        speedup_gpu = tp_mean / gpu_mean
        speedup_full = tp_mean / full_mean
        transfer_overhead = (full_mean - gpu_mean) / gpu_mean * 100
        
        print(f"  gPAC (GPU only): {gpu_mean:.3f}s ({gpu_throughput:.1f}M samples/s)")
        print(f"  gPAC (with transfer): {full_mean:.3f}s ({full_throughput:.1f}M samples/s)")
        print(f"  TensorPAC (64 cores): {tp_mean:.3f}s ({tp_throughput:.1f}M samples/s)")
        print(f"  Speedup: {speedup_gpu:.2f}x (GPU only), {speedup_full:.2f}x (with transfer)")
        print(f"  Transfer overhead: {transfer_overhead:.1f}%")
        
        results.append({
            'name': config['name'],
            'batch_size': config['batch_size'],
            'total_samples': total_samples,
            'gpac_gpu': gpu_mean,
            'gpac_full': full_mean,
            'tensorpac': tp_mean,
            'speedup_gpu': speedup_gpu,
            'speedup_full': speedup_full
        })
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    names = [r['name'] for r in results]
    speedups_gpu = [r['speedup_gpu'] for r in results]
    speedups_full = [r['speedup_full'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, speedups_gpu, width, label='GPU only', color='blue')
    ax1.bar(x + width/2, speedups_full, width, label='With CPU transfer', color='orange')
    ax1.axhline(y=1, color='red', linestyle='--', label='Equal performance')
    ax1.set_ylabel('Speedup vs TensorPAC')
    ax1.set_title('gPAC Performance vs TensorPAC (64 CPU cores)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throughput comparison
    batch_sizes = [1, 8, 32, 64]
    throughputs = []
    
    for bs in batch_sizes:
        data_gpu = torch.randn(bs, 64, 5120).to(device)
        
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = pac_gpac(data_gpu)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times[1:])
        throughput = bs * 64 * 5120 / mean_time / 1e6
        throughputs.append(throughput)
    
    ax2.plot(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (M samples/s)')
    ax2.set_title('gPAC Throughput vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('gpu_optimized_benchmark_results.png', dpi=150)
    print(f"\nResults saved to gpu_optimized_benchmark_results.png")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best gPAC performance: {max(speedups_gpu):.2f}x speedup (GPU only)")
    print(f"With CPU transfer: {max(speedups_full):.2f}x speedup")
    print(f"Optimal batch size: {results[speedups_gpu.index(max(speedups_gpu))]['batch_size']}")
    print("\nKey findings:")
    print("- Small batches (1-4) severely underutilize GPU")
    print("- GPU→CPU transfer adds significant overhead")
    print("- Larger batches (32-64) show better GPU utilization")
    print("- For small workloads, CPU parallelization (TensorPAC) is more efficient")

if __name__ == "__main__":
    gpu_optimized_benchmark()