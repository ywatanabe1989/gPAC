#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Speed comparison between gPAC and TensorPAC after cache fix
# Focus on actual computation speed, not caching

import time
import torch
import numpy as np
from gpac import PAC
from tensorpac import Pac as TensorPAC
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def benchmark_computation_speed(n_trials=5):
    """Compare computation speed between gPAC and TensorPAC."""
    
    print("=" * 80)
    print("gPAC vs TensorPAC COMPUTATION SPEED COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Note: Caching disabled for fair comparison")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # (n_epochs, n_times, name)
        (10, 1000, "Small"),
        (20, 2000, "Medium"), 
        (50, 5000, "Large"),
        (100, 10000, "XLarge"),
    ]
    
    fs = 500
    pha_range = (4, 8)    # Theta band
    amp_range = (30, 80)  # Gamma band
    pha_n_bands = 10
    amp_n_bands = 10
    n_perm = 20
    
    results = []
    
    for n_epochs, n_times, config_name in configs:
        print(f"\n{config_name} Configuration:")
        print(f"  Epochs: {n_epochs}, Time points: {n_times}")
        print(f"  Total samples: {n_epochs * n_times:,}")
        
        # Generate test data
        # For TensorPAC: (n_epochs, n_times)
        data_np = np.random.randn(n_epochs, n_times).astype(np.float32)
        
        # For gPAC: (batch=1, channels=n_epochs, time=n_times)
        data_torch = torch.from_numpy(data_np).unsqueeze(0).cuda()
        
        # Initialize gPAC (caching disabled)
        gpac = PAC(
            seq_len=n_times,
            fs=fs,
            pha_range_hz=pha_range,
            amp_range_hz=amp_range,
            pha_n_bands=pha_n_bands,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            enable_caching=False,  # Disable caching for fair comparison
            fp16=False,
        ).cuda()
        
        # Initialize TensorPAC
        tpac = TensorPAC(idpac=(2, 2, 1), f_pha=pha_range, f_amp=amp_range, 
                         n_bins=18, verbose=False)
        
        # Warm-up runs
        print("  Running warm-up...")
        _ = gpac(data_torch)
        _ = tpac.filterfit(fs, data_np, n_perm=n_perm, n_jobs=1)
        
        # Benchmark gPAC
        print("  Benchmarking gPAC...")
        times_gpac = []
        for trial in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.time()
            result_gpac = gpac(data_torch)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            times_gpac.append(elapsed)
            print(f"    Trial {trial+1}: {elapsed:.3f}s")
        
        # Benchmark TensorPAC
        print("  Benchmarking TensorPAC...")
        times_tpac = []
        for trial in range(n_trials):
            t0 = time.time()
            result_tpac = tpac.filterfit(fs, data_np, n_perm=n_perm, n_jobs=1)
            elapsed = time.time() - t0
            times_tpac.append(elapsed)
            print(f"    Trial {trial+1}: {elapsed:.3f}s")
        
        # Calculate statistics
        gpac_mean = np.mean(times_gpac)
        gpac_std = np.std(times_gpac)
        tpac_mean = np.mean(times_tpac)
        tpac_std = np.std(times_tpac)
        speedup = tpac_mean / gpac_mean
        
        # Verify output shapes
        gpac_shape = result_gpac['pac'].shape
        tpac_shape = result_tpac.shape
        print(f"\n  Output shapes:")
        print(f"    gPAC: {gpac_shape}")
        print(f"    TensorPAC: {tpac_shape}")
        
        results.append({
            'Config': config_name,
            'Epochs': n_epochs,
            'TimePoints': n_times,
            'TotalSamples': n_epochs * n_times,
            'gPAC_mean': gpac_mean,
            'gPAC_std': gpac_std,
            'TensorPAC_mean': tpac_mean,
            'TensorPAC_std': tpac_std,
            'Speedup': speedup
        })
        
        print(f"\n  Results:")
        print(f"    gPAC:      {gpac_mean:.3f} ± {gpac_std:.3f}s")
        print(f"    TensorPAC: {tpac_mean:.3f} ± {tpac_std:.3f}s")
        print(f"    Speedup:   {speedup:.1f}x")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('benchmark/gpac_vs_tensorpac_speed_comparison_out', exist_ok=True)
    df.to_csv('benchmark/gpac_vs_tensorpac_speed_comparison_out/results.csv', index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Time comparison
    ax = axes[0, 0]
    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width/2, df['gPAC_mean'], width, yerr=df['gPAC_std'], 
           label='gPAC', color='blue', capsize=5)
    ax.bar(x + width/2, df['TensorPAC_mean'], width, yerr=df['TensorPAC_std'],
           label='TensorPAC', color='red', capsize=5)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Processing Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Config'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Speedup
    ax = axes[0, 1]
    bars = ax.bar(df['Config'], df['Speedup'], color='green')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('gPAC Speedup vs TensorPAC')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    for bar, speedup in zip(bars, df['Speedup']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    # 3. Scaling with data size
    ax = axes[1, 0]
    ax.loglog(df['TotalSamples'], df['gPAC_mean'], 'o-', 
              label='gPAC', linewidth=2, markersize=8)
    ax.loglog(df['TotalSamples'], df['TensorPAC_mean'], 's-', 
              label='TensorPAC', linewidth=2, markersize=8)
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Time per sample
    ax = axes[1, 1]
    time_per_sample_gpac = df['gPAC_mean'] / df['TotalSamples'] * 1e6  # microseconds
    time_per_sample_tpac = df['TensorPAC_mean'] / df['TotalSamples'] * 1e6
    ax.plot(df['Config'], time_per_sample_gpac, 'o-', label='gPAC', linewidth=2, markersize=8)
    ax.plot(df['Config'], time_per_sample_tpac, 's-', label='TensorPAC', linewidth=2, markersize=8)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time per sample (μs)')
    ax.set_title('Processing Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark/gpac_vs_tensorpac_speed_comparison_out/speed_comparison.png', dpi=150)
    plt.close()
    
    # Create detailed report
    with open('benchmark/gpac_vs_tensorpac_speed_comparison_out/report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("gPAC vs TensorPAC Speed Comparison Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Phase bands: {pha_n_bands} in range {pha_range} Hz\n")
        f.write(f"  Amplitude bands: {amp_n_bands} in range {amp_range} Hz\n")
        f.write(f"  Permutations: {n_perm}\n")
        f.write(f"  Sampling rate: {fs} Hz\n")
        f.write(f"  Trials per test: {n_trials}\n\n")
        
        f.write("Results:\n")
        for _, row in df.iterrows():
            f.write(f"\n{row['Config']} ({row['Epochs']} epochs × {row['TimePoints']} time points):\n")
            f.write(f"  gPAC:      {row['gPAC_mean']:.3f} ± {row['gPAC_std']:.3f} seconds\n")
            f.write(f"  TensorPAC: {row['TensorPAC_mean']:.3f} ± {row['TensorPAC_std']:.3f} seconds\n")
            f.write(f"  Speedup:   {row['Speedup']:.1f}x\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"  Average speedup: {df['Speedup'].mean():.1f}x\n")
        f.write(f"  Min speedup: {df['Speedup'].min():.1f}x\n")
        f.write(f"  Max speedup: {df['Speedup'].max():.1f}x\n")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY:")
    print(f"Average speedup: {df['Speedup'].mean():.1f}x")
    print(f"Range: {df['Speedup'].min():.1f}x - {df['Speedup'].max():.1f}x")
    print("\nResults saved to: benchmark/gpac_vs_tensorpac_speed_comparison_out/")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    df = benchmark_computation_speed(n_trials=3)