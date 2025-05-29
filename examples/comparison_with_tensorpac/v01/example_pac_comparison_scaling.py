#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_pac_comparison_scaling.py

"""
Compare gPAC and TensorPAC performance with varying signal parameters
to find where GPU acceleration provides benefit.
"""

import numpy as np
import torch
import time
from tensorpac import Pac as TensorPAC_Pac
from gpac import SyntheticDataGenerator, PAC as gPAC_PAC
import matplotlib.pyplot as plt
import mngs

def time_gpac(signal, device='cuda'):
    """Time gPAC PAC computation"""
    # Convert to torch tensor
    signal_torch = torch.from_numpy(signal).float().to(device)
    if signal_torch.dim() == 2:
        signal_torch = signal_torch.unsqueeze(1)  # Add channel dimension
    
    # Initialize
    seq_len = signal.shape[-1]
    pac = gPAC_PAC(
        seq_len=seq_len,
        fs=1000,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=50,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=30,
        trainable=False
    ).to(device)
    
    # Warm up
    _ = pac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time computation
    start = time.time()
    output = pac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    compute_time = time.time() - start
    
    mi = output['pac'].squeeze().cpu().numpy()
    return compute_time, mi

def time_tensorpac(signal):
    """Time TensorPAC PAC computation"""
    # Initialize with same frequency ranges
    pac = TensorPAC_Pac(f_pha=(2, 20), f_amp=(60, 160), verbose=False)
    
    # Time computation
    start = time.time()
    mi = pac.filterfit(1000, signal, n_jobs=1)
    compute_time = time.time() - start
    
    return compute_time, mi

# Test parameters
durations = [1, 2, 5, 10, 20, 60]  # seconds, including 60s as requested
n_signals_list = [1, 4, 8, 16, 32]
sample_rate = 1000  # Will also test with 400Hz separately

results = {
    'duration': [],
    'n_signals': [],
    'gpac_time': [],
    'tensorpac_time': [],
    'speedup': []
}

print("Testing performance scaling...")
print("=" * 60)

for duration in durations:
    for n_signals in n_signals_list:
        # Generate signals
        n_samples = int(duration * sample_rate)
        generator = SyntheticDataGenerator(
            fs=sample_rate,
            duration_sec=duration
        )
        # Generate multiple signals
        signals = []
        for i in range(n_signals):
            signal = generator.generate_pac_signal(
                phase_freq=6.0,
                amp_freq=40.0,
                coupling_strength=0.5,
                noise_level=0.1
            )
            signals.append(signal)
        signals = np.array(signals)
        
        # Time gPAC
        gpac_time, gpac_mi = time_gpac(signals)
        
        # Time TensorPAC
        tensorpac_time, tensorpac_mi = time_tensorpac(signals)
        
        # Calculate speedup
        speedup = tensorpac_time / gpac_time
        
        # Store results
        results['duration'].append(duration)
        results['n_signals'].append(n_signals)
        results['gpac_time'].append(gpac_time)
        results['tensorpac_time'].append(tensorpac_time)
        results['speedup'].append(speedup)
        
        print(f"Duration: {duration:2d}s, Signals: {n_signals:2d} | "
              f"gPAC: {gpac_time:.4f}s, TensorPAC: {tensorpac_time:.4f}s | "
              f"Speedup: {speedup:.2f}x")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Speedup heatmap
ax = axes[0, 0]
speedup_matrix = np.array(results['speedup']).reshape(len(durations), len(n_signals_list))
im = ax.imshow(speedup_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
ax.set_xticks(range(len(n_signals_list)))
ax.set_xticklabels(n_signals_list)
ax.set_yticks(range(len(durations)))
ax.set_yticklabels(durations)
ax.set_xlabel('Number of Signals')
ax.set_ylabel('Duration (seconds)')
ax.set_title('Speedup Factor (TensorPAC time / gPAC time)')
plt.colorbar(im, ax=ax)

# Add text annotations
for i in range(len(durations)):
    for j in range(len(n_signals_list)):
        text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black")

# 2. Time vs duration for different batch sizes
ax = axes[0, 1]
for n_signals in [1, 8, 32]:
    mask = np.array(results['n_signals']) == n_signals
    durations_filtered = np.array(results['duration'])[mask]
    gpac_times = np.array(results['gpac_time'])[mask]
    tensorpac_times = np.array(results['tensorpac_time'])[mask]
    
    ax.plot(durations_filtered, gpac_times, 'o-', label=f'gPAC (n={n_signals})')
    ax.plot(durations_filtered, tensorpac_times, 's--', label=f'TensorPAC (n={n_signals})')

ax.set_xlabel('Signal Duration (seconds)')
ax.set_ylabel('Computation Time (seconds)')
ax.set_title('Computation Time vs Signal Duration')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Time per sample
ax = axes[1, 0]
total_samples = np.array(results['duration']) * sample_rate * np.array(results['n_signals'])
gpac_time_per_sample = np.array(results['gpac_time']) / total_samples * 1e6  # microseconds
tensorpac_time_per_sample = np.array(results['tensorpac_time']) / total_samples * 1e6

ax.scatter(total_samples/1000, gpac_time_per_sample, alpha=0.7, label='gPAC', s=50)
ax.scatter(total_samples/1000, tensorpac_time_per_sample, alpha=0.7, label='TensorPAC', s=50)
ax.set_xlabel('Total Samples (thousands)')
ax.set_ylabel('Time per Sample (μs)')
ax.set_title('Processing Efficiency')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# 4. Crossover analysis
ax = axes[1, 1]
# Find where speedup > 1
speedup_array = np.array(results['speedup'])
gpac_faster = speedup_array > 1.0

if np.any(gpac_faster):
    # Plot regions where each method is faster
    ax.text(0.5, 0.7, 'gPAC is faster for:', ha='center', transform=ax.transAxes, fontsize=12)
    faster_conditions = []
    for i, (d, n, s) in enumerate(zip(results['duration'], results['n_signals'], results['speedup'])):
        if s > 1.0:
            faster_conditions.append(f"Duration={d}s, N={n} (speedup={s:.2f}x)")
    
    for i, condition in enumerate(faster_conditions[:5]):  # Show top 5
        ax.text(0.5, 0.6-i*0.08, condition, ha='center', transform=ax.transAxes, fontsize=10)
else:
    ax.text(0.5, 0.5, 'TensorPAC is faster for all tested conditions', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    # Show closest to crossover
    best_speedup_idx = np.argmax(speedup_array)
    ax.text(0.5, 0.4, f'Best speedup: {speedup_array[best_speedup_idx]:.2f}x at\n'
            f'Duration={results["duration"][best_speedup_idx]}s, '
            f'N={results["n_signals"][best_speedup_idx]}',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Performance Summary')

plt.tight_layout()
mngs.io.save(plt.gcf(), "pac_scaling_comparison.png")
mngs.io.save(plt.gcf(), "pac_scaling_comparison.pdf")
print(f"\nVisualization saved")

# Save detailed results
import pandas as pd
df = pd.DataFrame(results)
mngs.io.save(df, "scaling_results.csv")
print(f"Detailed results saved")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"Best speedup: {max(results['speedup']):.2f}x")
best_idx = np.argmax(results['speedup'])
print(f"  at Duration={results['duration'][best_idx]}s, N={results['n_signals'][best_idx]}")
print(f"Average speedup: {np.mean(results['speedup']):.2f}x")
print(f"gPAC faster than TensorPAC: {np.sum(np.array(results['speedup']) > 1.0)} out of {len(results['speedup'])} tests")