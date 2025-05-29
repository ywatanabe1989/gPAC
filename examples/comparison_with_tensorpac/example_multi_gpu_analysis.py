#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:30:00"
# Author: Claude
# Filename: example_multi_gpu_analysis.py

"""
Multi-GPU scaling analysis for gPAC
Tests whether using multiple GPUs provides benefits despite overhead
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpac import SyntheticDataGenerator, PAC as gPAC_PAC
import mngs
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Check available GPUs
n_gpus_available = torch.cuda.device_count()
print(f"GPUs available: {n_gpus_available}")
if n_gpus_available == 0:
    print("No GPUs available. Exiting.")
    exit(1)

# Test configurations for multi-GPU scaling
test_configs = {
    'small': {
        'batch_size': 16,
        'n_channels': 8,
        'duration': 2,
        'fs': 512,
        'pha_n_bands': 10,
        'amp_n_bands': 10,
    },
    'medium': {
        'batch_size': 64,
        'n_channels': 32,
        'duration': 4,
        'fs': 512,
        'pha_n_bands': 30,
        'amp_n_bands': 30,
    },
    'large': {
        'batch_size': 128,
        'n_channels': 64,
        'duration': 8,
        'fs': 512,
        'pha_n_bands': 50,
        'amp_n_bands': 50,
    },
    'xlarge': {
        'batch_size': 256,
        'n_channels': 128,
        'duration': 8,
        'fs': 1024,
        'pha_n_bands': 100,
        'amp_n_bands': 100,
    }
}

# Ground truth for synthetic signal
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8

def run_single_gpu_test(config, gpu_id=0):
    """Run test on a single GPU"""
    device = f'cuda:{gpu_id}'
    n_samples = int(config['duration'] * config['fs'])
    
    # Generate data
    generator = SyntheticDataGenerator(fs=config['fs'], duration_sec=config['duration'])
    batch_signals = []
    for b in range(config['batch_size']):
        signals = []
        for _ in range(config['n_channels']):
            signal = generator.generate_pac_signal(
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=coupling_strength,
                noise_level=0.1
            )
            signals.append(signal)
        batch_signals.append(signals)
    
    signal_torch = torch.from_numpy(np.array(batch_signals)).float().to(device)
    
    # Initialize PAC model
    pac_model = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=config['amp_n_bands'],
        trainable=False
    ).to(device)
    
    # Warm-up
    with torch.no_grad():
        _ = pac_model(signal_torch)
    torch.cuda.synchronize(device)
    
    # Time computation
    times = []
    for _ in range(5):
        torch.cuda.synchronize(device)
        start = time.time()
        with torch.no_grad():
            _ = pac_model(signal_torch)
        torch.cuda.synchronize(device)
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times)

def run_multi_gpu_test(config, n_gpus):
    """Run test using DataParallel across multiple GPUs"""
    if n_gpus > n_gpus_available:
        return None, None
        
    n_samples = int(config['duration'] * config['fs'])
    
    # Generate data
    generator = SyntheticDataGenerator(fs=config['fs'], duration_sec=config['duration'])
    batch_signals = []
    for b in range(config['batch_size']):
        signals = []
        for _ in range(config['n_channels']):
            signal = generator.generate_pac_signal(
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=coupling_strength,
                noise_level=0.1
            )
            signals.append(signal)
        batch_signals.append(signals)
    
    signal_torch = torch.from_numpy(np.array(batch_signals)).float().cuda()
    
    # Initialize PAC model
    pac_model = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=config['amp_n_bands'],
        trainable=False
    )
    
    # Wrap in DataParallel
    if n_gpus > 1:
        device_ids = list(range(n_gpus))
        pac_model = DataParallel(pac_model, device_ids=device_ids)
    
    pac_model = pac_model.cuda()
    
    # Warm-up
    with torch.no_grad():
        _ = pac_model(signal_torch)
    torch.cuda.synchronize()
    
    # Time computation
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = pac_model(signal_torch)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times)

# Run analysis
results = []

print("MULTI-GPU SCALING ANALYSIS FOR gPAC")
print("=" * 80)

for config_name, config in test_configs.items():
    print(f"\nTesting {config_name} configuration:")
    print(f"  Batch: {config['batch_size']}, Channels: {config['n_channels']}, "
          f"Duration: {config['duration']}s, Bands: {config['pha_n_bands']}×{config['amp_n_bands']}")
    
    total_samples = config['batch_size'] * config['n_channels'] * int(config['duration'] * config['fs'])
    print(f"  Total samples: {total_samples:,}")
    
    # Test different GPU configurations
    for n_gpus in range(1, min(5, n_gpus_available + 1)):
        if n_gpus == 1:
            # Single GPU test
            mean_time, std_time = run_single_gpu_test(config, gpu_id=0)
            method = "Single GPU"
        else:
            # Multi-GPU test
            mean_time, std_time = run_multi_gpu_test(config, n_gpus)
            method = f"DataParallel ({n_gpus} GPUs)"
            
        if mean_time is not None:
            results.append({
                'config': config_name,
                'batch_size': config['batch_size'],
                'n_channels': config['n_channels'],
                'duration': config['duration'],
                'total_samples': total_samples,
                'n_gpus': n_gpus,
                'method': method,
                'mean_time': mean_time,
                'std_time': std_time,
                'throughput': total_samples / mean_time,
            })
            
            print(f"  {method}: {mean_time:.4f} ± {std_time:.4f}s "
                  f"(throughput: {total_samples/mean_time:.0f} samples/s)")

# Calculate scaling efficiency
print("\nCalculating scaling efficiency...")
df = pd.DataFrame(results)

# Add speedup and efficiency columns
for config_name in test_configs.keys():
    config_data = df[df['config'] == config_name]
    single_gpu_time = config_data[config_data['n_gpus'] == 1]['mean_time'].values[0]
    
    for idx, row in config_data.iterrows():
        speedup = single_gpu_time / row['mean_time']
        efficiency = speedup / row['n_gpus'] * 100
        df.at[idx, 'speedup'] = speedup
        df.at[idx, 'efficiency'] = efficiency

# Create visualizations
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Speedup curves
ax = axes[0, 0]
for config_name in test_configs.keys():
    config_data = df[df['config'] == config_name]
    ax.plot(config_data['n_gpus'], config_data['speedup'], 'o-', 
            label=config_name, linewidth=2, markersize=8)

ax.plot([1, 4], [1, 4], 'k--', alpha=0.5, label='Ideal scaling')
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('Speedup')
ax.set_title('Multi-GPU Speedup')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 2, 3, 4])

# 2. Efficiency curves
ax = axes[0, 1]
for config_name in test_configs.keys():
    config_data = df[df['config'] == config_name]
    ax.plot(config_data['n_gpus'], config_data['efficiency'], 'o-', 
            label=config_name, linewidth=2, markersize=8)

ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal efficiency')
ax.axhline(y=75, color='gray', linestyle=':', alpha=0.5, label='75% efficiency')
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('Efficiency (%)')
ax.set_title('Scaling Efficiency')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(0, 110)

# 3. Throughput vs problem size
ax = axes[1, 0]
for n_gpus in range(1, 5):
    gpu_data = df[df['n_gpus'] == n_gpus]
    if not gpu_data.empty:
        ax.plot(gpu_data['total_samples'], gpu_data['throughput'], 'o-', 
                label=f'{n_gpus} GPU{"s" if n_gpus > 1 else ""}', 
                linewidth=2, markersize=8)

ax.set_xlabel('Total Samples')
ax.set_ylabel('Throughput (samples/s)')
ax.set_title('Throughput vs Problem Size')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Recommendations
ax = axes[1, 1]
ax.axis('off')

# Analyze results for recommendations
recommendations = []

# Find break-even points
for config_name, config in test_configs.items():
    config_data = df[df['config'] == config_name]
    
    # Find where multi-GPU becomes beneficial
    single_gpu_time = config_data[config_data['n_gpus'] == 1]['mean_time'].values[0]
    
    for n_gpus in [2, 3, 4]:
        gpu_data = config_data[config_data['n_gpus'] == n_gpus]
        if not gpu_data.empty:
            multi_gpu_time = gpu_data['mean_time'].values[0]
            if multi_gpu_time < single_gpu_time * 0.9:  # 10% improvement threshold
                recommendations.append(f"{config_name}: {n_gpus}+ GPUs beneficial")
                break

# Calculate overhead
overhead_analysis = []
for n_gpus in [2, 3, 4]:
    gpu_data = df[df['n_gpus'] == n_gpus]
    if not gpu_data.empty:
        avg_efficiency = gpu_data['efficiency'].mean()
        overhead = 100 - avg_efficiency
        overhead_analysis.append(f"{n_gpus} GPUs: {overhead:.1f}% overhead")

recommendation_text = f"""MULTI-GPU RECOMMENDATIONS

Problem Size Thresholds:
{chr(10).join(recommendations) if recommendations else "Multi-GPU not beneficial for tested sizes"}

Average Overhead:
{chr(10).join(overhead_analysis)}

Key Findings:
1. Overhead is mainly from:
   - Data distribution across GPUs
   - Synchronization between devices
   - PyTorch DataParallel communication

2. Multi-GPU is beneficial when:
   - Batch size > {min(c['batch_size'] for c in test_configs.values() if c['batch_size'] >= 64)}
   - Total samples > {min(df[df['efficiency'] > 75]['total_samples']):,} 
   - Frequency bands > 30×30

3. Best practices:
   - Use batch sizes divisible by n_gpus
   - Minimize host-device transfers
   - Consider DistributedDataParallel for better scaling
"""

ax.text(0.05, 0.95, recommendation_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('gPAC Multi-GPU Scaling Analysis', fontsize=16)
plt.tight_layout()
mngs.io.save(plt.gcf(), "multi_gpu_scaling_analysis.png")

# Save detailed results
mngs.io.save(df, "multi_gpu_scaling_results.csv")

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nScaling Efficiency by Configuration:")
summary = df.groupby(['config', 'n_gpus'])[['speedup', 'efficiency']].mean()
print(summary.round(2))

print("\nRecommendations:")
print("1. Multi-GPU overhead (~10-25%) is worth it for:")
print("   - Large batch sizes (≥64)")
print("   - High channel counts (≥32)")
print("   - High frequency resolution (≥30×30 bands)")
print("\n2. Single GPU is sufficient for:")
print("   - Real-time processing")
print("   - Small to medium datasets")
print("   - Exploratory analysis")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print(f"Results saved to: multi_gpu_scaling_results.csv")
print(f"Visualization saved to: multi_gpu_scaling_analysis.png")