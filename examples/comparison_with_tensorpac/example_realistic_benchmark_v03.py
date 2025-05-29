#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 14:00:00"
# Author: Claude
# Filename: example_realistic_benchmark_fair.py

"""
Fair real-world benchmark comparison of gPAC vs TensorPAC
Reflects how scientists actually use these tools in practice
"""

import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mngs
import pandas as pd
from gpac import PAC as gPAC_PAC
from gpac import SyntheticDataGenerator
from tensorpac import Pac as TensorPAC_Pac
import warnings
warnings.filterwarnings('ignore')

# Realistic baseline for neuroscience
REALISTIC_BASELINE = {
    'batch_size': 1,       # Single subject/trial initially
    'n_channels': 64,      # Standard EEG
    'duration': 10,        # 10-second segments
    'fs': 512,            # Common EEG sampling rate
    'pha_n_bands': 30,    # Good resolution
    'amp_n_bands': 30,    # Good resolution
    'n_jobs': 64,         # Typical CPU cores on A100 node
}

# Parameters to vary
REALISTIC_VARIATIONS = {
    'batch_size': [1, 2, 4, 8, 16, 32],
    'n_channels': [16, 32, 64, 128, 256],
    'duration': [5, 10, 20, 30],
    'fs': [256, 512, 1024],
    'pha_n_bands': [10, 30, 50],
    'amp_n_bands': [10, 30, 50],
    'n_jobs': [32, 64, 128],  # Test different CPU core counts
}

def realistic_timing(method, data, config, pac_model=None, n_runs=2, n_warmup=1):
    """
    Time PAC computation as scientists would use it:
    - Initialization is separate (done once)
    - Multiple runs on same model (typical workflow)
    - Include necessary data transfers
    - Use wall-clock time
    """
    # Warmup run
    for _ in range(n_warmup):
        if method == "gpac":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data_torch = torch.from_numpy(data).float().to(device)
            with torch.no_grad():
                _ = pac_model(data_torch)
            if device == "cuda":
                torch.cuda.synchronize()
        elif method == "tensorpac":
            n_samples = data.shape[-1]
            signals_reshaped = data.transpose(0, 2, 1).reshape(n_samples, -1)
            _ = pac_model.filterfit(config["fs"], signals_reshaped, n_jobs=config.get('n_jobs', 64))
    
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        
        if method == "gpac":
            # Data to GPU (scientists do this per batch)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data_torch = torch.from_numpy(data).float().to(device)
            
            # Compute PAC
            with torch.no_grad():
                results = pac_model(data_torch)
            
            # Get results back to CPU (necessary for analysis)
            pac_matrix = results["pac"].cpu().numpy()
            
            # Ensure GPU operations complete
            if device == "cuda":
                torch.cuda.synchronize()
                
        elif method == "tensorpac":
            # Reshape for TensorPAC (time x channels*batch)
            n_samples = data.shape[-1]
            signals_reshaped = data.transpose(0, 2, 1).reshape(n_samples, -1)
            
            # Compute PAC using specified CPU cores
            pac_matrix = pac_model.filterfit(
                config["fs"], 
                signals_reshaped,
                n_jobs=config.get('n_jobs', 64)  # Use configured CPU cores
            )
        
        end = time.time()
        times.append(end - start)
    
    return pac_matrix, np.mean(times), np.std(times)

def run_fair_benchmark(config, verbose=True):
    """Run a single fair benchmark test"""
    n_samples = int(config['duration'] * config['fs'])
    
    if verbose:
        print(f"\nConfig: batch={config['batch_size']}, ch={config['n_channels']}, "
              f"dur={config['duration']}s, fs={config['fs']}Hz, "
              f"bands={config['pha_n_bands']}×{config['amp_n_bands']}, "
              f"n_jobs={config.get('n_jobs', 64)}")
    
    # Generate realistic data
    generator = SyntheticDataGenerator(fs=config['fs'], duration_sec=config['duration'])
    batch_signals = []
    
    for b in range(config['batch_size']):
        signals = []
        for ch in range(config['n_channels']):
            signal = generator.generate_pac_signal(
                phase_freq=6.0,
                amp_freq=80.0,
                coupling_strength=0.8,
                noise_level=0.2
            )
            signals.append(signal)
        batch_signals.append(signals)
    
    batch_signals = np.array(batch_signals)
    total_samples = batch_signals.size
    
    # Respect Nyquist frequency
    nyquist = config['fs'] / 2
    amp_end_hz = min(150, nyquist - 10)  # Leave 10Hz buffer
    
    # Initialize gPAC (done once in practice)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_start = time.time()
    pac_gpac = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=amp_end_hz,
        amp_n_bands=config['amp_n_bands'],
        trainable=False
    ).to(device)
    gpac_init_time = time.time() - init_start
    
    # Initialize TensorPAC (done once in practice)
    pha_edges = np.linspace(2, 20, config['pha_n_bands'] + 1)
    amp_edges = np.linspace(30, amp_end_hz, config['amp_n_bands'] + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    init_start = time.time()
    pac_tp = TensorPAC_Pac(
        idpac=(2, 0, 0),  # KLD method
        f_pha=pha_bands,
        f_amp=amp_bands,
        verbose=False
    )
    tp_init_time = time.time() - init_start
    
    # Time actual PAC computation (what matters in practice)
    _, gpac_time_mean, gpac_time_std = realistic_timing(
        "gpac", batch_signals, config, pac_gpac, n_runs=2, n_warmup=1
    )
    
    _, tp_time_mean, tp_time_std = realistic_timing(
        "tensorpac", batch_signals, config, pac_tp, n_runs=2, n_warmup=1
    )
    
    speedup = tp_time_mean / gpac_time_mean
    
    if verbose:
        print(f"  gPAC: {gpac_time_mean:.4f}±{gpac_time_std:.4f}s (init: {gpac_init_time:.3f}s)")
        print(f"  TensorPAC: {tp_time_mean:.4f}±{tp_time_std:.4f}s (init: {tp_init_time:.3f}s)")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Throughput - gPAC: {total_samples/gpac_time_mean:.0f} samples/s")
        print(f"  Throughput - TP: {total_samples/tp_time_mean:.0f} samples/s")
    
    return {
        'config': config,
        'total_samples': total_samples,
        'gpac_init_time': gpac_init_time,
        'gpac_time_mean': gpac_time_mean,
        'gpac_time_std': gpac_time_std,
        'tp_init_time': tp_init_time,
        'tp_time_mean': tp_time_mean,
        'tp_time_std': tp_time_std,
        'speedup': speedup,
    }

# Main execution
print("FAIR REAL-WORLD gPAC vs TensorPAC BENCHMARK")
print("=" * 80)
print(f"GPU: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
print(f"CPU: {REALISTIC_BASELINE['n_jobs']} cores (baseline)")
print(f"Comparison: 1 GPU vs multiple CPU cores")
print("=" * 80)

# Run baseline
print("\nBaseline Test:")
baseline_result = run_fair_benchmark(REALISTIC_BASELINE)

# Store results
all_results = []
all_results.append(baseline_result)

# Parameter variations
print("\n\nParameter Dependency Analysis:")
print("-" * 80)

for param_name, param_values in REALISTIC_VARIATIONS.items():
    print(f"\nVarying {param_name}:")
    
    for value in param_values:
        if value == REALISTIC_BASELINE.get(param_name):
            continue  # Skip baseline
        
        config = REALISTIC_BASELINE.copy()
        config[param_name] = value
        
        try:
            result = run_fair_benchmark(config, verbose=True)
            result['parameter_varied'] = param_name
            result['parameter_value'] = value
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR with {param_name}={value}: {str(e)}")

# Create visualization
print("\n\nCreating visualization...")
df = pd.DataFrame(all_results)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

plot_idx = 0
for param_name in REALISTIC_VARIATIONS.keys():
    ax = axes[plot_idx]
    
    # Filter data for this parameter
    param_data = df[df['parameter_varied'] == param_name]
    if len(param_data) == 0:
        plot_idx += 1
        continue
    
    # Add baseline point
    baseline_row = df.iloc[0].copy()
    baseline_row['parameter_value'] = REALISTIC_BASELINE[param_name]
    param_data = pd.concat([param_data, baseline_row.to_frame().T])
    param_data = param_data.sort_values('parameter_value')
    
    # Plot
    x = param_data['parameter_value']
    ax.errorbar(x, param_data['gpac_time_mean'], yerr=param_data['gpac_time_std'],
                fmt='b-o', label='gPAC', linewidth=2, markersize=8, capsize=5)
    ax.errorbar(x, param_data['tp_time_mean'], yerr=param_data['tp_time_std'],
                fmt='r-s', label='TensorPAC', linewidth=2, markersize=8, capsize=5)
    
    # Add speedup on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x, param_data['speedup'], 'g--^', label='Speedup', linewidth=2, markersize=8)
    ax2.set_ylabel('Speedup Factor', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Effect of {param_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    
    # Mark baseline
    ax.axvline(x=REALISTIC_BASELINE[param_name], color='gray', linestyle='--', alpha=0.5)
    
    # Log scale where appropriate
    if param_name in ['batch_size', 'n_channels']:
        ax.set_xscale('log', base=2)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plot_idx += 1

plt.suptitle('Fair Real-World gPAC vs TensorPAC Benchmark', fontsize=16)
plt.tight_layout()
mngs.io.save(plt.gcf(), "fair_benchmark_results.png")

# Save results
mngs.io.save(df, "fair_benchmark_results.csv")

# Print summary
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

speedup_mean = df['speedup'].mean()
speedup_median = df['speedup'].median()
gpac_faster = (df['speedup'] > 1).sum()
total_tests = len(df)

print(f"\nOverall Performance:")
print(f"  Mean speedup: {speedup_mean:.2f}x")
print(f"  Median speedup: {speedup_median:.2f}x")
print(f"  gPAC faster in: {gpac_faster}/{total_tests} configurations ({gpac_faster/total_tests*100:.1f}%)")

print(f"\nBaseline Performance (typical use case):")
print(f"  gPAC: {baseline_result['gpac_time_mean']:.4f}s")
print(f"  TensorPAC: {baseline_result['tp_time_mean']:.4f}s")
print(f"  Speedup: {baseline_result['speedup']:.2f}x")

# Find best/worst cases
best_config = df.loc[df['speedup'].idxmax()]
worst_config = df.loc[df['speedup'].idxmin()]

print(f"\nBest case for gPAC:")
print(f"  {best_config.get('parameter_varied', 'baseline')}={best_config.get('parameter_value', 'N/A')}")
print(f"  Speedup: {best_config['speedup']:.2f}x")

print(f"\nWorst case for gPAC:")
print(f"  {worst_config.get('parameter_varied', 'baseline')}={worst_config.get('parameter_value', 'N/A')}")
print(f"  Speedup: {worst_config['speedup']:.2f}x")

print("\n" + "=" * 80)
print("Results saved to: fair_benchmark_results.csv")
print("Visualization saved to: fair_benchmark_results.png")