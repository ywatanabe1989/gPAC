#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_parameter_dependency_analysis.py

"""
Parameter dependency analysis for gPAC vs TensorPAC
Uses baseline configuration and varies one parameter at a time
"""

import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorpac import Pac as TensorPAC_Pac
from gpac import SyntheticDataGenerator, PAC as gPAC_PAC
import mngs
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Baseline configuration from gPAC paper
BASELINE = {
    'batch_size': 2,
    'n_channels': 2,
    'duration': 1,  # t_sec
    'fs': 256,
    'pha_n_bands': 10,
    'amp_n_bands': 10,
    'trainable': False,
}

# Parameters to vary (one at a time)
VARIATIONS = {
    'batch_size': [1, 2, 4, 8, 16, 32, 64],
    'n_channels': [1, 2, 4, 8, 16, 32, 64],
    'duration': [0.5, 1, 2, 4, 8],  # seconds
    'fs': [128, 256, 512, 1024],
    'pha_n_bands': [5, 10, 30, 50, 70, 100],
    'amp_n_bands': [5, 10, 30, 50, 70, 100],
    'trainable': [False, True],
}

# Ground truth for synthetic signal
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8

def run_single_test(config, test_name=""):
    """Run a single test with given configuration"""
    n_samples = int(config['duration'] * config['fs'])
    
    # Generate batch signals
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
    batch_signals = np.array(batch_signals)
    
    # Test gPAC
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = torch.from_numpy(batch_signals).float().to(device)
    
    # Initialize gPAC
    init_start = time.time()
    pac_gpac = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=config['amp_n_bands'],
        trainable=config['trainable']
    ).to(device)
    gpac_init_time = time.time() - init_start
    
    # Warm-up
    with torch.no_grad():
        _ = pac_gpac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time computation (average of 5 runs)
    times = []
    for _ in range(5):
        start = time.time()
        if config['trainable']:
            output_gpac = pac_gpac(signal_torch)
        else:
            with torch.no_grad():
                output_gpac = pac_gpac(signal_torch)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    gpac_comp_time = np.mean(times)
    
    # Extract results
    if config['trainable']:
        pac_matrix_gpac = output_gpac['pac'].detach().cpu().numpy()
    else:
        pac_matrix_gpac = output_gpac['pac'].cpu().numpy()
    
    # Test TensorPAC (only for non-trainable)
    tp_comp_time = None
    speedup = None
    
    if not config['trainable']:
        # Create frequency bands
        pha_edges = np.linspace(2, 20, config['pha_n_bands'] + 1)
        amp_edges = np.linspace(30, 150, config['amp_n_bands'] + 1)
        pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
        
        # Process for TensorPAC
        signals_for_tp = batch_signals.reshape(-1, n_samples).T
        
        # Initialize and time TensorPAC
        pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
        
        times = []
        for _ in range(5):
            start = time.time()
            _ = pac_tp.filterfit(config['fs'], signals_for_tp, n_jobs=1)
            times.append(time.time() - start)
        tp_comp_time = np.mean(times)
        
        speedup = tp_comp_time / gpac_comp_time
    
    return {
        'config': config,
        'test_name': test_name,
        'gpac_init_time': gpac_init_time,
        'gpac_comp_time': gpac_comp_time,
        'tp_comp_time': tp_comp_time,
        'speedup': speedup,
        'total_samples': n_samples * config['n_channels'] * config['batch_size'],
    }

# Run baseline test
print("PARAMETER DEPENDENCY ANALYSIS")
print("=" * 80)
print("Running baseline configuration...")
baseline_result = run_single_test(BASELINE.copy(), "baseline")
print(f"Baseline - gPAC: {baseline_result['gpac_comp_time']:.4f}s, "
      f"TensorPAC: {baseline_result['tp_comp_time']:.4f}s, "
      f"Speedup: {baseline_result['speedup']:.2f}x")

# Store all results
all_results = {'baseline': baseline_result}

# Run parameter variations
print("\nRunning parameter variations...")
for param_name, param_values in VARIATIONS.items():
    print(f"\nVarying {param_name}:")
    param_results = []
    
    for value in param_values:
        # Skip if this is the baseline value
        if value == BASELINE[param_name]:
            param_results.append(baseline_result)
            continue
            
        # Create config with single parameter changed
        config = BASELINE.copy()
        config[param_name] = value
        
        # Skip combinations that don't make sense
        if param_name == 'trainable' and value == True and 'tp_comp_time' in baseline_result:
            # For trainable, we only test gPAC
            pass
        
        print(f"  {param_name}={value}...", end='', flush=True)
        try:
            result = run_single_test(config, f"{param_name}={value}")
            param_results.append(result)
            
            if result['speedup'] is not None:
                print(f" gPAC: {result['gpac_comp_time']:.4f}s, "
                      f"TensorPAC: {result['tp_comp_time']:.4f}s, "
                      f"Speedup: {result['speedup']:.2f}x")
            else:
                print(f" gPAC: {result['gpac_comp_time']:.4f}s (trainable)")
        except Exception as e:
            print(f" ERROR: {str(e)}")
            continue
    
    all_results[param_name] = param_results

# Create visualizations
print("\nCreating visualizations...")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

plot_idx = 0
for param_name, param_results in all_results.items():
    if param_name == 'baseline':
        continue
        
    ax = axes[plot_idx]
    
    # Extract data
    param_values = [r['config'][param_name] for r in param_results]
    gpac_times = [r['gpac_comp_time'] for r in param_results]
    tp_times = [r['tp_comp_time'] if r['tp_comp_time'] is not None else np.nan for r in param_results]
    speedups = [r['speedup'] if r['speedup'] is not None else np.nan for r in param_results]
    
    # Plot computation times
    ax2 = ax.twinx()
    
    # Times on left axis
    line1 = ax.plot(param_values, gpac_times, 'b-o', label='gPAC', linewidth=2, markersize=8)
    line2 = ax.plot(param_values, tp_times, 'r-s', label='TensorPAC', linewidth=2, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel('Computation Time (s)', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    
    # Speedup on right axis
    line3 = ax2.plot(param_values, speedups, 'g--^', label='Speedup', linewidth=2, markersize=8)
    ax2.set_ylabel('Speedup Factor', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    
    # Add baseline marker
    baseline_value = BASELINE[param_name]
    if baseline_value in param_values:
        idx = param_values.index(baseline_value)
        ax.axvline(x=baseline_value, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best', fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Effect of {param_name.replace("_", " ").title()}')
    
    # Log scale for appropriate parameters
    if param_name in ['batch_size', 'n_channels', 'fs']:
        ax.set_xscale('log', base=2)
    
    plot_idx += 1

# Summary plot
ax = axes[plot_idx]
ax.axis('off')

# Create summary table
summary_text = f"""PARAMETER DEPENDENCY SUMMARY

Baseline Configuration:
  Batch size: {BASELINE['batch_size']}
  Channels: {BASELINE['n_channels']}
  Duration: {BASELINE['duration']}s
  Sample rate: {BASELINE['fs']} Hz
  Phase bands: {BASELINE['pha_n_bands']}
  Amplitude bands: {BASELINE['amp_n_bands']}

Baseline Performance:
  gPAC: {baseline_result['gpac_comp_time']:.4f}s
  TensorPAC: {baseline_result['tp_comp_time']:.4f}s
  Speedup: {baseline_result['speedup']:.2f}x

Key Findings:
"""

# Find parameter with maximum impact
max_impact_param = None
max_speedup_range = 0

for param_name, param_results in all_results.items():
    if param_name == 'baseline':
        continue
    speedups = [r['speedup'] for r in param_results if r['speedup'] is not None]
    if speedups:
        speedup_range = max(speedups) - min(speedups)
        if speedup_range > max_speedup_range:
            max_speedup_range = speedup_range
            max_impact_param = param_name

summary_text += f"  Most impactful parameter: {max_impact_param}\n"
summary_text += f"  Speedup range: {min([r['speedup'] for r in all_results[max_impact_param] if r['speedup'] is not None]):.1f}x - "
summary_text += f"{max([r['speedup'] for r in all_results[max_impact_param] if r['speedup'] is not None]):.1f}x"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove unused subplots
for idx in range(plot_idx + 1, len(axes)):
    axes[idx].axis('off')

plt.suptitle('gPAC vs TensorPAC Parameter Dependency Analysis', fontsize=16)
plt.tight_layout()
mngs.io.save(plt.gcf(), "parameter_dependency_analysis.png")
print("Visualization saved")

# Save detailed results
results_data = []
for param_name, param_results in all_results.items():
    if param_name == 'baseline':
        continue
    for result in param_results:
        row = result['config'].copy()
        row.update({
            'parameter_varied': param_name,
            'gpac_init_time': result['gpac_init_time'],
            'gpac_comp_time': result['gpac_comp_time'],
            'tp_comp_time': result['tp_comp_time'],
            'speedup': result['speedup'],
            'total_samples': result['total_samples']
        })
        results_data.append(row)

df = pd.DataFrame(results_data)
mngs.io.save(df, "parameter_dependency_results.csv")
print("Detailed results saved")

# Print parameter sensitivity analysis
print("\n" + "="*80)
print("PARAMETER SENSITIVITY ANALYSIS")
print("="*80)

for param_name in ['batch_size', 'n_channels', 'duration', 'fs', 'pha_n_bands', 'amp_n_bands']:
    param_results = all_results.get(param_name, [])
    if not param_results:
        continue
        
    speedups = [r['speedup'] for r in param_results if r['speedup'] is not None]
    if speedups:
        print(f"\n{param_name.replace('_', ' ').title()}:")
        print(f"  Range tested: {min(r['config'][param_name] for r in param_results)} - "
              f"{max(r['config'][param_name] for r in param_results)}")
        print(f"  Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        
        # Find optimal value
        optimal_idx = np.argmax(speedups)
        optimal_value = param_results[optimal_idx]['config'][param_name]
        print(f"  Optimal value for speedup: {optimal_value} ({speedups[optimal_idx]:.2f}x)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")