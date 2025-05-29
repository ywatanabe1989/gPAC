#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_comprehensive_speed_analysis.py

"""
Comprehensive speed analysis of gPAC vs TensorPAC with varying parameters
Tests dependencies on: signal length, number of channels, frequency resolution
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
from itertools import product

# Test parameters from gPAC paper config
test_configs = {
    'batch_sizes': [1, 2, 4, 8, 16],  # batch_size from PARAMS.yaml
    'durations': [1, 2, 4, 8],  # seconds (t_sec from PARAMS.yaml)
    'n_channels': [2, 4, 8, 16, 32],  # (n_chs from PARAMS.yaml)
    'n_freq_bands': [(10, 10), (30, 30), (50, 50)],  # (n_pha_bands, n_amp_bands)
    'sample_rates': [256, 512, 1024],  # Hz (fs from PARAMS.yaml)
    'trainable': [True, False]  # Test both differentiable and static filters
}

# Limit combinations to avoid excessive runtime
max_tests = 120

# Fixed ground truth
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8

results = []

print("Running comprehensive speed analysis...")
print("=" * 80)

# Create subset of combinations for reasonable runtime
all_combinations = list(product(
    test_configs['batch_sizes'],
    test_configs['durations'],
    test_configs['n_channels'], 
    test_configs['n_freq_bands'],
    test_configs['sample_rates']
))

# Sample combinations to test
import random
random.seed(42)
if len(all_combinations) > max_tests:
    test_combinations = random.sample(all_combinations, max_tests)
else:
    test_combinations = all_combinations

print(f"Testing {len(test_combinations)} configurations out of {len(all_combinations)} total")

# Run selected combinations
for batch_size, duration, n_ch, (n_pha, n_amp), fs in test_combinations:
    n_samples = int(duration * fs)
    
    print(f"\nTesting: batch={batch_size}, duration={duration}s, channels={n_ch}, "
          f"bands=({n_pha},{n_amp}), fs={fs}Hz")
    
    # Generate signals for batch
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    batch_signals = []
    for b in range(batch_size):
        signals = []
        for _ in range(n_ch):
            signal = generator.generate_pac_signal(
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=coupling_strength,
                noise_level=0.1
            )
            signals.append(signal)
        batch_signals.append(signals)
    batch_signals = np.array(batch_signals)  # (batch, channels, time)
    
    # Test gPAC
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = torch.from_numpy(signals).float().unsqueeze(0).to(device)
    
    # Initialize gPAC
    init_start = time.time()
    pac_gpac = gPAC_PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_pha,
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=n_amp,
        trainable=False
    ).to(device)
    gpac_init_time = time.time() - init_start
    
    # Warm-up
    _ = pac_gpac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time computation
    comp_start = time.time()
    output_gpac = pac_gpac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    gpac_comp_time = time.time() - comp_start
    
    # Extract results
    pac_matrix_gpac = output_gpac['pac'].squeeze().cpu().numpy()
    if pac_matrix_gpac.ndim > 2:
        pac_matrix_gpac = pac_matrix_gpac.mean(axis=0)  # Average over channels
    
    pha_freqs_gpac = output_gpac['phase_frequencies'].cpu().numpy()
    amp_freqs_gpac = output_gpac['amplitude_frequencies'].cpu().numpy()
    
    # Find peak
    peak_idx = np.unravel_index(pac_matrix_gpac.argmax(), pac_matrix_gpac.shape)
    gpac_peak_phase = pha_freqs_gpac[peak_idx[0]]
    gpac_peak_amp = amp_freqs_gpac[peak_idx[1]]
    gpac_peak_value = pac_matrix_gpac.max()
    
    # Test TensorPAC
    # Create frequency bands
    pha_edges = np.linspace(2, 20, n_pha + 1)
    amp_edges = np.linspace(30, 150, n_amp + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    # Initialize TensorPAC
    init_start = time.time()
    pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
    tp_init_time = time.time() - init_start
    
    # Time computation
    comp_start = time.time()
    pac_matrix_tp = pac_tp.filterfit(fs, signals.T if n_ch > 1 else signals.squeeze(), n_jobs=1)
    tp_comp_time = time.time() - comp_start
    
    # Process TensorPAC output
    if pac_matrix_tp.ndim == 3:
        pac_matrix_tp = pac_matrix_tp.squeeze().T
    else:
        pac_matrix_tp = pac_matrix_tp.T
        
    if pac_matrix_tp.ndim > 2:
        pac_matrix_tp = pac_matrix_tp.mean(axis=0)
    
    # Find peak
    pha_freqs_tp = (pha_bands[:, 0] + pha_bands[:, 1]) / 2
    amp_freqs_tp = (amp_bands[:, 0] + amp_bands[:, 1]) / 2
    peak_idx_tp = np.unravel_index(pac_matrix_tp.argmax(), pac_matrix_tp.shape)
    tp_peak_phase = pha_freqs_tp[peak_idx_tp[0]]
    tp_peak_amp = amp_freqs_tp[peak_idx_tp[1]]
    tp_peak_value = pac_matrix_tp.max()
    
    # Store results
    result = {
        'duration': duration,
        'n_channels': n_ch,
        'n_pha_bands': n_pha,
        'n_amp_bands': n_amp,
        'sample_rate': fs,
        'n_samples': n_samples,
        'total_samples': n_samples * n_ch,
        'gpac_init_time': gpac_init_time,
        'gpac_comp_time': gpac_comp_time,
        'gpac_total_time': gpac_init_time + gpac_comp_time,
        'gpac_peak_phase': gpac_peak_phase,
        'gpac_peak_amp': gpac_peak_amp,
        'gpac_peak_value': gpac_peak_value,
        'gpac_phase_error': abs(gpac_peak_phase - true_phase_freq),
        'gpac_amp_error': abs(gpac_peak_amp - true_amp_freq),
        'tp_init_time': tp_init_time,
        'tp_comp_time': tp_comp_time,
        'tp_total_time': tp_init_time + tp_comp_time,
        'tp_peak_phase': tp_peak_phase,
        'tp_peak_amp': tp_peak_amp,
        'tp_peak_value': tp_peak_value,
        'tp_phase_error': abs(tp_peak_phase - true_phase_freq),
        'tp_amp_error': abs(tp_peak_amp - true_amp_freq),
        'speedup_comp': tp_comp_time / gpac_comp_time,
        'speedup_total': (tp_init_time + tp_comp_time) / (gpac_init_time + gpac_comp_time),
        'pac_matrices': {
            'gpac': pac_matrix_gpac,
            'tensorpac': pac_matrix_tp
        }
    }
    results.append(result)
    
    print(f"  Speedup: {result['speedup_comp']:.1f}x (computation), "
          f"{result['speedup_total']:.1f}x (total)")
    print(f"  Peak detection - gPAC: ({gpac_peak_phase:.1f}, {gpac_peak_amp:.1f}), "
          f"TensorPAC: ({tp_peak_phase:.1f}, {tp_peak_amp:.1f})")

# Create comprehensive visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Speedup vs total samples
ax1 = fig.add_subplot(gs[0, :2])
total_samples = [r['total_samples'] for r in results]
speedups = [r['speedup_comp'] for r in results]
colors = ['red' if r['n_pha_bands'] == 10 else 'blue' if r['n_pha_bands'] == 20 else 'green' 
          for r in results]
ax1.scatter(total_samples, speedups, c=colors, alpha=0.6, s=100)
ax1.set_xlabel('Total Samples (channels × time points)')
ax1.set_ylabel('Speedup Factor (computation)')
ax1.set_title('Speedup vs Data Size')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='10×10 bands'),
    Patch(facecolor='blue', alpha=0.6, label='20×20 bands'),
    Patch(facecolor='green', alpha=0.6, label='50×30 bands')
]
ax1.legend(handles=legend_elements)

# 2. Speedup vs frequency resolution
ax2 = fig.add_subplot(gs[0, 2:])
n_bands_total = [r['n_pha_bands'] * r['n_amp_bands'] for r in results]
ax2.scatter(n_bands_total, speedups, alpha=0.6, s=100)
ax2.set_xlabel('Total Frequency Bands (phase × amplitude)')
ax2.set_ylabel('Speedup Factor')
ax2.set_title('Speedup vs Frequency Resolution')
ax2.grid(True, alpha=0.3)

# 3. Accuracy comparison - Phase frequency
ax3 = fig.add_subplot(gs[1, :2])
gpac_phase_errors = [r['gpac_phase_error'] for r in results]
tp_phase_errors = [r['tp_phase_error'] for r in results]
x = np.arange(len(results))
width = 0.35
ax3.bar(x - width/2, gpac_phase_errors, width, label='gPAC', alpha=0.7)
ax3.bar(x + width/2, tp_phase_errors, width, label='TensorPAC', alpha=0.7)
ax3.set_xlabel('Test Configuration')
ax3.set_ylabel('Phase Frequency Error (Hz)')
ax3.set_title(f'Phase Detection Error (Ground Truth: {true_phase_freq} Hz)')
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)

# 4. Accuracy comparison - Amplitude frequency
ax4 = fig.add_subplot(gs[1, 2:])
gpac_amp_errors = [r['gpac_amp_error'] for r in results]
tp_amp_errors = [r['tp_amp_error'] for r in results]
ax4.bar(x - width/2, gpac_amp_errors, width, label='gPAC', alpha=0.7)
ax4.bar(x + width/2, tp_amp_errors, width, label='TensorPAC', alpha=0.7)
ax4.set_xlabel('Test Configuration')
ax4.set_ylabel('Amplitude Frequency Error (Hz)')
ax4.set_title(f'Amplitude Detection Error (Ground Truth: {true_amp_freq} Hz)')
ax4.legend()
ax4.grid(True, axis='y', alpha=0.3)

# 5. Time breakdown for different data sizes
ax5 = fig.add_subplot(gs[2, :2])
# Group by duration and channels
for n_ch in test_configs['n_channels']:
    ch_results = [r for r in results if r['n_channels'] == n_ch and r['n_pha_bands'] == 20]
    if ch_results:
        durations = [r['duration'] for r in ch_results]
        gpac_times = [r['gpac_comp_time'] for r in ch_results]
        tp_times = [r['tp_comp_time'] for r in ch_results]
        ax5.plot(durations, gpac_times, 'o-', label=f'gPAC ({n_ch}ch)', linewidth=2)
        ax5.plot(durations, tp_times, 's--', label=f'TensorPAC ({n_ch}ch)', linewidth=2)

ax5.set_xlabel('Signal Duration (seconds)')
ax5.set_ylabel('Computation Time (seconds)')
ax5.set_title('Computation Time vs Signal Duration')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# 6. PAC value comparison
ax6 = fig.add_subplot(gs[2, 2:])
gpac_values = [r['gpac_peak_value'] for r in results]
tp_values = [r['tp_peak_value'] for r in results]
ax6.scatter(tp_values, gpac_values, alpha=0.6, s=100)
ax6.set_xlabel('TensorPAC Peak Value')
ax6.set_ylabel('gPAC Peak Value')
ax6.set_title('Peak PAC Value Comparison')
ax6.grid(True, alpha=0.3)
# Add diagonal line
max_val = max(max(gpac_values), max(tp_values))
ax6.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
ax6.legend()

# 7. Example comodulograms for best accuracy case
# Find configuration with best combined accuracy
accuracy_scores = [r['gpac_phase_error'] + r['gpac_amp_error'] for r in results]
best_idx = np.argmin(accuracy_scores)
best_result = results[best_idx]

ax7 = fig.add_subplot(gs[3, 0])
im1 = ax7.imshow(best_result['pac_matrices']['gpac'].T, aspect='auto', origin='lower',
                 cmap='hot', extent=[2, 20, 30, 150])
ax7.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=1)
ax7.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=1)
ax7.set_title(f'gPAC (Best Config: {best_result["n_pha_bands"]}×{best_result["n_amp_bands"]})')
ax7.set_xlabel('Phase (Hz)')
ax7.set_ylabel('Amplitude (Hz)')
plt.colorbar(im1, ax=ax7)

ax8 = fig.add_subplot(gs[3, 1])
im2 = ax8.imshow(best_result['pac_matrices']['tensorpac'].T, aspect='auto', origin='lower',
                 cmap='hot', extent=[2, 20, 30, 150])
ax8.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=1)
ax8.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=1)
ax8.set_title('TensorPAC (Same Config)')
ax8.set_xlabel('Phase (Hz)')
ax8.set_ylabel('Amplitude (Hz)')
plt.colorbar(im2, ax=ax8)

# 8. Summary statistics table
ax9 = fig.add_subplot(gs[3, 2:])
ax9.axis('off')

# Calculate summary statistics
avg_speedup = np.mean([r['speedup_comp'] for r in results])
max_speedup = np.max([r['speedup_comp'] for r in results])
min_speedup = np.min([r['speedup_comp'] for r in results])

avg_gpac_phase_error = np.mean(gpac_phase_errors)
avg_tp_phase_error = np.mean(tp_phase_errors)
avg_gpac_amp_error = np.mean(gpac_amp_errors)
avg_tp_amp_error = np.mean(tp_amp_errors)

summary_text = f"""Summary Statistics:

Speedup (Computation):
  Average: {avg_speedup:.1f}x
  Maximum: {max_speedup:.1f}x
  Minimum: {min_speedup:.1f}x

Accuracy (Average Error):
  Phase - gPAC: {avg_gpac_phase_error:.2f} Hz, TensorPAC: {avg_tp_phase_error:.2f} Hz
  Amplitude - gPAC: {avg_gpac_amp_error:.2f} Hz, TensorPAC: {avg_tp_amp_error:.2f} Hz

Best Configuration (Speed):
  {max([r for r in results], key=lambda x: x['speedup_comp'])['duration']}s, 
  {max([r for r in results], key=lambda x: x['speedup_comp'])['n_channels']} channels,
  {max([r for r in results], key=lambda x: x['speedup_comp'])['n_pha_bands']}×{max([r for r in results], key=lambda x: x['speedup_comp'])['n_amp_bands']} bands
  
PAC Value Scaling:
  gPAC/TensorPAC ratio: {np.mean([r['gpac_peak_value']/r['tp_peak_value'] for r in results]):.3f}
"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive gPAC vs TensorPAC Analysis', fontsize=16)
mngs.io.save(plt.gcf(), "comprehensive_speed_analysis.png")
print("Visualization saved")

# Save detailed results
import pandas as pd
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'pac_matrices'} for r in results])
mngs.io.save(df, "comprehensive_speed_analysis_results.csv")
print("Detailed results saved")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"Average speedup: {avg_speedup:.1f}x")
print(f"gPAC is faster in {sum(1 for r in results if r['speedup_comp'] > 1)}/{len(results)} configurations")