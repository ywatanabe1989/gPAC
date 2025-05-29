#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_comprehensive_analysis_final_fixed.py

"""
Comprehensive analysis of gPAC vs TensorPAC with multiple test scenarios
Tests both static and trainable filters across various configurations
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
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Test parameters based on gPAC paper config
test_configs = {
    'batch_sizes': [1, 2, 4, 8, 16],
    'durations': [1, 2, 4, 8],  # seconds
    'n_channels': [2, 4, 8, 16, 32],
    'n_freq_bands': [(10, 10), (30, 30), (50, 50)],  # (n_pha, n_amp)
    'sample_rates': [256, 512, 1024],  # Hz
    'trainable': [True, False]  # Both filter types
}

# Ground truth for synthetic signal
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8

# Create test combinations
all_combinations = list(product(
    test_configs['batch_sizes'],
    test_configs['durations'],
    test_configs['n_channels'], 
    test_configs['n_freq_bands'],
    test_configs['sample_rates'],
    test_configs['trainable']
))

# Sample for reasonable runtime
max_tests = 50
import random
random.seed(42)
if len(all_combinations) > max_tests:
    test_combinations = random.sample(all_combinations, max_tests)
else:
    test_combinations = all_combinations

results = []
skipped = 0

print("COMPREHENSIVE gPAC vs TensorPAC ANALYSIS")
print("=" * 80)
print(f"Testing {len(test_combinations)} configurations out of {len(all_combinations)} total")
print(f"Ground truth: Phase={true_phase_freq} Hz, Amplitude={true_amp_freq} Hz")
print("=" * 80)

# Run tests
for idx, (batch_size, duration, n_ch, (n_pha, n_amp), fs, trainable) in enumerate(test_combinations):
    n_samples = int(duration * fs)
    
    print(f"\n[{idx+1}/{len(test_combinations)}] Testing: batch={batch_size}, duration={duration}s, "
          f"channels={n_ch}, bands=({n_pha},{n_amp}), fs={fs}Hz, trainable={trainable}")
    
    try:
        # Generate batch signals
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
        signal_torch = torch.from_numpy(batch_signals).float().to(device)
        
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
            trainable=trainable
        ).to(device)
        gpac_init_time = time.time() - init_start
        
        # Warm-up
        with torch.no_grad():
            _ = pac_gpac(signal_torch)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time computation (average of 3 runs)
        times = []
        for _ in range(3):
            start = time.time()
            if trainable:
                output_gpac = pac_gpac(signal_torch)
            else:
                with torch.no_grad():
                    output_gpac = pac_gpac(signal_torch)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
        gpac_comp_time = np.mean(times)
        
        # Extract results
        if trainable:
            pac_matrix_gpac = output_gpac['pac'].detach().cpu().numpy()
        else:
            pac_matrix_gpac = output_gpac['pac'].cpu().numpy()
            
        if pac_matrix_gpac.ndim > 2:
            pac_matrix_gpac = pac_matrix_gpac.reshape(-1, n_pha, n_amp).mean(axis=0)
        
        pha_freqs_gpac = output_gpac['phase_frequencies'].cpu().numpy()
        amp_freqs_gpac = output_gpac['amplitude_frequencies'].cpu().numpy()
        
        # Find peak
        peak_idx = np.unravel_index(pac_matrix_gpac.argmax(), pac_matrix_gpac.shape)
        gpac_peak_phase = pha_freqs_gpac[peak_idx[0]]
        gpac_peak_amp = amp_freqs_gpac[peak_idx[1]]
        gpac_peak_value = pac_matrix_gpac.max()
        
        # Test TensorPAC (only for static filter comparison)
        if not trainable:
            # Create frequency bands
            pha_edges = np.linspace(2, 20, n_pha + 1)
            amp_edges = np.linspace(30, 150, n_amp + 1)
            pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
            amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
            
            # Process for TensorPAC
            signals_for_tp = batch_signals.reshape(-1, n_samples).T  # (time, batch*channels)
            
            # Initialize TensorPAC
            init_start = time.time()
            pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
            tp_init_time = time.time() - init_start
            
            # Time computation
            times = []
            for _ in range(3):
                start = time.time()
                pac_matrix_tp = pac_tp.filterfit(fs, signals_for_tp, n_jobs=1)
                times.append(time.time() - start)
            tp_comp_time = np.mean(times)
            
            # Process output
            if pac_matrix_tp.ndim == 3:
                pac_matrix_tp = pac_matrix_tp.squeeze().T.reshape(-1, n_pha, n_amp).mean(axis=0)
            else:
                pac_matrix_tp = pac_matrix_tp.T
            
            # Find peak
            pha_freqs_tp = (pha_bands[:, 0] + pha_bands[:, 1]) / 2
            amp_freqs_tp = (amp_bands[:, 0] + amp_bands[:, 1]) / 2
            peak_idx_tp = np.unravel_index(pac_matrix_tp.argmax(), pac_matrix_tp.shape)
            tp_peak_phase = pha_freqs_tp[peak_idx_tp[0]]
            tp_peak_amp = amp_freqs_tp[peak_idx_tp[1]]
            tp_peak_value = pac_matrix_tp.max()
        else:
            tp_init_time = tp_comp_time = None
            tp_peak_phase = tp_peak_amp = tp_peak_value = None
            pac_matrix_tp = None
        
        # Store results
        result = {
            'batch_size': batch_size,
            'duration': duration,
            'n_channels': n_ch,
            'n_pha_bands': n_pha,
            'n_amp_bands': n_amp,
            'sample_rate': fs,
            'trainable': trainable,
            'n_samples': n_samples,
            'total_samples': n_samples * n_ch * batch_size,
            'gpac_init_time': gpac_init_time,
            'gpac_comp_time': gpac_comp_time,
            'gpac_total_time': gpac_init_time + gpac_comp_time,
            'gpac_peak_phase': gpac_peak_phase,
            'gpac_peak_amp': gpac_peak_amp,
            'gpac_peak_value': gpac_peak_value,
            'gpac_phase_error': abs(gpac_peak_phase - true_phase_freq),
            'gpac_amp_error': abs(gpac_peak_amp - true_amp_freq),
        }
        
        if not trainable:
            result.update({
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
            })
        
        results.append(result)
        
        if not trainable:
            print(f"  Speedup: {result['speedup_comp']:.1f}x (computation), {result['speedup_total']:.1f}x (total)")
            print(f"  Peak detection - gPAC: ({gpac_peak_phase:.1f}, {gpac_peak_amp:.1f}), "
                  f"TensorPAC: ({tp_peak_phase:.1f}, {tp_peak_amp:.1f})")
        else:
            print(f"  gPAC trainable: time={gpac_comp_time:.4f}s, peak=({gpac_peak_phase:.1f}, {gpac_peak_amp:.1f})")
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        skipped += 1
        continue

if skipped > 0:
    print(f"\nSkipped {skipped} configurations due to errors")

# Create comprehensive visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# Filter results
static_results = [r for r in results if not r['trainable'] and 'speedup_comp' in r]
trainable_results = [r for r in results if r['trainable']]

# 1. Speedup vs total samples
ax1 = fig.add_subplot(gs[0, :2])
if static_results:
    total_samples = [r['total_samples'] for r in static_results]
    speedups = [r['speedup_comp'] for r in static_results]
    colors = ['red' if r['n_pha_bands'] == 10 else 'blue' if r['n_pha_bands'] == 30 else 'green' 
              for r in static_results]
    scatter = ax1.scatter(total_samples, speedups, c=colors, alpha=0.6, s=100)
    ax1.set_xlabel('Total Samples (batch × channels × time)')
    ax1.set_ylabel('Speedup Factor')
    ax1.set_title('gPAC Speedup vs Data Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='10×10 bands'),
        Patch(facecolor='blue', alpha=0.6, label='30×30 bands'),
        Patch(facecolor='green', alpha=0.6, label='50×50 bands')
    ]
    ax1.legend(handles=legend_elements)

# 2. Static vs Trainable comparison
ax2 = fig.add_subplot(gs[0, 2:])
config_keys = ['batch_size', 'duration', 'n_channels', 'n_pha_bands', 'sample_rate']
static_times = []
trainable_times = []
labels = []

for sr in static_results[:10]:  # Sample configurations
    config = tuple(sr[k] for k in config_keys)
    tr = next((r for r in trainable_results if all(r[k] == sr[k] for k in config_keys)), None)
    if tr:
        static_times.append(sr['gpac_comp_time'])
        trainable_times.append(tr['gpac_comp_time'])
        labels.append(f"B{sr['batch_size']}_C{sr['n_channels']}_F{sr['n_pha_bands']}")

if static_times:
    x = np.arange(len(static_times))
    width = 0.35
    ax2.bar(x - width/2, static_times, width, label='Static', alpha=0.7)
    ax2.bar(x + width/2, trainable_times, width, label='Trainable', alpha=0.7)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Static vs Trainable Filter Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_yscale('log')

# 3. Accuracy comparison - Phase
ax3 = fig.add_subplot(gs[1, :2])
if static_results:
    gpac_phase_errors_static = [r['gpac_phase_error'] for r in static_results]
    gpac_phase_errors_train = [r['gpac_phase_error'] for r in trainable_results]
    tp_phase_errors = [r['tp_phase_error'] for r in static_results]
    
    data_to_plot = [gpac_phase_errors_static, gpac_phase_errors_train, tp_phase_errors]
    ax3.boxplot(data_to_plot, labels=['gPAC\n(Static)', 'gPAC\n(Trainable)', 'TensorPAC'])
    ax3.set_ylabel('Phase Frequency Error (Hz)')
    ax3.set_title(f'Phase Detection Error (Truth: {true_phase_freq} Hz)')
    ax3.grid(True, axis='y', alpha=0.3)

# 4. Speedup by frequency resolution
ax4 = fig.add_subplot(gs[1, 2:])
for n_bands in [10, 30, 50]:
    band_results = [r for r in static_results if r['n_pha_bands'] == n_bands]
    if band_results:
        durations = sorted(set(r['duration'] for r in band_results))
        avg_speedups = []
        for d in durations:
            d_results = [r['speedup_comp'] for r in band_results if r['duration'] == d]
            if d_results:
                avg_speedups.append(np.mean(d_results))
        if avg_speedups:
            ax4.plot(durations, avg_speedups, 'o-', label=f'{n_bands}×{n_bands} bands', linewidth=2, markersize=8)

ax4.set_xlabel('Signal Duration (seconds)')
ax4.set_ylabel('Average Speedup Factor')
ax4.set_title('Speedup vs Duration by Frequency Resolution')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# 5. PAC value scaling
ax5 = fig.add_subplot(gs[2, :2])
if static_results:
    gpac_values = [r['gpac_peak_value'] for r in static_results]
    tp_values = [r['tp_peak_value'] for r in static_results]
    ax5.scatter(tp_values, gpac_values, alpha=0.6, s=100)
    ax5.set_xlabel('TensorPAC Peak Value')
    ax5.set_ylabel('gPAC Peak Value')
    ax5.set_title('Peak PAC Value Comparison')
    ax5.grid(True, alpha=0.3)
    
    # Add diagonal and scaling
    max_val = max(max(gpac_values), max(tp_values))
    ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    scaling = np.mean([g/t for g, t in zip(gpac_values, tp_values) if t > 0])
    ax5.plot([0, max_val], [0, max_val*scaling], 'r-', alpha=0.5, label=f'y={scaling:.3f}x')
    ax5.legend()

# 6. Performance heatmap
ax6 = fig.add_subplot(gs[2, 2:])
if static_results:
    # Create pivot table
    channels = sorted(set(r['n_channels'] for r in static_results))
    durations = sorted(set(r['duration'] for r in static_results))
    pivot_data = np.zeros((len(channels), len(durations)))
    
    for i, ch in enumerate(channels):
        for j, dur in enumerate(durations):
            matching = [r['speedup_comp'] for r in static_results 
                       if r['n_channels'] == ch and r['duration'] == dur]
            pivot_data[i, j] = np.mean(matching) if matching else np.nan
    
    im = ax6.imshow(pivot_data, aspect='auto', cmap='RdYlGn', vmin=0)
    ax6.set_yticks(range(len(channels)))
    ax6.set_yticklabels(channels)
    ax6.set_xticks(range(len(durations)))
    ax6.set_xticklabels(durations)
    ax6.set_ylabel('Number of Channels')
    ax6.set_xlabel('Duration (seconds)')
    ax6.set_title('Speedup Heatmap (gPAC vs TensorPAC)')
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Speedup Factor')

# 7-8. Example comodulograms
if static_results and 'pac_matrices' in static_results[0]:
    # Find best accuracy case
    accuracy_scores = [r['gpac_phase_error'] + r['gpac_amp_error'] for r in static_results]
    best_idx = np.argmin(accuracy_scores)
    best_result = static_results[best_idx]
    
    # gPAC comodulogram
    ax7 = fig.add_subplot(gs[3, 0])
    im1 = ax7.imshow(best_result['pac_matrices']['gpac'].T, aspect='auto', origin='lower',
                     cmap='hot', extent=[2, 20, 30, 150])
    ax7.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=1)
    ax7.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=1)
    ax7.set_title(f'gPAC (Config: {best_result["n_pha_bands"]}×{best_result["n_amp_bands"]})')
    ax7.set_xlabel('Phase (Hz)')
    ax7.set_ylabel('Amplitude (Hz)')
    plt.colorbar(im1, ax=ax7)
    
    # TensorPAC comodulogram
    ax8 = fig.add_subplot(gs[3, 1])
    im2 = ax8.imshow(best_result['pac_matrices']['tensorpac'].T, aspect='auto', origin='lower',
                     cmap='hot', extent=[2, 20, 30, 150])
    ax8.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=1)
    ax8.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=1)
    ax8.set_title('TensorPAC (Same Config)')
    ax8.set_xlabel('Phase (Hz)')
    ax8.set_ylabel('Amplitude (Hz)')
    plt.colorbar(im2, ax=ax8)

# 9. Summary statistics
ax9 = fig.add_subplot(gs[3, 2:])
ax9.axis('off')

if static_results:
    avg_speedup = np.mean([r['speedup_comp'] for r in static_results])
    max_speedup = np.max([r['speedup_comp'] for r in static_results])
    min_speedup = np.min([r['speedup_comp'] for r in static_results])
    
    best_speed_config = max(static_results, key=lambda x: x['speedup_comp'])
    
    avg_overhead = 0
    if trainable_results:
        avg_overhead = np.mean([(tr['gpac_comp_time'] - sr['gpac_comp_time'])/sr['gpac_comp_time']*100 
                               for sr, tr in zip(static_results[:len(trainable_results)], trainable_results)])
    
    summary_text = f"""SUMMARY STATISTICS:

Speedup (Computation):
  Average: {avg_speedup:.1f}x
  Maximum: {max_speedup:.1f}x
  Minimum: {min_speedup:.1f}x
  
Best Configuration:
  {best_speed_config['duration']}s, {best_speed_config['n_channels']} channels, 
  {best_speed_config['n_pha_bands']}×{best_speed_config['n_amp_bands']} bands
  Batch size: {best_speed_config['batch_size']}

Accuracy (Average Error):
  gPAC Phase: {np.mean([r['gpac_phase_error'] for r in static_results]):.2f} Hz
  TensorPAC Phase: {np.mean([r['tp_phase_error'] for r in static_results]):.2f} Hz
  
PAC Value Scaling:
  gPAC/TensorPAC ratio: {scaling:.3f}
  
Trainable Filter Overhead: {avg_overhead:.1f}%
"""
else:
    summary_text = "No comparison results available"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive gPAC vs TensorPAC Analysis', fontsize=16)
mngs.io.save(plt.gcf(), "comprehensive_analysis_final.png")
print("Visualization saved")

# Save detailed results
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'pac_matrices'} for r in results])
mngs.io.save(df, "comprehensive_analysis_results.csv")
print("Detailed results saved")

# Create a focused comparison for the paper
if static_results:
    # Find representative configurations
    small_config = next((r for r in static_results if r['total_samples'] < 10000), None)
    medium_config = next((r for r in static_results if 10000 <= r['total_samples'] < 100000), None)
    large_config = next((r for r in static_results if r['total_samples'] >= 100000), None)
    
    print("\n" + "="*80)
    print("KEY RESULTS FOR PAPER:")
    print("="*80)
    
    if small_config:
        print(f"\nSmall data ({small_config['n_channels']} ch, {small_config['duration']}s):")
        print(f"  gPAC: {small_config['gpac_comp_time']:.4f}s")
        print(f"  TensorPAC: {small_config['tp_comp_time']:.4f}s")
        print(f"  Speedup: {small_config['speedup_comp']:.2f}x")
    
    if medium_config:
        print(f"\nMedium data ({medium_config['n_channels']} ch, {medium_config['duration']}s):")
        print(f"  gPAC: {medium_config['gpac_comp_time']:.4f}s")
        print(f"  TensorPAC: {medium_config['tp_comp_time']:.4f}s")
        print(f"  Speedup: {medium_config['speedup_comp']:.2f}x")
    
    if large_config:
        print(f"\nLarge data ({large_config['n_channels']} ch, {large_config['duration']}s):")
        print(f"  gPAC: {large_config['gpac_comp_time']:.4f}s")
        print(f"  TensorPAC: {large_config['tp_comp_time']:.4f}s")
        print(f"  Speedup: {large_config['speedup_comp']:.2f}x")
    
    print(f"\nOverall:")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  gPAC faster in {sum(1 for r in static_results if r['speedup_comp'] > 1)}/{len(static_results)} configurations")
    print(f"  Trainable filter overhead: {avg_overhead:.1f}%")
    print(f"  PAC value scaling factor: {scaling:.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"Total configurations tested: {len(results)}")
print(f"Results saved to: comprehensive_analysis_results.csv")
print(f"Visualization saved to: comprehensive_analysis_final.png")