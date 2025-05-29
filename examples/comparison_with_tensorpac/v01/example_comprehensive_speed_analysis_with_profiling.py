#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_comprehensive_speed_analysis_with_profiling.py

"""
Comprehensive speed analysis of gPAC vs TensorPAC with profiling
Tests both static and differentiable filters with CPU/GPU usage tracking
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
import psutil
import GPUtil
import threading
import pandas as pd

class ResourceMonitor:
    """Monitor CPU and GPU usage during computation"""
    def __init__(self):
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return {
            'cpu_usage': np.array(self.cpu_usage),
            'gpu_usage': np.array(self.gpu_usage),
            'gpu_memory': np.array(self.gpu_memory),
            'timestamps': np.array(self.timestamps)
        }
        
    def _monitor(self):
        """Monitor resources in background"""
        start_time = time.time()
        while self.monitoring:
            # CPU usage
            cpu = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu)
            
            # GPU usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_usage.append(gpu.load * 100)
                    self.gpu_memory.append(gpu.memoryUsed)
                else:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
            except:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
                
            self.timestamps.append(time.time() - start_time)
            time.sleep(0.1)  # Sample every 100ms

# Test parameters from gPAC paper config
test_configs = {
    'batch_sizes': [1, 4, 16],  # Reduced for profiling
    'durations': [1, 4, 8],  # seconds
    'n_channels': [4, 16, 32],  # channels
    'n_freq_bands': [(10, 10), (30, 30), (50, 50)],  # (n_pha_bands, n_amp_bands)
    'sample_rates': [256, 1024],  # Hz
    'trainable': [True, False]  # Both filter types
}

# Fixed ground truth
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8

results = []
monitor = ResourceMonitor()

print("Running comprehensive speed analysis with profiling...")
print("=" * 80)

# Create subset of combinations
all_combinations = list(product(
    test_configs['batch_sizes'],
    test_configs['durations'],
    test_configs['n_channels'], 
    test_configs['n_freq_bands'],
    test_configs['sample_rates'],
    test_configs['trainable']
))

# Limit tests for reasonable runtime
max_tests = 50
import random
random.seed(42)
if len(all_combinations) > max_tests:
    test_combinations = random.sample(all_combinations, max_tests)
else:
    test_combinations = all_combinations

print(f"Testing {len(test_combinations)} configurations out of {len(all_combinations)} total")

# Run selected combinations
for idx, (batch_size, duration, n_ch, (n_pha, n_amp), fs, trainable) in enumerate(test_combinations):
    n_samples = int(duration * fs)
    
    print(f"\n[{idx+1}/{len(test_combinations)}] Testing: batch={batch_size}, duration={duration}s, "
          f"channels={n_ch}, bands=({n_pha},{n_amp}), fs={fs}Hz, trainable={trainable}")
    
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
    
    # Test gPAC with resource monitoring
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
        trainable=trainable  # Use trainable parameter
    ).to(device)
    gpac_init_time = time.time() - init_start
    
    # Warm-up
    _ = pac_gpac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time computation with profiling
    monitor.start()
    comp_start = time.time()
    output_gpac = pac_gpac(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    gpac_comp_time = time.time() - comp_start
    gpac_resources = monitor.stop()
    
    # Extract results
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
    
    # Test TensorPAC (only for trainable=False to avoid redundancy)
    if not trainable:
        # Create frequency bands
        pha_edges = np.linspace(2, 20, n_pha + 1)
        amp_edges = np.linspace(30, 150, n_amp + 1)
        pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
        
        # Process batch for TensorPAC
        signals_for_tp = batch_signals.reshape(-1, n_samples).T  # (time, batch*channels)
        
        # Initialize TensorPAC
        init_start = time.time()
        pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
        tp_init_time = time.time() - init_start
        
        # Time computation with profiling
        monitor.start()
        comp_start = time.time()
        pac_matrix_tp = pac_tp.filterfit(fs, signals_for_tp, n_jobs=1)
        tp_comp_time = time.time() - comp_start
        tp_resources = monitor.stop()
        
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
        # Set TensorPAC values to None for trainable=True
        tp_init_time = tp_comp_time = None
        tp_peak_phase = tp_peak_amp = tp_peak_value = None
        tp_resources = None
    
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
        'gpac_cpu_usage': gpac_resources['cpu_usage'].mean(),
        'gpac_gpu_usage': gpac_resources['gpu_usage'].mean(),
        'gpac_gpu_memory': gpac_resources['gpu_memory'].mean(),
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
            'tp_cpu_usage': tp_resources['cpu_usage'].mean(),
            'tp_gpu_usage': tp_resources['gpu_usage'].mean(),
            'tp_gpu_memory': tp_resources['gpu_memory'].mean(),
        })
    
    results.append(result)
    
    if not trainable:
        print(f"  Speedup: {result['speedup_comp']:.1f}x (computation)")
        print(f"  Resource usage - gPAC: CPU={result['gpac_cpu_usage']:.1f}%, GPU={result['gpac_gpu_usage']:.1f}%")
        print(f"  Resource usage - TensorPAC: CPU={result['tp_cpu_usage']:.1f}%")

# Create comprehensive visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)

# Filter results for plotting
static_results = [r for r in results if not r['trainable'] and 'speedup_comp' in r]
trainable_results = [r for r in results if r['trainable']]

# 1. Speedup vs total samples (Static only)
ax1 = fig.add_subplot(gs[0, :2])
total_samples = [r['total_samples'] for r in static_results]
speedups = [r['speedup_comp'] for r in static_results]
colors = ['red' if r['n_pha_bands'] == 10 else 'blue' if r['n_pha_bands'] == 30 else 'green' 
          for r in static_results]
ax1.scatter(total_samples, speedups, c=colors, alpha=0.6, s=100)
ax1.set_xlabel('Total Samples (batch × channels × time)')
ax1.set_ylabel('Speedup Factor')
ax1.set_title('gPAC Speedup vs Data Size (Static Filters)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# 2. Static vs Trainable comparison
ax2 = fig.add_subplot(gs[0, 2:])
# Group by configuration
config_keys = ['batch_size', 'duration', 'n_channels', 'n_pha_bands', 'sample_rate']
static_times = []
trainable_times = []
labels = []

for sr in static_results[:10]:  # Sample 10 configurations
    config = tuple(sr[k] for k in config_keys)
    # Find matching trainable result
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

# 3. Resource Usage Comparison
ax3 = fig.add_subplot(gs[1, :2])
if static_results:
    gpac_cpu = [r['gpac_cpu_usage'] for r in static_results]
    gpac_gpu = [r['gpac_gpu_usage'] for r in static_results]
    tp_cpu = [r['tp_cpu_usage'] for r in static_results]
    
    ax3.scatter(gpac_cpu, gpac_gpu, label='gPAC', alpha=0.6, s=100)
    ax3.scatter(tp_cpu, [0]*len(tp_cpu), label='TensorPAC', alpha=0.6, s=100)
    ax3.set_xlabel('CPU Usage (%)')
    ax3.set_ylabel('GPU Usage (%)')
    ax3.set_title('Resource Usage Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4. Accuracy comparison - Phase frequency
ax4 = fig.add_subplot(gs[1, 2:])
gpac_phase_errors_static = [r['gpac_phase_error'] for r in static_results]
gpac_phase_errors_train = [r['gpac_phase_error'] for r in trainable_results]
tp_phase_errors = [r['tp_phase_error'] for r in static_results]

# Box plot
data_to_plot = [gpac_phase_errors_static, gpac_phase_errors_train, tp_phase_errors]
ax4.boxplot(data_to_plot, labels=['gPAC\n(Static)', 'gPAC\n(Trainable)', 'TensorPAC'])
ax4.set_ylabel('Phase Frequency Error (Hz)')
ax4.set_title(f'Phase Detection Error Distribution (Truth: {true_phase_freq} Hz)')
ax4.grid(True, axis='y', alpha=0.3)

# 5. Speedup by frequency resolution
ax5 = fig.add_subplot(gs[2, :2])
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
            ax5.plot(durations, avg_speedups, 'o-', label=f'{n_bands}×{n_bands} bands', linewidth=2)

ax5.set_xlabel('Signal Duration (seconds)')
ax5.set_ylabel('Average Speedup Factor')
ax5.set_title('Speedup vs Duration by Frequency Resolution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. PAC value scaling
ax6 = fig.add_subplot(gs[2, 2:])
gpac_values = [r['gpac_peak_value'] for r in static_results]
tp_values = [r['tp_peak_value'] for r in static_results]
ax6.scatter(tp_values, gpac_values, alpha=0.6, s=100)
ax6.set_xlabel('TensorPAC Peak Value')
ax6.set_ylabel('gPAC Peak Value')
ax6.set_title('Peak PAC Value Comparison')
ax6.grid(True, alpha=0.3)
# Add diagonal and scaling factor
max_val = max(max(gpac_values), max(tp_values))
ax6.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
# Calculate average scaling
scaling = np.mean([g/t for g, t in zip(gpac_values, tp_values) if t > 0])
ax6.plot([0, max_val], [0, max_val*scaling], 'r-', alpha=0.5, label=f'y={scaling:.3f}x')
ax6.legend()

# 7. Performance heatmap
ax7 = fig.add_subplot(gs[3, :2])
# Create pivot table for heatmap
pivot_data = []
for n_ch in sorted(set(r['n_channels'] for r in static_results)):
    row = []
    for dur in sorted(set(r['duration'] for r in static_results)):
        matching = [r['speedup_comp'] for r in static_results 
                   if r['n_channels'] == n_ch and r['duration'] == dur]
        row.append(np.mean(matching) if matching else np.nan)
    pivot_data.append(row)

im = ax7.imshow(pivot_data, aspect='auto', cmap='RdYlGn', vmin=0)
ax7.set_yticks(range(len(pivot_data)))
ax7.set_yticklabels(sorted(set(r['n_channels'] for r in static_results)))
ax7.set_xticks(range(len(pivot_data[0])))
ax7.set_xticklabels(sorted(set(r['duration'] for r in static_results)))
ax7.set_ylabel('Number of Channels')
ax7.set_xlabel('Duration (seconds)')
ax7.set_title('Speedup Heatmap')
plt.colorbar(im, ax=ax7, label='Speedup Factor')

# 8. GPU Memory Usage
ax8 = fig.add_subplot(gs[3, 2:])
gpu_mem = [r['gpac_gpu_memory'] for r in results if r['gpac_gpu_memory'] > 0]
total_samples_gpu = [r['total_samples'] for r in results if r['gpac_gpu_memory'] > 0]
if gpu_mem:
    ax8.scatter(total_samples_gpu, gpu_mem, alpha=0.6, s=100)
    ax8.set_xlabel('Total Samples')
    ax8.set_ylabel('GPU Memory (MB)')
    ax8.set_title('GPU Memory Usage vs Data Size')
    ax8.set_xscale('log')
    ax8.grid(True, alpha=0.3)

# 9. Summary statistics table
ax9 = fig.add_subplot(gs[4, :])
ax9.axis('off')

# Calculate summary statistics
if static_results:
    avg_speedup = np.mean([r['speedup_comp'] for r in static_results])
    max_speedup = np.max([r['speedup_comp'] for r in static_results])
    min_speedup = np.min([r['speedup_comp'] for r in static_results])
    
    avg_gpac_phase_error_static = np.mean(gpac_phase_errors_static)
    avg_gpac_phase_error_train = np.mean(gpac_phase_errors_train)
    avg_tp_phase_error = np.mean(tp_phase_errors)
    
    best_speed_config = max(static_results, key=lambda x: x['speedup_comp'])
    
    summary_text = f"""Summary Statistics:

Speedup (Computation):
  Average: {avg_speedup:.1f}x
  Maximum: {max_speedup:.1f}x (at {best_speed_config['duration']}s, {best_speed_config['n_channels']} channels, {best_speed_config['n_pha_bands']}×{best_speed_config['n_amp_bands']} bands)
  Minimum: {min_speedup:.1f}x

Accuracy (Average Phase Error):
  gPAC Static: {avg_gpac_phase_error_static:.2f} Hz
  gPAC Trainable: {avg_gpac_phase_error_train:.2f} Hz
  TensorPAC: {avg_tp_phase_error:.2f} Hz

Resource Usage (Average):
  gPAC: CPU={np.mean([r['gpac_cpu_usage'] for r in results]):.1f}%, GPU={np.mean([r['gpac_gpu_usage'] for r in results]):.1f}%
  TensorPAC: CPU={np.mean([r['tp_cpu_usage'] for r in static_results]):.1f}%

PAC Value Scaling:
  gPAC/TensorPAC ratio: {scaling:.3f}

Filter Type Comparison:
  Static filter avg time: {np.mean([r['gpac_comp_time'] for r in static_results]):.4f}s
  Trainable filter avg time: {np.mean([r['gpac_comp_time'] for r in trainable_results]):.4f}s
  Overhead: {(np.mean([r['gpac_comp_time'] for r in trainable_results]) / np.mean([r['gpac_comp_time'] for r in static_results]) - 1) * 100:.1f}%
"""
else:
    summary_text = "No static filter results available for comparison"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive gPAC Analysis with Resource Profiling', fontsize=16)
mngs.io.save(plt.gcf(), "comprehensive_speed_analysis_with_profiling.png")
print("Visualization saved")

# Save detailed results
df = pd.DataFrame(results)
mngs.io.save(df, "comprehensive_speed_analysis_results_with_profiling.csv")
print("Detailed results saved")

# Create example comodulogram comparison
if static_results:
    # Find best accuracy configuration
    best_config = min(static_results, key=lambda x: x['gpac_phase_error'] + x['gpac_amp_error'])
    print(f"\nBest accuracy configuration: {best_config['n_pha_bands']}×{best_config['n_amp_bands']} bands")
    print(f"  Phase error: gPAC={best_config['gpac_phase_error']:.2f} Hz, TensorPAC={best_config['tp_phase_error']:.2f} Hz")
    print(f"  Speedup: {best_config['speedup_comp']:.1f}x")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
if static_results:
    print(f"Average speedup: {avg_speedup:.1f}x")
    print(f"gPAC is faster in {sum(1 for r in static_results if r['speedup_comp'] > 1)}/{len(static_results)} configurations")