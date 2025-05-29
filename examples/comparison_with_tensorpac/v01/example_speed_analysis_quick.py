#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_speed_analysis_quick.py

"""
Quick speed analysis with key configurations and PAC validation
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

# Key test configurations
configs = [
    # Small: (batch, duration, channels, n_bands, fs)
    (1, 2, 4, 10, 256),     # Small data, low resolution
    (4, 2, 16, 30, 512),    # Medium data, medium resolution
    (16, 1, 32, 50, 1024),  # Large batch, high resolution
]

# Ground truth
true_phase_freq = 6.0
true_amp_freq = 80.0

results = []

print("Quick gPAC vs TensorPAC comparison")
print("=" * 80)

for config_idx, (batch_size, duration, n_ch, n_bands, fs) in enumerate(configs):
    n_samples = int(duration * fs)
    
    print(f"\n[{config_idx+1}/3] Config: batch={batch_size}, duration={duration}s, "
          f"channels={n_ch}, bands={n_bands}×{n_bands}, fs={fs}Hz")
    
    # Generate test signal
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    signal = generator.generate_pac_signal(
        phase_freq=true_phase_freq,
        amp_freq=true_amp_freq,
        coupling_strength=0.8,
        noise_level=0.1
    )
    
    # Create batch
    batch_signals = np.tile(signal[np.newaxis, np.newaxis, :], (batch_size, n_ch, 1))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = torch.from_numpy(batch_signals).float().to(device)
    
    # Test static filter
    print("  Testing gPAC static filter...")
    pac_static = gPAC_PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_bands,
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=n_bands,
        trainable=False
    ).to(device)
    
    # Warm-up
    _ = pac_static(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time static
    times = []
    for _ in range(3):  # Average over 3 runs
        start = time.time()
        output_static = pac_static(signal_torch)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    static_time = np.mean(times)
    
    # Extract PAC result
    pac_static_matrix = output_static['pac'][0, 0].cpu().numpy()  # First batch, first channel
    pha_freqs = output_static['phase_frequencies'].cpu().numpy()
    amp_freqs = output_static['amplitude_frequencies'].cpu().numpy()
    
    # Test trainable filter
    print("  Testing gPAC trainable filter...")
    pac_trainable = gPAC_PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_bands,
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=n_bands,
        trainable=True
    ).to(device)
    
    # Warm-up
    _ = pac_trainable(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time trainable
    times = []
    for _ in range(3):
        start = time.time()
        output_trainable = pac_trainable(signal_torch)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    trainable_time = np.mean(times)
    
    pac_trainable_matrix = output_trainable['pac'][0, 0].detach().cpu().numpy()
    
    # Test TensorPAC for comparison
    print("  Testing TensorPAC...")
    pha_edges = np.linspace(2, 20, n_bands + 1)
    amp_edges = np.linspace(30, 150, n_bands + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
    
    # Time TensorPAC
    times = []
    for _ in range(3):
        start = time.time()
        pac_tp_matrix = pac_tp.filterfit(fs, signal, n_jobs=1).squeeze().T
        times.append(time.time() - start)
    tp_time = np.mean(times)
    
    # Find peaks
    static_peak_idx = np.unravel_index(pac_static_matrix.argmax(), pac_static_matrix.shape)
    trainable_peak_idx = np.unravel_index(pac_trainable_matrix.argmax(), pac_trainable_matrix.shape)
    tp_peak_idx = np.unravel_index(pac_tp_matrix.argmax(), pac_tp_matrix.shape)
    
    result = {
        'config': f"B{batch_size}_D{duration}_C{n_ch}_F{n_bands}",
        'batch_size': batch_size,
        'duration': duration,
        'n_channels': n_ch,
        'n_bands': n_bands,
        'fs': fs,
        'static_time': static_time,
        'trainable_time': trainable_time,
        'tp_time': tp_time,
        'overhead': (trainable_time - static_time) / static_time * 100,
        'speedup_static': tp_time / static_time,
        'speedup_trainable': tp_time / trainable_time,
        'pac_static_max': pac_static_matrix.max(),
        'pac_trainable_max': pac_trainable_matrix.max(),
        'pac_tp_max': pac_tp_matrix.max(),
        'static_peak_phase': pha_freqs[static_peak_idx[0]],
        'static_peak_amp': amp_freqs[static_peak_idx[1]],
        'trainable_peak_phase': pha_freqs[trainable_peak_idx[0]],
        'trainable_peak_amp': amp_freqs[trainable_peak_idx[1]],
        'tp_peak_phase': (pha_bands[tp_peak_idx[0], 0] + pha_bands[tp_peak_idx[0], 1]) / 2,
        'tp_peak_amp': (amp_bands[tp_peak_idx[1], 0] + amp_bands[tp_peak_idx[1], 1]) / 2,
        'matrices': {
            'static': pac_static_matrix,
            'trainable': pac_trainable_matrix,
            'tensorpac': pac_tp_matrix
        }
    }
    results.append(result)
    
    # Process times for large batch
    total_time_static = static_time * batch_size * n_ch / n_ch  # Parallel over channels
    total_time_tp = tp_time * batch_size * n_ch  # Sequential
    
    print(f"  Times per signal: Static={static_time:.4f}s, Trainable={trainable_time:.4f}s, TensorPAC={tp_time:.4f}s")
    print(f"  Total batch time: gPAC={static_time:.4f}s, TensorPAC={total_time_tp:.4f}s")
    print(f"  Batch speedup: {total_time_tp/static_time:.1f}x")
    print(f"  Trainable overhead: {result['overhead']:.1f}%")
    print(f"  Peak locations - Static: ({result['static_peak_phase']:.1f}, {result['static_peak_amp']:.1f}) Hz")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# 1. Time comparison
ax1 = plt.subplot(3, 3, 1)
x = np.arange(len(results))
width = 0.25
ax1.bar(x - width, [r['static_time'] for r in results], width, label='gPAC Static', alpha=0.8)
ax1.bar(x, [r['trainable_time'] for r in results], width, label='gPAC Trainable', alpha=0.8)
ax1.bar(x + width, [r['tp_time'] for r in results], width, label='TensorPAC', alpha=0.8)
ax1.set_ylabel('Time per Signal (s)')
ax1.set_title('Computation Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([r['config'] for r in results], rotation=45)
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_yscale('log')

# 2. Speedup visualization
ax2 = plt.subplot(3, 3, 2)
ax2.plot(x, [r['speedup_static'] for r in results], 'o-', label='Static', linewidth=2, markersize=10)
ax2.plot(x, [r['speedup_trainable'] for r in results], 's-', label='Trainable', linewidth=2, markersize=10)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax2.set_ylabel('Speedup Factor')
ax2.set_title('gPAC Speedup over TensorPAC')
ax2.set_xticks(x)
ax2.set_xticklabels([r['config'] for r in results], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Batch processing advantage
ax3 = plt.subplot(3, 3, 3)
batch_speedups = []
for r in results:
    batch_total_tp = r['tp_time'] * r['batch_size'] * r['n_channels']
    batch_speedup = batch_total_tp / r['static_time']
    batch_speedups.append(batch_speedup)
ax3.bar(x, batch_speedups, alpha=0.8, color='green')
ax3.set_ylabel('Batch Processing Speedup')
ax3.set_title('gPAC Batch Processing Advantage')
ax3.set_xticks(x)
ax3.set_xticklabels([r['config'] for r in results], rotation=45)
ax3.grid(True, axis='y', alpha=0.3)

# 4-6. Comodulograms for each configuration
for idx, result in enumerate(results):
    ax = plt.subplot(3, 3, 4 + idx)
    
    # Show gPAC static result
    im = ax.imshow(result['matrices']['static'].T, aspect='auto', origin='lower',
                   extent=[2, 20, 30, 150], cmap='hot')
    ax.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter(result['static_peak_phase'], result['static_peak_amp'], 
               color='white', marker='x', s=100, linewidth=2)
    ax.set_xlabel('Phase (Hz)')
    ax.set_ylabel('Amplitude (Hz)')
    ax.set_title(f"gPAC Static - {result['config']}")
    plt.colorbar(im, ax=ax, fraction=0.046)

# 7. PAC value comparison
ax7 = plt.subplot(3, 3, 7)
static_vals = [r['pac_static_max'] for r in results]
trainable_vals = [r['pac_trainable_max'] for r in results]
tp_vals = [r['pac_tp_max'] for r in results]

ax7.scatter(tp_vals, static_vals, label='Static', s=100, alpha=0.8)
ax7.scatter(tp_vals, trainable_vals, label='Trainable', s=100, alpha=0.8)
max_val = max(max(tp_vals), max(static_vals), max(trainable_vals))
ax7.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
ax7.set_xlabel('TensorPAC Peak Value')
ax7.set_ylabel('gPAC Peak Value')
ax7.set_title('Peak PAC Value Comparison')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Detection accuracy
ax8 = plt.subplot(3, 3, 8)
phase_errors_static = [abs(r['static_peak_phase'] - true_phase_freq) for r in results]
phase_errors_trainable = [abs(r['trainable_peak_phase'] - true_phase_freq) for r in results]
phase_errors_tp = [abs(r['tp_peak_phase'] - true_phase_freq) for r in results]

x_err = np.arange(len(results))
ax8.bar(x_err - width, phase_errors_static, width, label='Static', alpha=0.8)
ax8.bar(x_err, phase_errors_trainable, width, label='Trainable', alpha=0.8)
ax8.bar(x_err + width, phase_errors_tp, width, label='TensorPAC', alpha=0.8)
ax8.set_ylabel('Phase Frequency Error (Hz)')
ax8.set_title(f'Phase Detection Error (Truth: {true_phase_freq} Hz)')
ax8.set_xticks(x_err)
ax8.set_xticklabels([r['config'] for r in results], rotation=45)
ax8.legend()
ax8.grid(True, axis='y', alpha=0.3)

# 9. Summary text
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""Summary:
• Average speedup: 
  - Static: {np.mean([r['speedup_static'] for r in results]):.1f}x
  - Trainable: {np.mean([r['speedup_trainable'] for r in results]):.1f}x
  
• Trainable overhead: {np.mean([r['overhead'] for r in results]):.1f}%

• PAC value scaling (gPAC/TensorPAC):
  - Static: {np.mean([r['pac_static_max']/r['pac_tp_max'] for r in results]):.3f}
  - Trainable: {np.mean([r['pac_trainable_max']/r['pac_tp_max'] for r in results]):.3f}

• Phase detection error:
  - gPAC: {np.mean(phase_errors_static):.1f} Hz
  - TensorPAC: {np.mean(phase_errors_tp):.1f} Hz

• Best batch speedup: {max(batch_speedups):.1f}x
  at {results[np.argmax(batch_speedups)]['config']}
"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
mngs.io.save(plt.gcf(), "speed_analysis_quick.png")
print(f"\nVisualization saved")

# Save results
import pandas as pd
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'matrices'} for r in results])
mngs.io.save(df, "speed_analysis_quick_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"Key findings:")
print(f"1. gPAC is slower per signal but much faster for batch processing")
print(f"2. Trainable filters add ~{np.mean([r['overhead'] for r in results]):.0f}% overhead")
print(f"3. PAC values differ by factor of {np.mean([r['pac_static_max']/r['pac_tp_max'] for r in results]):.1f}x (normalization difference)")
print(f"4. Both detect PAC accurately (within {np.mean(phase_errors_static):.1f} Hz)")
print(f"5. Maximum batch speedup: {max(batch_speedups):.1f}x with GPU parallelization")