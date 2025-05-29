#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_speed_analysis_simple.py

"""
Simple speed analysis comparing static vs trainable filters
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

# Test configurations
configs = [
    # (batch, duration, channels, n_bands, fs)
    (1, 1, 4, 10, 256),
    (1, 2, 8, 30, 512),
    (4, 1, 16, 50, 1024),
    (8, 2, 16, 30, 512),
    (16, 1, 32, 10, 256),
]

# Ground truth
true_phase_freq = 6.0
true_amp_freq = 80.0

results = []

print("Testing gPAC speed with static vs trainable filters")
print("=" * 70)

for batch_size, duration, n_ch, n_bands, fs in configs:
    n_samples = int(duration * fs)
    
    print(f"\nConfig: batch={batch_size}, duration={duration}s, channels={n_ch}, "
          f"bands={n_bands}×{n_bands}, fs={fs}Hz")
    
    # Generate signals
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    batch_signals = []
    for b in range(batch_size):
        signals = []
        for _ in range(n_ch):
            signal = generator.generate_pac_signal(
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=0.8,
                noise_level=0.1
            )
            signals.append(signal)
        batch_signals.append(signals)
    batch_signals = np.array(batch_signals)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = torch.from_numpy(batch_signals).float().to(device)
    
    # Test static filter
    print("  Testing static filter...")
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
    start = time.time()
    output_static = pac_static(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    static_time = time.time() - start
    
    # Test trainable filter
    print("  Testing trainable filter...")
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
    start = time.time()
    output_trainable = pac_trainable(signal_torch)
    if device == 'cuda':
        torch.cuda.synchronize()
    trainable_time = time.time() - start
    
    # Test TensorPAC for comparison
    print("  Testing TensorPAC...")
    pha_edges = np.linspace(2, 20, n_bands + 1)
    amp_edges = np.linspace(30, 150, n_bands + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    pac_tp = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
    signals_tp = batch_signals.reshape(-1, n_samples).T
    
    start = time.time()
    pac_matrix_tp = pac_tp.filterfit(fs, signals_tp, n_jobs=1)
    tp_time = time.time() - start
    
    # Extract PAC values
    pac_static_max = output_static['pac'].max().item()
    pac_trainable_max = output_trainable['pac'].max().item()
    pac_tp_max = pac_matrix_tp.max()
    
    result = {
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
        'pac_static': pac_static_max,
        'pac_trainable': pac_trainable_max,
        'pac_tp': pac_tp_max,
    }
    results.append(result)
    
    print(f"  Times: Static={static_time:.4f}s, Trainable={trainable_time:.4f}s, TensorPAC={tp_time:.4f}s")
    print(f"  Overhead: {result['overhead']:.1f}%")
    print(f"  Speedup: Static={result['speedup_static']:.1f}x, Trainable={result['speedup_trainable']:.1f}x")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Time comparison
ax = axes[0, 0]
x = np.arange(len(results))
width = 0.25
ax.bar(x - width, [r['static_time'] for r in results], width, label='gPAC Static', alpha=0.8)
ax.bar(x, [r['trainable_time'] for r in results], width, label='gPAC Trainable', alpha=0.8)
ax.bar(x + width, [r['tp_time'] for r in results], width, label='TensorPAC', alpha=0.8)
ax.set_xlabel('Configuration')
ax.set_ylabel('Computation Time (s)')
ax.set_title('Computation Time Comparison')
ax.set_xticks(x)
ax.set_xticklabels([f"B{r['batch_size']}_C{r['n_channels']}" for r in results], rotation=45)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# 2. Overhead
ax = axes[0, 1]
ax.bar(x, [r['overhead'] for r in results], alpha=0.8, color='orange')
ax.set_xlabel('Configuration')
ax.set_ylabel('Overhead (%)')
ax.set_title('Trainable Filter Overhead vs Static')
ax.set_xticks(x)
ax.set_xticklabels([f"B{r['batch_size']}_C{r['n_channels']}" for r in results], rotation=45)
ax.grid(True, axis='y', alpha=0.3)

# 3. Speedup
ax = axes[1, 0]
ax.plot(x, [r['speedup_static'] for r in results], 'o-', label='Static', linewidth=2)
ax.plot(x, [r['speedup_trainable'] for r in results], 's-', label='Trainable', linewidth=2)
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Configuration')
ax.set_ylabel('Speedup Factor')
ax.set_title('gPAC Speedup over TensorPAC')
ax.set_xticks(x)
ax.set_xticklabels([f"B{r['batch_size']}_C{r['n_channels']}" for r in results], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. PAC values
ax = axes[1, 1]
ax.scatter([r['pac_tp'] for r in results], [r['pac_static'] for r in results], 
           label='Static', alpha=0.8, s=100)
ax.scatter([r['pac_tp'] for r in results], [r['pac_trainable'] for r in results], 
           label='Trainable', alpha=0.8, s=100)
max_val = max(max(r['pac_tp'] for r in results), 
              max(r['pac_static'] for r in results),
              max(r['pac_trainable'] for r in results))
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
ax.set_xlabel('TensorPAC Peak Value')
ax.set_ylabel('gPAC Peak Value')
ax.set_title('PAC Value Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
mngs.io.save(plt.gcf(), "speed_analysis_simple.png")
print(f"\nVisualization saved")

# Summary
print("\n" + "="*70)
print("SUMMARY:")
print(f"Average overhead (trainable vs static): {np.mean([r['overhead'] for r in results]):.1f}%")
print(f"Average speedup (static): {np.mean([r['speedup_static'] for r in results]):.1f}x")
print(f"Average speedup (trainable): {np.mean([r['speedup_trainable'] for r in results]):.1f}x")
print(f"PAC value ratio (gPAC/TensorPAC):")
print(f"  Static: {np.mean([r['pac_static']/r['pac_tp'] for r in results]):.3f}")
print(f"  Trainable: {np.mean([r['pac_trainable']/r['pac_tp'] for r in results]):.3f}")