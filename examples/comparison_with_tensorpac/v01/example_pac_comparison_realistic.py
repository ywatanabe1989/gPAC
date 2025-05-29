#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_pac_comparison_realistic.py

"""
Test gPAC vs TensorPAC with realistic EEG-like parameters
(n_batches, n_chs, n_segments, seq_len) = (1, 16, 1, 400Hz*60sec)
"""

import numpy as np
import torch
import time
from tensorpac import Pac as TensorPAC_Pac
from gpac import SyntheticDataGenerator, PAC as gPAC_PAC
import mngs

# Parameters matching user's request
n_channels = 16
duration = 60  # seconds
sample_rate = 400  # Hz
n_samples = duration * sample_rate  # 24000 samples

print(f"Testing with realistic EEG parameters:")
print(f"  Channels: {n_channels}")
print(f"  Duration: {duration} seconds")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Total samples per channel: {n_samples}")
print("=" * 60)

# Generate multi-channel signal
generator = SyntheticDataGenerator(fs=sample_rate, duration_sec=duration)
signals = []
for ch in range(n_channels):
    signal = generator.generate_pac_signal(
        phase_freq=6.0,
        amp_freq=80.0,
        coupling_strength=0.5,
        noise_level=0.1
    )
    signals.append(signal)
signals = np.array(signals)  # (n_channels, n_samples)

print(f"\nSignal shape: {signals.shape}")

# Split channels for GPU memory
n_channels_per_batch = 8
print(f"\nProcessing in batches of {n_channels_per_batch} channels due to GPU memory constraints...")

# Test gPAC
print("\nTesting gPAC (GPU)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Process in two batches
batch1 = signals[:n_channels_per_batch]  # First 8 channels
batch2 = signals[n_channels_per_batch:n_channels]  # Next 8 channels

# Convert to torch tensor with proper shape
signal_torch1 = torch.from_numpy(batch1).float().unsqueeze(0).to(device)  # (1, 8, 24000)
signal_torch2 = torch.from_numpy(batch2).float().unsqueeze(0).to(device)  # (1, 8, 24000)

# Initialize gPAC with single frequency bands as requested
init_start = time.time()
pac_gpac = gPAC_PAC(
    seq_len=n_samples,
    fs=sample_rate,
    pha_start_hz=4,
    pha_end_hz=8,
    pha_n_bands=1,  # Single band
    amp_start_hz=30,
    amp_end_hz=60,
    amp_n_bands=1,  # Single band
    trainable=False
).to(device)
init_time = time.time() - init_start

# Warm-up on first batch
_ = pac_gpac(signal_torch1)
if device == 'cuda':
    torch.cuda.synchronize()

# Time computation for both batches
start = time.time()
output_gpac1 = pac_gpac(signal_torch1)
output_gpac2 = pac_gpac(signal_torch2)
if device == 'cuda':
    torch.cuda.synchronize()
gpac_time = time.time() - start

# Extract PAC values
pac_values1 = output_gpac1['pac'].squeeze().cpu().numpy()  # (8,) for 8 channels
pac_values2 = output_gpac2['pac'].squeeze().cpu().numpy()  # (8,) for 8 channels
pac_values_gpac = np.concatenate([pac_values1, pac_values2])  # (16,) for all channels

print(f"  Initialization: {init_time:.4f}s")
print(f"  Computation (2 batches): {gpac_time:.4f}s")
print(f"  PAC values shape: {pac_values_gpac.shape}")
print(f"  PAC values: {pac_values_gpac}")
print(f"  Mean PAC value: {pac_values_gpac.mean():.4f}")

# Test TensorPAC
print("\nTesting TensorPAC (CPU)...")

# TensorPAC expects (n_channels, n_times) format for filterfit
signals_tensorpac = signals  # (16, 24000)

# Initialize TensorPAC with single frequency bands
init_start = time.time()
pac_tensorpac = TensorPAC_Pac(f_pha=(4, 8), f_amp=(30, 60), verbose=False)
init_time_tp = time.time() - init_start

# Time computation
start = time.time()
pac_values_tensorpac = pac_tensorpac.filterfit(sample_rate, signals_tensorpac, n_jobs=1)
tensorpac_time = time.time() - start

print(f"  Initialization: {init_time_tp:.4f}s")
print(f"  Computation: {tensorpac_time:.4f}s")
print(f"  PAC values shape: {pac_values_tensorpac.shape}")
print(f"  PAC values: {pac_values_tensorpac}")
print(f"  Mean PAC value: {pac_values_tensorpac.mean():.4f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"Speedup (computation): {tensorpac_time/gpac_time:.2f}x")
print(f"Total time gPAC: {init_time + gpac_time:.4f}s")
print(f"Total time TensorPAC: {init_time_tp + tensorpac_time:.4f}s")
print(f"Total speedup: {(init_time_tp + tensorpac_time)/(init_time + gpac_time):.2f}x")

# Save results
results = {
    'parameters': {
        'n_channels': n_channels,
        'duration': duration,
        'sample_rate': sample_rate,
        'n_samples': n_samples
    },
    'gpac': {
        'init_time': init_time,
        'compute_time': gpac_time,
        'total_time': init_time + gpac_time,
        'mean_pac': pac_values_gpac.mean(),
        'pac_values': pac_values_gpac
    },
    'tensorpac': {
        'init_time': init_time_tp,
        'compute_time': tensorpac_time,
        'total_time': init_time_tp + tensorpac_time,
        'mean_pac': pac_values_tensorpac.mean(),
        'pac_values': pac_values_tensorpac
    },
    'speedup': {
        'computation': tensorpac_time/gpac_time,
        'total': (init_time_tp + tensorpac_time)/(init_time + gpac_time)
    }
}

mngs.io.save(results, "realistic_eeg_comparison_results.pkl")
print(f"\nResults saved to realistic_eeg_comparison_results.pkl")

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. PAC values comparison
ax = axes[0, 0]
x = np.arange(n_channels)
width = 0.35
ax.bar(x - width/2, pac_values_gpac, width, label='gPAC', alpha=0.8)
ax.bar(x + width/2, pac_values_tensorpac.squeeze(), width, label='TensorPAC', alpha=0.8)
ax.set_xlabel('Channel')
ax.set_ylabel('PAC Value')
ax.set_title('PAC Values by Channel')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Time comparison
ax = axes[0, 1]
categories = ['Initialization', 'Computation', 'Total']
gpac_times = [init_time, gpac_time, init_time + gpac_time]
tensorpac_times = [init_time_tp, tensorpac_time, init_time_tp + tensorpac_time]
x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, gpac_times, width, label='gPAC', alpha=0.8)
ax.bar(x + width/2, tensorpac_times, width, label='TensorPAC', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Time (seconds)')
ax.set_title('Time Comparison')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# 3. Speedup visualization
ax = axes[1, 0]
speedups = [tensorpac_time/gpac_time, (init_time_tp + tensorpac_time)/(init_time + gpac_time)]
bars = ax.bar(['Computation Only', 'Total'], speedups, alpha=0.8, color=['green', 'blue'])
ax.set_ylabel('Speedup Factor')
ax.set_title(f'gPAC Speedup over TensorPAC')
ax.grid(True, axis='y', alpha=0.3)
# Add value labels on bars
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.1f}x', ha='center', va='bottom')

# 4. Signal example
ax = axes[1, 1]
# Show first 2 seconds of first channel
t = np.arange(2 * sample_rate) / sample_rate
ax.plot(t, signals[0, :2*sample_rate], 'b-', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title(f'Example Signal (Channel 1, first 2s)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
mngs.io.save(plt.gcf(), "realistic_eeg_comparison.png")
print(f"\nVisualization saved")