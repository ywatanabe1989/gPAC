#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: example_pac_comodulogram_comparison.py

"""
Compare PAC comodulograms between gPAC and TensorPAC with ground truth
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorpac import Pac as TensorPAC_Pac
from gpac import SyntheticDataGenerator, PAC as gPAC_PAC
import mngs

# Parameters
duration = 10  # seconds
sample_rate = 1000  # Hz
n_samples = duration * sample_rate

# Ground truth PAC parameters
true_phase_freq = 6.0  # Hz
true_amp_freq = 80.0  # Hz
coupling_strength = 0.8

print("Generating synthetic PAC signal with known ground truth...")
print(f"  True phase frequency: {true_phase_freq} Hz")
print(f"  True amplitude frequency: {true_amp_freq} Hz")
print(f"  Coupling strength: {coupling_strength}")

# Generate signal
generator = SyntheticDataGenerator(fs=sample_rate, duration_sec=duration)
signal = generator.generate_pac_signal(
    phase_freq=true_phase_freq,
    amp_freq=true_amp_freq,
    coupling_strength=coupling_strength,
    noise_level=0.1
)

# Frequency parameters for PAC analysis
pha_range = (2, 20)
amp_range = (30, 150)
n_pha_bands = 18
n_amp_bands = 24

print(f"\nPAC analysis parameters:")
print(f"  Phase range: {pha_range} Hz, {n_pha_bands} bands")
print(f"  Amplitude range: {amp_range} Hz, {n_amp_bands} bands")

# Compute gPAC
print("\nComputing gPAC...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
signal_torch = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0).to(device)

pac_gpac = gPAC_PAC(
    seq_len=n_samples,
    fs=sample_rate,
    pha_start_hz=pha_range[0],
    pha_end_hz=pha_range[1],
    pha_n_bands=n_pha_bands,
    amp_start_hz=amp_range[0],
    amp_end_hz=amp_range[1],
    amp_n_bands=n_amp_bands,
    trainable=False
).to(device)

output_gpac = pac_gpac(signal_torch)
pac_matrix_gpac = output_gpac['pac'].squeeze().cpu().numpy()
pha_freqs_gpac = output_gpac['phase_frequencies'].cpu().numpy()
amp_freqs_gpac = output_gpac['amplitude_frequencies'].cpu().numpy()

print(f"  PAC matrix shape: {pac_matrix_gpac.shape}")
print(f"  Max PAC value: {pac_matrix_gpac.max():.4f}")
print(f"  PAC at ground truth location: {pac_matrix_gpac[np.argmin(np.abs(pha_freqs_gpac - true_phase_freq)), np.argmin(np.abs(amp_freqs_gpac - true_amp_freq))]:.4f}")

# Compute TensorPAC
print("\nComputing TensorPAC...")
# Create explicit frequency bands for TensorPAC
pha_edges = np.linspace(pha_range[0], pha_range[1], n_pha_bands + 1)
amp_edges = np.linspace(amp_range[0], amp_range[1], n_amp_bands + 1)
pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]

pac_tensorpac = TensorPAC_Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, verbose=False)
pac_matrix_tensorpac = pac_tensorpac.filterfit(sample_rate, signal, n_jobs=1).squeeze().T  # Transpose to match gPAC

# Get center frequencies for TensorPAC
pha_freqs_tensorpac = (pha_bands[:, 0] + pha_bands[:, 1]) / 2
amp_freqs_tensorpac = (amp_bands[:, 0] + amp_bands[:, 1]) / 2

print(f"  PAC matrix shape: {pac_matrix_tensorpac.shape}")
print(f"  Max PAC value: {pac_matrix_tensorpac.max():.4f}")
print(f"  PAC at ground truth location: {pac_matrix_tensorpac[np.argmin(np.abs(pha_freqs_tensorpac - true_phase_freq)), np.argmin(np.abs(amp_freqs_tensorpac - true_amp_freq))]:.4f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Common colormap settings
vmin_gpac = 0
vmax_gpac = pac_matrix_gpac.max()
vmin_tp = pac_matrix_tensorpac.min()
vmax_tp = pac_matrix_tensorpac.max()

# 1. gPAC comodulogram
ax = axes[0, 0]
im1 = ax.imshow(pac_matrix_gpac.T, aspect='auto', origin='lower',
                extent=[pha_range[0], pha_range[1], amp_range[0], amp_range[1]],
                cmap='hot', vmin=vmin_gpac, vmax=vmax_gpac)
ax.axvline(true_phase_freq, color='cyan', linestyle='--', linewidth=2, label='True phase')
ax.axhline(true_amp_freq, color='cyan', linestyle='--', linewidth=2, label='True amp')
ax.set_xlabel('Phase Frequency (Hz)')
ax.set_ylabel('Amplitude Frequency (Hz)')
ax.set_title(f'gPAC Comodulogram (max={pac_matrix_gpac.max():.4f})')
plt.colorbar(im1, ax=ax)

# 2. TensorPAC comodulogram
ax = axes[0, 1]
im2 = ax.imshow(pac_matrix_tensorpac.T, aspect='auto', origin='lower',
                extent=[pha_range[0], pha_range[1], amp_range[0], amp_range[1]],
                cmap='RdBu_r', vmin=vmin_tp, vmax=vmax_tp)
ax.axvline(true_phase_freq, color='black', linestyle='--', linewidth=2, label='True phase')
ax.axhline(true_amp_freq, color='black', linestyle='--', linewidth=2, label='True amp')
ax.set_xlabel('Phase Frequency (Hz)')
ax.set_ylabel('Amplitude Frequency (Hz)')
ax.set_title(f'TensorPAC Comodulogram (max={pac_matrix_tensorpac.max():.4f})')
plt.colorbar(im2, ax=ax)

# 3. Direct comparison at ground truth
ax = axes[1, 0]
# Extract cross-sections at true frequencies
phase_idx_gpac = np.argmin(np.abs(pha_freqs_gpac - true_phase_freq))
phase_idx_tp = np.argmin(np.abs(pha_freqs_tensorpac - true_phase_freq))
amp_idx_gpac = np.argmin(np.abs(amp_freqs_gpac - true_amp_freq))
amp_idx_tp = np.argmin(np.abs(amp_freqs_tensorpac - true_amp_freq))

# Plot amplitude frequency profile at true phase frequency
ax.plot(amp_freqs_gpac, pac_matrix_gpac[phase_idx_gpac, :], 'r-', linewidth=2, label='gPAC')
ax.plot(amp_freqs_tensorpac, pac_matrix_tensorpac[phase_idx_tp, :], 'b-', linewidth=2, label='TensorPAC')
ax.axvline(true_amp_freq, color='gray', linestyle='--', linewidth=1, label='True amp freq')
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('PAC Value')
ax.set_title(f'PAC Profile at Phase={true_phase_freq} Hz')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Peak detection comparison
ax = axes[1, 1]
# Find peaks
gpac_peak_idx = np.unravel_index(pac_matrix_gpac.argmax(), pac_matrix_gpac.shape)
gpac_peak_phase = pha_freqs_gpac[gpac_peak_idx[0]]
gpac_peak_amp = amp_freqs_gpac[gpac_peak_idx[1]]

tp_peak_idx = np.unravel_index(pac_matrix_tensorpac.argmax(), pac_matrix_tensorpac.shape)
tp_peak_phase = pha_freqs_tensorpac[tp_peak_idx[0]]
tp_peak_amp = amp_freqs_tensorpac[tp_peak_idx[1]]

# Create comparison table
comparison_data = [
    ['', 'gPAC', 'TensorPAC', 'Ground Truth'],
    ['Peak Phase (Hz)', f'{gpac_peak_phase:.1f}', f'{tp_peak_phase:.1f}', f'{true_phase_freq:.1f}'],
    ['Peak Amp (Hz)', f'{gpac_peak_amp:.1f}', f'{tp_peak_amp:.1f}', f'{true_amp_freq:.1f}'],
    ['Peak PAC Value', f'{pac_matrix_gpac.max():.4f}', f'{pac_matrix_tensorpac.max():.4f}', '-'],
    ['Value Range', f'[{pac_matrix_gpac.min():.4f}, {pac_matrix_gpac.max():.4f}]', 
     f'[{pac_matrix_tensorpac.min():.4f}, {pac_matrix_tensorpac.max():.4f}]', '-']
]

# Plot table
ax.axis('off')
table = ax.table(cellText=comparison_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax.set_title('Peak Detection Comparison', pad=20)

plt.tight_layout()
mngs.io.save(plt.gcf(), "pac_comodulogram_comparison.png")
print(f"\nVisualization saved")

# Print summary
print("\n" + "="*60)
print("SUMMARY:")
print(f"Ground truth: Phase={true_phase_freq} Hz, Amplitude={true_amp_freq} Hz")
print(f"gPAC peak: Phase={gpac_peak_phase:.1f} Hz, Amplitude={gpac_peak_amp:.1f} Hz")
print(f"TensorPAC peak: Phase={tp_peak_phase:.1f} Hz, Amplitude={tp_peak_amp:.1f} Hz")
print(f"Phase frequency error - gPAC: {abs(gpac_peak_phase - true_phase_freq):.1f} Hz, TensorPAC: {abs(tp_peak_phase - true_phase_freq):.1f} Hz")
print(f"Amplitude frequency error - gPAC: {abs(gpac_peak_amp - true_amp_freq):.1f} Hz, TensorPAC: {abs(tp_peak_amp - true_amp_freq):.1f} Hz")