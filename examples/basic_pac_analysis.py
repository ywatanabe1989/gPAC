#!/usr/bin/env python3
"""
Basic Phase-Amplitude Coupling (PAC) Analysis Example

This example demonstrates how to:
1. Generate synthetic signals with known PAC
2. Calculate PAC using gPAC
3. Visualize the results
4. Compare with statistical surrogate testing
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gpac import calculate_pac

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
fs = 1000  # Sampling frequency (Hz)
duration = 5  # Signal duration (seconds)
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)

# 1. Generate synthetic signal with PAC
print("Generating synthetic signal with PAC...")

# Phase component (theta, 6 Hz)
phase_freq = 6  # Hz
phase_signal = np.sin(2 * np.pi * phase_freq * t)

# Amplitude component (gamma, 80 Hz) modulated by phase
amp_freq = 80  # Hz
modulation_depth = 0.7  # How strongly phase modulates amplitude (0-1)

# Create amplitude modulation envelope
amp_envelope = 1 + modulation_depth * np.sin(2 * np.pi * phase_freq * t)

# Generate modulated high-frequency signal
amp_signal = amp_envelope * np.sin(2 * np.pi * amp_freq * t)

# Combine signals with some noise
noise_level = 0.2
noise = noise_level * np.random.randn(n_samples)
signal = phase_signal + 0.3 * amp_signal + noise

# 2. Prepare signal for gPAC
# gPAC expects 4D tensor: (batch, channels, segments, time)
signal_tensor = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

print(f"Signal shape: {signal_tensor.shape}")
print(f"Signal duration: {duration} seconds")
print(f"Sampling rate: {fs} Hz")

# 3. Calculate PAC without statistical testing
print("\nCalculating PAC...")
pac_values, pha_freqs, amp_freqs = calculate_pac(
    signal_tensor,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=20,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=20,
    n_perm=None  # No permutation testing
)

# Convert to numpy for plotting
pac_matrix = pac_values[0, 0].cpu().numpy()

# 4. Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot original signal (first second)
ax = axes[0, 0]
plot_duration = 1  # seconds
plot_samples = int(fs * plot_duration)
ax.plot(t[:plot_samples], signal[:plot_samples], 'b-', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Original Signal (first second)')
ax.set_xlim(0, plot_duration)

# Plot phase and amplitude components
ax = axes[0, 1]
ax.plot(t[:plot_samples], phase_signal[:plot_samples], 'g-', label=f'Phase ({phase_freq} Hz)')
ax.plot(t[:plot_samples], 0.3 * amp_signal[:plot_samples], 'r-', alpha=0.7, label=f'Amplitude ({amp_freq} Hz)')
ax.plot(t[:plot_samples], 0.3 * amp_envelope[:plot_samples], 'k--', label='Modulation envelope')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Signal Components')
ax.set_xlim(0, plot_duration)
ax.legend()

# Plot PAC matrix
ax = axes[1, 0]
im = ax.imshow(pac_matrix, aspect='auto', origin='lower',
               extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
               cmap='hot')
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('Phase Frequency (Hz)')
ax.set_title('Phase-Amplitude Coupling')
plt.colorbar(im, ax=ax, label='PAC Strength')

# Add markers for true coupling
ax.plot(amp_freq, phase_freq, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)
ax.text(amp_freq + 2, phase_freq + 0.5, 'True\ncoupling', color='white', fontsize=9)

# 5. Calculate PAC with statistical testing
print("\nCalculating PAC with permutation testing...")
pac_zscore, pha_freqs, amp_freqs = calculate_pac(
    signal_tensor,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=20,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=20,
    n_perm=200  # 200 permutations for z-score calculation
)

# Plot z-scored PAC
ax = axes[1, 1]
pac_z_matrix = pac_zscore[0, 0].cpu().numpy()
im = ax.imshow(pac_z_matrix, aspect='auto', origin='lower',
               extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
               cmap='hot', vmin=0, vmax=5)
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('Phase Frequency (Hz)')
ax.set_title('PAC Z-scores (n_perm=200)')
plt.colorbar(im, ax=ax, label='Z-score')

# Mark significant regions (z > 2 ~ p < 0.05)
significant = pac_z_matrix > 2
ax.contour(amp_freqs, pha_freqs, significant, levels=[0.5], colors='cyan', linewidths=2)
ax.plot(amp_freq, phase_freq, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)

plt.tight_layout()
plt.show()

# 6. Report findings
print("\n=== PAC Analysis Results ===")
print(f"Maximum PAC value: {pac_matrix.max():.4f}")
max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
print(f"Peak coupling at: Phase {pha_freqs[max_idx[0]]:.1f} Hz, Amplitude {amp_freqs[max_idx[1]]:.1f} Hz")
print(f"Expected coupling: Phase {phase_freq} Hz, Amplitude {amp_freq} Hz")

print(f"\nMaximum z-score: {pac_z_matrix.max():.2f}")
significant_pairs = np.sum(pac_z_matrix > 2)
total_pairs = pac_z_matrix.size
print(f"Significant frequency pairs (z > 2): {significant_pairs}/{total_pairs} ({100*significant_pairs/total_pairs:.1f}%)")

# 7. Save results
print("\nSaving results...")
np.savez('pac_results.npz',
         pac_values=pac_matrix,
         pac_zscores=pac_z_matrix,
         pha_freqs=pha_freqs,
         amp_freqs=amp_freqs,
         signal=signal,
         fs=fs)
print("Results saved to pac_results.npz")

print("\nExample completed successfully!")