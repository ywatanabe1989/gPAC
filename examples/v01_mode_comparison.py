#!/usr/bin/env python3
"""
V01 Mode Comparison Example

This example demonstrates the v01_mode option which uses depthwise convolution
for potentially better TensorPAC compatibility.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gpac import calculate_pac
import time

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Generate test signal
fs = 512  # Higher sampling rate to avoid frequency warnings
duration = 3
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Create signal with PAC
phase_freq = 8  # Hz
amp_freq = 80   # Hz
modulation_depth = 0.6

phase_signal = np.sin(2 * np.pi * phase_freq * t)
amp_envelope = 1 + modulation_depth * phase_signal
amp_signal = amp_envelope * np.sin(2 * np.pi * amp_freq * t)
signal = phase_signal + 0.4 * amp_signal + 0.1 * np.random.randn(len(t))

# Prepare for gPAC
signal_4d = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

print("Comparing v01_mode with standard mode...")
print("=" * 50)

# 1. Standard mode
print("\n1. Standard mode (scipy-compatible filtfilt)")
start_time = time.time()
pac_standard, pha_freqs, amp_freqs = calculate_pac(
    signal_4d,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=15,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=15,
    v01_mode=False  # Standard mode
)
standard_time = time.time() - start_time
pac_standard_np = pac_standard[0, 0].cpu().numpy()

print(f"  Time: {standard_time:.3f}s")
print(f"  Max PAC: {pac_standard_np.max():.4f}")
print(f"  Mean PAC: {pac_standard_np.mean():.4f}")

# 2. V01 mode
print("\n2. V01 mode (depthwise convolution)")
start_time = time.time()
pac_v01, _, _ = calculate_pac(
    signal_4d,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=15,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=15,
    v01_mode=True  # V01 mode
)
v01_time = time.time() - start_time
pac_v01_np = pac_v01[0, 0].cpu().numpy()

print(f"  Time: {v01_time:.3f}s")
print(f"  Max PAC: {pac_v01_np.max():.4f}")
print(f"  Mean PAC: {pac_v01_np.mean():.4f}")

# Compare the two
print(f"\n3. Comparison")
print(f"  Speed ratio (v01/standard): {v01_time/standard_time:.2f}x")
print(f"  Max value ratio: {pac_v01_np.max()/pac_standard_np.max():.3f}")
print(f"  Correlation: {np.corrcoef(pac_standard_np.flatten(), pac_v01_np.flatten())[0,1]:.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Standard mode
ax = axes[0]
im = ax.imshow(pac_standard_np, aspect='auto', origin='lower',
               extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
               cmap='hot')
ax.set_title(f'Standard Mode\n(max={pac_standard_np.max():.4f})')
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('Phase Frequency (Hz)')
plt.colorbar(im, ax=ax)

# V01 mode
ax = axes[1]
im = ax.imshow(pac_v01_np, aspect='auto', origin='lower',
               extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
               cmap='hot')
ax.set_title(f'V01 Mode\n(max={pac_v01_np.max():.4f})')
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('Phase Frequency (Hz)')
plt.colorbar(im, ax=ax)

# Difference
ax = axes[2]
diff = pac_v01_np - pac_standard_np
im = ax.imshow(diff, aspect='auto', origin='lower',
               extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
               cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
ax.set_title(f'Difference (V01 - Standard)\n(max abs diff={np.abs(diff).max():.4f})')
ax.set_xlabel('Amplitude Frequency (Hz)')
ax.set_ylabel('Phase Frequency (Hz)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# Key findings
print("\n" + "=" * 50)
print("KEY FINDINGS:")
print("- V01 mode uses depthwise convolution (groups=n_filters)")
print("- May provide better TensorPAC compatibility")
print("- Generally produces very similar results to standard mode")
print("- Performance may vary depending on hardware and signal size")
print("\nRecommendation: Use v01_mode=True when comparing with TensorPAC")