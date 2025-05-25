#!/usr/bin/env python3
"""
Demo showing how different parameters affect PAC resolution and values.
"""

import torch
import gpac
import numpy as np
import matplotlib.pyplot as plt

# Create test signal with known coupling
def create_signal(fs=512, duration=5.0, snr_db=10):
    t = np.linspace(0, duration, int(fs * duration))
    
    # Phase: 6 Hz theta
    phase = np.sin(2 * np.pi * 6 * t)
    
    # Amplitude: 80 Hz gamma modulated by theta phase
    amp_env = (1 + 0.8 * np.cos(2 * np.pi * 6 * t)) / 2
    carrier = np.sin(2 * np.pi * 80 * t)
    
    # Combine
    signal = phase + amp_env * carrier * 0.5
    
    # Add noise
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(t))
    
    return signal + noise, fs

# Test different parameters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("How Parameters Affect PAC Analysis", fontsize=16, fontweight='bold')

# 1. Effect of frequency resolution
signal, fs = create_signal()
signal_4d = signal.reshape(1, 1, 1, -1)

for i, n_bands in enumerate([10, 30, 100]):
    pac, pha_f, amp_f = gpac.calculate_pac(
        signal_4d, fs=fs,
        pha_n_bands=n_bands,
        amp_n_bands=n_bands,
    )
    
    ax = axes[0, i]
    im = ax.imshow(pac[0, 0].cpu().numpy(), aspect='auto', origin='lower',
                   extent=[60, 120, 2, 20], cmap='viridis')
    ax.set_title(f"Resolution: {n_bands}×{n_bands} bands")
    ax.set_xlabel("Amplitude (Hz)")
    ax.set_ylabel("Phase (Hz)")
    ax.plot(80, 6, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    plt.colorbar(im, ax=ax)

# 2. Effect of signal properties
for i, (param, value, label) in enumerate([
    ('duration', 2.0, 'Short signal (2s)'),
    ('snr_db', 0, 'Low SNR (0 dB)'),
    ('duration', 10.0, 'Long signal (10s)')
]):
    if param == 'duration':
        sig, fs = create_signal(duration=value)
    else:
        sig, fs = create_signal(snr_db=value)
    
    sig_4d = sig.reshape(1, 1, 1, -1)
    
    pac, _, _ = gpac.calculate_pac(
        sig_4d, fs=fs,
        pha_n_bands=50,
        amp_n_bands=30,
    )
    
    ax = axes[1, i]
    im = ax.imshow(pac[0, 0].cpu().numpy(), aspect='auto', origin='lower',
                   extent=[60, 120, 2, 20], cmap='viridis')
    ax.set_title(label)
    ax.set_xlabel("Amplitude (Hz)")
    ax.set_ylabel("Phase (Hz)")
    ax.plot(80, 6, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("resolution_effects.png", dpi=150, bbox_inches='tight')
print("Saved visualization to resolution_effects.png")

# Print key insights
print("\n🔑 KEY INSIGHTS:")
print("1. More frequency bands → finer resolution but noisier estimates")
print("2. Longer signals → cleaner PAC patterns")
print("3. Higher SNR → stronger PAC values")
print("4. Filter cycles affect selectivity (not shown here)")
print("\nFor real-world data:")
print("- Use 30-100 frequency bands for good balance")
print("- Ensure signals are >2-3 seconds long")
print("- Consider SNR when interpreting results")