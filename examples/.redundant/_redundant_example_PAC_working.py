#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./examples/gpac/example_PAC_working.py

"""Working PAC example using correct API"""

import numpy as np
import torch
from gpac import PAC, generate_pac_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate synthetic PAC signal
print("🚀 PAC Analysis with gPAC")
print("=" * 50)

# Parameters
fs = 1000  # Hz
duration = 5  # seconds

# Generate signal with known PAC
print("\n📡 Generating synthetic PAC signal...")
signal = generate_pac_signal(
    duration=duration,
    fs=fs,
    phase_freq=6.0,  # Theta
    amp_freq=80.0,   # Gamma  
    coupling_strength=0.7,
    noise_level=0.1
)

# Convert to torch tensor
signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
print(f"Signal shape: {signal_tensor.shape}")

# Initialize PAC calculator
print("\n🔧 Initializing PAC calculator...")
pac = PAC(
    seq_len=len(signal),
    fs=fs,
    pha_start_hz=4,
    pha_end_hz=10,
    pha_n_bands=7,
    amp_start_hz=60,
    amp_end_hz=100,
    amp_n_bands=8,
)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pac = pac.to(device)
signal_tensor = signal_tensor.to(device)
print(f"✅ Using device: {device}")

# Calculate PAC
print("\n🔄 Calculating PAC...")
with torch.no_grad():
    results = pac(signal_tensor)

# Extract PAC values
pac_values = results["pac"][0, 0].cpu().numpy()
pha_freqs = results["phase_frequencies"].cpu().numpy()
amp_freqs = results["amplitude_frequencies"].cpu().numpy()

print(f"✅ PAC matrix shape: {pac_values.shape}")

# Find peak coupling
max_idx = np.unravel_index(pac_values.argmax(), pac_values.shape)
print(f"\n📊 Results:")
print(f"Maximum PAC value: {pac_values.max():.4f}")
print(f"Peak at: θ={pha_freqs[max_idx[0]]:.1f} Hz, γ={amp_freqs[max_idx[1]]:.1f} Hz")
print(f"Expected: θ=6.0 Hz, γ=80.0 Hz")

# Visualization
print("\n📊 Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Signal plot
t = np.arange(len(signal)) / fs
ax1.plot(t[:1000], signal[:1000], 'b-', linewidth=0.5)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Synthetic PAC Signal (first second)")
ax1.grid(True, alpha=0.3)

# PAC matrix
im = ax2.imshow(
    pac_values,
    aspect='auto',
    origin='lower',
    extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
    cmap='hot'
)
ax2.set_xlabel("Amplitude Frequency (Hz)")
ax2.set_ylabel("Phase Frequency (Hz)")
ax2.set_title("Phase-Amplitude Coupling")
plt.colorbar(im, ax=ax2, label='PAC Strength')

# Mark true coupling
ax2.plot(80, 6, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)
ax2.text(82, 6, 'True', color='white', fontsize=9)

plt.tight_layout()
plt.savefig("pac_working_demo.gif", dpi=150, bbox_inches='tight')
print(f"💾 Saved to: pac_working_demo.gif")

print("\n✅ PAC analysis completed successfully!")