#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simplified PAC example that uses the correct API

"""Imports"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""Generate PAC signal"""
def generate_pac_signal(fs=1000, duration=5):
    """Generate synthetic signal with known PAC."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
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
    
    return signal, t, phase_freq, amp_freq, phase_signal, amp_signal, amp_envelope

"""Main execution"""
print("🚀 PAC Analysis Example (Simplified)")
print("=" * 50)

# Import gPAC
from gpac import PAC

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Parameters
fs = 1000  # Hz
duration = 5  # seconds

# Generate synthetic signal with PAC
print("\n📡 Generating synthetic signal with PAC...")
signal, t, phase_freq, amp_freq, phase_signal, amp_signal, amp_envelope = generate_pac_signal(fs, duration)

# Prepare signal for gPAC (batch, channels, time)
signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

print(f"Signal shape: {signal_tensor.shape}")
print(f"Signal duration: {duration} seconds")
print(f"Sampling rate: {fs} Hz")
print(f"True coupling: θ={phase_freq} Hz → γ={amp_freq} Hz")

# Initialize PAC calculator
print("\n🔧 Initializing PAC calculator...")
pac_calculator = PAC(
    seq_len=len(signal),
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=10,
)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
pac_calculator = pac_calculator.to(device)
signal_tensor = signal_tensor.to(device)
print(f"✅ PAC calculator initialized on {device}")

# Calculate PAC
print("\n🔄 Calculating PAC...")
with torch.no_grad():
    pac_results = pac_calculator(signal_tensor)

# Extract PAC values from the results dictionary
pac_values = pac_results["pac"]
print(f"✅ PAC output shape: {pac_values.shape}")

# Convert to numpy for plotting
pac_matrix = pac_values[0, 0].cpu().numpy()

# Get frequency bins
pha_freqs = np.linspace(2, 20, 10)
amp_freqs = np.linspace(60, 120, 10)

# Create visualization
print("\n📊 Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot original signal (first second)
ax = axes[0, 0]
plot_duration = 1  # seconds
plot_samples = int(fs * plot_duration)
ax.plot(t[:plot_samples], signal[:plot_samples], 'b-', linewidth=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Original Signal (first second)")
ax.set_xlim(0, plot_duration)
ax.grid(True, alpha=0.3)

# Plot phase and amplitude components
ax = axes[0, 1]
ax.plot(t[:plot_samples], phase_signal[:plot_samples], 'g-', label=f'Phase ({phase_freq} Hz)')
ax.plot(t[:plot_samples], 0.3 * amp_signal[:plot_samples], 'r-', alpha=0.7, label=f'Amplitude ({amp_freq} Hz)')
ax.plot(t[:plot_samples], 0.3 * amp_envelope[:plot_samples], 'k--', label='Modulation envelope')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Signal Components")
ax.set_xlim(0, plot_duration)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot PAC matrix
ax = axes[1, 0]
im = ax.imshow(
    pac_matrix,
    aspect='auto',
    origin='lower',
    extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
    cmap='hot'
)
ax.set_xlabel("Amplitude Frequency (Hz)")
ax.set_ylabel("Phase Frequency (Hz)")
ax.set_title("Phase-Amplitude Coupling")
plt.colorbar(im, ax=ax, label='PAC Strength')

# Add marker for true coupling
ax.plot(amp_freq, phase_freq, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)
ax.text(amp_freq + 2, phase_freq + 0.5, 'True\ncoupling', color='white', fontsize=9)

# Plot PAC values as line plot
ax = axes[1, 1]
# Find phase band closest to true phase frequency
phase_idx = np.argmin(np.abs(pha_freqs - phase_freq))
ax.plot(amp_freqs, pac_matrix[phase_idx, :], 'b-', linewidth=2)
ax.axvline(amp_freq, color='r', linestyle='--', label=f'True amp freq ({amp_freq} Hz)')
ax.set_xlabel("Amplitude Frequency (Hz)")
ax.set_ylabel("PAC Strength")
ax.set_title(f"PAC values for phase ≈ {pha_freqs[phase_idx]:.1f} Hz")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
spath = "pac_analysis_simple.gif"
plt.savefig(spath, dpi=150, bbox_inches='tight')
print(f"\n💾 Figure saved to: {spath}")

# Report findings
print("\n📊 PAC Analysis Results")
print("=" * 50)
print(f"Maximum PAC value: {pac_matrix.max():.4f}")
max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
print(f"Peak coupling at: Phase {pha_freqs[max_idx[0]]:.1f} Hz, Amplitude {amp_freqs[max_idx[1]]:.1f} Hz")
print(f"Expected coupling: Phase {phase_freq} Hz, Amplitude {amp_freq} Hz")

print("\n✅ PAC analysis completed successfully!")