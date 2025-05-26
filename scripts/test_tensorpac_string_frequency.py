#!/usr/bin/env python3
"""Test TensorPAC's string frequency configuration behavior."""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tensorpac_source.tensorpac import Pac
from tensorpac_source.tensorpac.utils import pac_vec

# Test different string configurations
print("Testing TensorPAC string frequency configurations:\n")

configs = ['lres', 'mres', 'hres']
for config in configs:
    f_pha, f_amp = pac_vec(config, config)
    print(f"Config: '{config}'")
    print(f"  Phase bands: {len(f_pha)} bands from {f_pha[0,0]:.1f} to {f_pha[-1,1]:.1f} Hz")
    print(f"  Amp bands: {len(f_amp)} bands from {f_amp[0,0]:.1f} to {f_amp[-1,1]:.1f} Hz")
    print(f"  Phase band example: {f_pha[0]} (first), {f_pha[-1]} (last)")
    print(f"  Amp band example: {f_amp[0]} (first), {f_amp[-1]} (last)")
    print()

# Now test with explicit frequency vectors to match gPAC
print("\nTesting explicit frequency vectors (to match gPAC):")

# gPAC style: 10 phase bands from 2-20 Hz, 10 amp bands from 60-160 Hz
pha_vec = np.linspace(2, 20, 11)  # 11 points = 10 bands
amp_vec = np.linspace(60, 160, 11)

# Convert to band pairs
f_pha_explicit = np.c_[pha_vec[:-1], pha_vec[1:]]
f_amp_explicit = np.c_[amp_vec[:-1], amp_vec[1:]]

print(f"Explicit configuration:")
print(f"  Phase bands: {len(f_pha_explicit)} bands")
print(f"  First 3 phase bands: {f_pha_explicit[:3]}")
print(f"  Amp bands: {len(f_amp_explicit)} bands")
print(f"  First 3 amp bands: {f_amp_explicit[:3]}")

# Test with Pac object
print("\nTesting Pac object initialization:")

# String-based
pac_string = Pac(idpac=(2, 0, 0), f_pha='mres', f_amp='mres')
print(f"String-based Pac:")
print(f"  Phase bands shape: {pac_string.f_pha.shape}")
print(f"  Amp bands shape: {pac_string.f_amp.shape}")

# Explicit bands
pac_explicit = Pac(idpac=(2, 0, 0), f_pha=f_pha_explicit, f_amp=f_amp_explicit)
print(f"\nExplicit-bands Pac:")
print(f"  Phase bands shape: {pac_explicit.f_pha.shape}")
print(f"  Amp bands shape: {pac_explicit.f_amp.shape}")

# Test signal processing
print("\nTesting signal processing:")
fs = 256
duration = 5
t = np.linspace(0, duration, int(fs * duration), False)

# Create test signal with PAC
phase_freq = 10  # Hz
amp_freq = 80    # Hz
phase_signal = np.sin(2 * np.pi * phase_freq * t)
amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * phase_freq * t)
amp_signal = amp_mod * np.sin(2 * np.pi * amp_freq * t)
signal = phase_signal + amp_signal
signal = signal.reshape(1, -1)  # (1 epoch, n_times)

# Process with both configurations
pac_string_result = pac_string.filterfit(fs, signal, n_perm=0)
pac_explicit_result = pac_explicit.filterfit(fs, signal, n_perm=0)

print(f"\nResults:")
print(f"String-based PAC shape: {pac_string_result.shape}")
print(f"String-based PAC max: {pac_string_result.max():.4f}")
print(f"Explicit-bands PAC shape: {pac_explicit_result.shape}")
print(f"Explicit-bands PAC max: {pac_explicit_result.max():.4f}")

# Find peak locations
string_peak = np.unravel_index(pac_string_result.argmax(), pac_string_result.shape)
explicit_peak = np.unravel_index(pac_explicit_result.argmax(), pac_explicit_result.shape)

print(f"\nPeak locations:")
print(f"String-based peak: phase band {string_peak[1]}, amp band {string_peak[2]}")
print(f"Explicit-bands peak: phase band {explicit_peak[1]}, amp band {explicit_peak[2]}")

# Compare frequency ranges
print(f"\nFrequency at peaks:")
if len(string_peak) == 3:
    pha_band_str = pac_string.f_pha[string_peak[1]]
    amp_band_str = pac_string.f_amp[string_peak[2]]
    print(f"String-based: phase {pha_band_str}, amp {amp_band_str}")
    
if len(explicit_peak) == 3:
    pha_band_exp = pac_explicit.f_pha[explicit_peak[1]]
    amp_band_exp = pac_explicit.f_amp[explicit_peak[2]]
    print(f"Explicit-bands: phase {pha_band_exp}, amp {amp_band_exp}")