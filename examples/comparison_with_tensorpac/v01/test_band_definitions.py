#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 12:00:00"
# Author: Claude
# Filename: test_band_definitions.py

"""
Test and compare band definitions between gPAC and TensorPAC
"""

import numpy as np
import torch
from gpac import PAC as gPAC_PAC
from tensorpac import Pac as TensorPAC_Pac
import mngs

# Simple test parameters
fs = 1000
duration = 2
n_samples = fs * duration

# Test with simple band configuration
print("Testing with simple 1-band configuration:")
print("Phase: 4-8 Hz (1 band)")
print("Amplitude: 30-60 Hz (1 band)")
print("-" * 50)

# gPAC test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_signal = torch.randn(1, 1, n_samples).to(device)

pac_gpac = gPAC_PAC(
    seq_len=n_samples,
    fs=fs,
    pha_start_hz=4,
    pha_end_hz=8,
    pha_n_bands=1,
    amp_start_hz=30,
    amp_end_hz=60,
    amp_n_bands=1,
    trainable=False
).to(device)

# Get the actual filter frequencies
output = pac_gpac(test_signal)
print("\ngPAC:")
print(f"  Phase frequencies: {output['phase_frequencies'].cpu().numpy()}")
print(f"  Amplitude frequencies: {output['amplitude_frequencies'].cpu().numpy()}")
print(f"  PAC matrix shape: {output['pac'].shape}")

# Now test with multiple bands
print("\n" + "="*60)
print("Testing with multiple bands:")
print("Phase: 2-20 Hz (10 bands)")
print("Amplitude: 60-160 Hz (10 bands)")
print("-" * 50)

pac_gpac_multi = gPAC_PAC(
    seq_len=n_samples,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=10,
    trainable=False
).to(device)

output_multi = pac_gpac_multi(test_signal)
pha_freqs = output_multi['phase_frequencies'].cpu().numpy()
amp_freqs = output_multi['amplitude_frequencies'].cpu().numpy()

print("\ngPAC Multiple Bands:")
print(f"  Phase frequencies: {pha_freqs}")
print(f"  Amplitude frequencies: {amp_freqs}")
print(f"  PAC matrix shape: {output_multi['pac'].shape}")

# Check band spacing
print("\nPhase band analysis:")
print(f"  Spacing: {np.diff(pha_freqs)}")
print(f"  Linear spacing? {np.allclose(np.diff(pha_freqs), np.diff(pha_freqs)[0])}")

print("\nAmplitude band analysis:")
print(f"  Spacing: {np.diff(amp_freqs)}")
print(f"  Linear spacing? {np.allclose(np.diff(amp_freqs), np.diff(amp_freqs)[0])}")

# Compare with TensorPAC explicit bands
print("\n" + "="*60)
print("TensorPAC with explicit linear bands:")
# Create linear bands
pha_edges = np.linspace(2, 20, 11)
amp_edges = np.linspace(60, 160, 11)
pha_bands_tp = np.c_[pha_edges[:-1], pha_edges[1:]]
amp_bands_tp = np.c_[amp_edges[:-1], amp_edges[1:]]

print(f"\nPhase bands (edges):")
for i, (low, high) in enumerate(pha_bands_tp):
    center = (low + high) / 2
    print(f"  Band {i}: [{low:.1f}, {high:.1f}] Hz, center={center:.1f} Hz")

print(f"\nAmplitude bands (edges):")
for i, (low, high) in enumerate(amp_bands_tp[:5]):  # Show first 5
    center = (low + high) / 2
    print(f"  Band {i}: [{low:.1f}, {high:.1f}] Hz, center={center:.1f} Hz")
print("  ...")

# Save results
results = {
    'gpac_single': {
        'phase_freqs': output['phase_frequencies'].cpu().numpy(),
        'amp_freqs': output['amplitude_frequencies'].cpu().numpy(),
    },
    'gpac_multi': {
        'phase_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'phase_spacing': np.diff(pha_freqs),
        'amp_spacing': np.diff(amp_freqs),
    },
    'tensorpac_explicit': {
        'phase_bands': pha_bands_tp,
        'amp_bands': amp_bands_tp,
        'phase_centers': (pha_bands_tp[:, 0] + pha_bands_tp[:, 1]) / 2,
        'amp_centers': (amp_bands_tp[:, 0] + amp_bands_tp[:, 1]) / 2,
    }
}

mngs.io.save(results, "band_definitions_comparison.pkl")
print(f"\nResults saved")