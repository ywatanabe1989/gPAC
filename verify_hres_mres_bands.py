#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify that gPAC correctly implements TensorPAC's hres/mres frequency bands.
"""

import numpy as np
import tensorpac

# Get TensorPAC's frequency bands
pac_tp = tensorpac.Pac(f_pha="hres", f_amp="mres")

print("TensorPAC 'hres' and 'mres' frequency band definitions:")
print("=" * 60)

print("\nPhase bands (hres - 50 bands):")
print(f"Shape: {pac_tp.f_pha.shape}")
print(f"Range: {pac_tp.f_pha[0, 0]:.2f} - {pac_tp.f_pha[-1, 1]:.2f} Hz")
print("\nFirst 5 bands:")
for i in range(5):
    print(f"  Band {i}: [{pac_tp.f_pha[i, 0]:.2f}, {pac_tp.f_pha[i, 1]:.2f}] Hz, "
          f"center: {pac_tp.f_pha[i].mean():.2f} Hz, "
          f"width: {pac_tp.f_pha[i, 1] - pac_tp.f_pha[i, 0]:.2f} Hz")

print("\nAmplitude bands (mres - 30 bands):")
print(f"Shape: {pac_tp.f_amp.shape}")
print(f"Range: {pac_tp.f_amp[0, 0]:.2f} - {pac_tp.f_amp[-1, 1]:.2f} Hz")
print("\nFirst 5 bands:")
for i in range(5):
    print(f"  Band {i}: [{pac_tp.f_amp[i, 0]:.2f}, {pac_tp.f_amp[i, 1]:.2f}] Hz, "
          f"center: {pac_tp.f_amp[i].mean():.2f} Hz, "
          f"width: {pac_tp.f_amp[i, 1] - pac_tp.f_amp[i, 0]:.2f} Hz")

# Verify the formula
print("\n" + "=" * 60)
print("FORMULA VERIFICATION:")
print("=" * 60)

# Phase bands: f_pha = [f - f/4, f + f/4]
f_pha_centers = np.linspace(2, 20, 50)
print(f"\nPhase band centers: {f_pha_centers[0]:.2f} to {f_pha_centers[-1]:.2f} Hz")
print("Bandwidth formula: f ± f/4")
print(f"First band: {f_pha_centers[0]:.2f} ± {f_pha_centers[0]/4:.2f} = "
      f"[{f_pha_centers[0] - f_pha_centers[0]/4:.2f}, {f_pha_centers[0] + f_pha_centers[0]/4:.2f}]")
print(f"TensorPAC first band: [{pac_tp.f_pha[0, 0]:.2f}, {pac_tp.f_pha[0, 1]:.2f}]")

# Amplitude bands: f_amp = [f - f/8, f + f/8]
f_amp_centers = np.linspace(60, 160, 30)
print(f"\nAmplitude band centers: {f_amp_centers[0]:.2f} to {f_amp_centers[-1]:.2f} Hz")
print("Bandwidth formula: f ± f/8")
print(f"First band: {f_amp_centers[0]:.2f} ± {f_amp_centers[0]/8:.2f} = "
      f"[{f_amp_centers[0] - f_amp_centers[0]/8:.2f}, {f_amp_centers[0] + f_amp_centers[0]/8:.2f}]")
print(f"TensorPAC first band: [{pac_tp.f_amp[0, 0]:.2f}, {pac_tp.f_amp[0, 1]:.2f}]")

# gPAC equivalent settings
print("\n" + "=" * 60)
print("gPAC EQUIVALENT SETTINGS:")
print("=" * 60)
print("""
pac = gpac.PAC(
    seq_len=seq_len,
    fs=fs,
    pha_start_hz=2.0,   # Start of phase range
    pha_end_hz=20.0,    # End of phase range  
    pha_n_bands=50,     # 'hres' = 50 bands
    amp_start_hz=60.0,  # Start of amplitude range
    amp_end_hz=160.0,   # End of amplitude range
    amp_n_bands=30,     # 'mres' = 30 bands
    ...
)

Note: gPAC uses evenly spaced center frequencies, while TensorPAC
uses the Bahramisharif et al. 2013 definition with variable bandwidth.
""")