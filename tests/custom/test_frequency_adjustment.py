#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
import numpy as np
import torch
from gpac._pac import calculate_pac

# Test frequency adjustment
torch.manual_seed(42)
np.random.seed(42)

# Test case 1: Low sampling rate that requires frequency adjustment
print("Test 1: Low sampling rate requiring frequency adjustment")
signal = np.random.randn(1, 1, 1000)
fs = 250  # Low sampling rate, Nyquist = 125 Hz

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # This should trigger frequency adjustment warnings
    pac_values, pha_freqs, amp_freqs = calculate_pac(
        signal, fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=10,
        amp_start_hz=60,
        amp_end_hz=160,  # This exceeds 90% of Nyquist (112.5 Hz)
        amp_n_bands=8
    )
    
    print(f"  Input fs: {fs} Hz, Nyquist: {fs/2} Hz, Safe limit: {fs/2*0.9:.1f} Hz")
    print(f"  Output shape: {pac_values.shape}")
    print(f"  Phase freq range: {pha_freqs.min():.1f} - {pha_freqs.max():.1f} Hz")
    print(f"  Amp freq range: {amp_freqs.min():.1f} - {amp_freqs.max():.1f} Hz")
    
    if w:
        print(f"  Warnings triggered: {len(w)}")
        for warning in w:
            print(f"    {warning.message}")
    else:
        print("  No warnings triggered")

print("\nTest 2: Very low sampling rate")
signal = np.random.randn(1, 1, 1000)
fs = 100  # Very low sampling rate, Nyquist = 50 Hz

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    pac_values, pha_freqs, amp_freqs = calculate_pac(
        signal, fs,
        pha_start_hz=2,
        pha_end_hz=20,  # This might need adjustment
        pha_n_bands=5,
        amp_start_hz=30,
        amp_end_hz=80,  # This will definitely need adjustment
        amp_n_bands=5
    )
    
    print(f"  Input fs: {fs} Hz, Nyquist: {fs/2} Hz, Safe limit: {fs/2*0.9:.1f} Hz")
    print(f"  Output shape: {pac_values.shape}")
    print(f"  Phase freq range: {pha_freqs.min():.1f} - {pha_freqs.max():.1f} Hz")
    print(f"  Amp freq range: {amp_freqs.min():.1f} - {amp_freqs.max():.1f} Hz")
    
    if w:
        print(f"  Warnings triggered: {len(w)}")
        for warning in w:
            print(f"    {warning.message}")
    else:
        print("  No warnings triggered")

print("\nTest 3: Normal case (no adjustment needed)")
signal = np.random.randn(1, 1, 1000)
fs = 500  # Normal sampling rate

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    pac_values, pha_freqs, amp_freqs = calculate_pac(
        signal, fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=10,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=8
    )
    
    print(f"  Input fs: {fs} Hz, Nyquist: {fs/2} Hz, Safe limit: {fs/2*0.9:.1f} Hz")
    print(f"  Output shape: {pac_values.shape}")
    print(f"  Phase freq range: {pha_freqs.min():.1f} - {pha_freqs.max():.1f} Hz")
    print(f"  Amp freq range: {amp_freqs.min():.1f} - {amp_freqs.max():.1f} Hz")
    
    if w:
        print(f"  Warnings triggered: {len(w)}")
        for warning in w:
            print(f"    {warning.message}")
    else:
        print("  No warnings triggered")

print("\nAll tests completed successfully!")