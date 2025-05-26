#\!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the TensorPAC compatibility module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from gpac import calculate_pac_tensorpac_compat, compare_with_tensorpac, TENSORPAC_CONFIGS

def test_compatibility():
    """Test TensorPAC compatibility features."""
    
    print("="*80)
    print("TENSORPAC COMPATIBILITY MODULE TEST")
    print("="*80)
    
    # Create test signal
    fs = 1000.0
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    
    # PAC signal: 5 Hz phase modulates 70 Hz amplitude
    phase = np.sin(2 * np.pi * 5 * t)
    signal = (1 + 0.7 * phase) * np.sin(2 * np.pi * 70 * t)
    signal += 0.1 * np.random.randn(len(t))  # Add noise
    
    print("\nTest signal: 5 Hz phase modulates 70 Hz amplitude")
    print(f"Signal length: {len(signal)} samples ({duration} seconds)")
    
    # Test different configurations
    print("\n" + "-"*80)
    print("AVAILABLE CONFIGURATIONS:")
    print("-"*80)
    
    for name, cfg in TENSORPAC_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Phase bands: {cfg['n_pha']}")
        print(f"  Amplitude bands: {cfg['n_amp']}")
        print(f"  Phase range: {cfg['pha_range']} Hz")
        print(f"  Amplitude range: {cfg['amp_range']} Hz")
        print(f"  Scale factor: {cfg['scale_factor']}x")
    
    # Test compatible configuration
    print("\n" + "-"*80)
    print("TESTING 'compatible' CONFIGURATION (50x30):")
    print("-"*80)
    
    pac_scaled, pha_freqs, amp_freqs = calculate_pac_tensorpac_compat(
        signal, fs, config='compatible'
    )
    
    print(f"\nResults:")
    print(f"  PAC shape: {pac_scaled.shape}")
    print(f"  Max PAC value: {pac_scaled.max():.6f}")
    print(f"  Phase frequencies: {pha_freqs.min():.1f} - {pha_freqs.max():.1f} Hz")
    print(f"  Amplitude frequencies: {amp_freqs.min():.1f} - {amp_freqs.max():.1f} Hz")
    
    # Find peak
    peak_idx = np.unravel_index(pac_scaled.argmax(), pac_scaled.shape)
    print(f"  Peak at: {pha_freqs[peak_idx[0]]:.1f} Hz / {amp_freqs[peak_idx[1]]:.1f} Hz")
    
    # Test with unscaled values
    print("\n" + "-"*80)
    print("COMPARING SCALED VS UNSCALED:")
    print("-"*80)
    
    pac_scaled, pha_freqs, amp_freqs, pac_raw = calculate_pac_tensorpac_compat(
        signal, fs, config='compatible', return_unscaled=True
    )
    
    print(f"  Raw gPAC max: {pac_raw.max():.6f}")
    print(f"  Scaled gPAC max: {pac_scaled.max():.6f}")
    print(f"  Applied scale factor: {pac_scaled.max() / pac_raw.max():.1f}x")
    
    # Test comparison function
    print("\n" + "-"*80)
    print("COMPARING WITH TENSORPAC:")
    print("-"*80)
    
    comparison = compare_with_tensorpac(signal, fs, config='compatible')
    
    if 'error' not in comparison:
        print("\nComparison results:")
        print(f"  TensorPAC max MI: {comparison['tensorpac_max']:.6f}")
        print(f"  gPAC scaled max: {comparison['gpac_max_scaled']:.6f}")
        print(f"  gPAC raw max: {comparison['gpac_max_raw']:.6f}")
        print(f"  Actual scale factor: {comparison['actual_scale_factor']:.1f}x")
        print(f"  Applied scale factor: {comparison['applied_scale_factor']:.1f}x")
        print(f"  Peak location match: {comparison['peak_location_match']}")
    else:
        print(f"  {comparison['error']}")
    
    # Test multiple epochs
    print("\n" + "-"*80)
    print("TESTING WITH MULTIPLE EPOCHS:")
    print("-"*80)
    
    # Create 5 epochs
    n_epochs = 5
    epochs_signal = np.vstack([signal + 0.1 * np.random.randn(len(signal)) 
                              for _ in range(n_epochs)])
    
    pac_epochs, _, _ = calculate_pac_tensorpac_compat(
        epochs_signal, fs, config='compatible'
    )
    
    print(f"  Input shape: {epochs_signal.shape}")
    print(f"  Output shape: {pac_epochs.shape}")
    print(f"  Max PAC across epochs: {pac_epochs.max():.6f}")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE:")
    print("="*80)
    print("""
from gpac import calculate_pac_tensorpac_compat

# Basic usage with default 50x30 configuration
pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000)

# Use high-resolution configuration
pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000, config='hres')

# Custom scaling factor
pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000, custom_scale=10.0)
""")

if __name__ == "__main__":
    test_compatibility()
EOF < /dev/null
