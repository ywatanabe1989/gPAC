#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 12:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/filter_length_comparison.py
# ----------------------------------------
"""
Compare filter lengths between gPAC and TensorPAC implementations.
"""

import numpy as np
import torch
import gpac
from gpac._tensorpac_fir1 import fir_order, design_filter_tensorpac

# Try to import tensorpac
try:
    from tensorpac import Pac
    from tensorpac.spectral import fir_order as tp_fir_order
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("⚠️  TensorPAC not available")


def compare_filter_lengths():
    """Compare filter lengths for various parameters."""
    # Common parameters
    fs = 512.0
    seq_len = 2560  # 5 seconds at 512 Hz
    
    # Phase and amplitude frequency ranges
    pha_freqs = [(2, 10), (4, 8), (6, 12)]
    amp_freqs = [(30, 80), (60, 120), (80, 160)]
    
    print("=" * 80)
    print("FILTER LENGTH COMPARISON")
    print("=" * 80)
    print(f"Sampling rate: {fs} Hz")
    print(f"Signal length: {seq_len} samples ({seq_len/fs:.1f} seconds)")
    print()
    
    # Test phase filters (cycle=3)
    print("PHASE FILTERS (cycle=3):")
    print("-" * 60)
    print(f"{'Freq Range':^15} | {'gPAC Order':^12} | {'gPAC Length':^12} | {'TP Order':^12} | {'TP Length':^12}")
    print("-" * 60)
    
    for low_hz, high_hz in pha_freqs:
        # gPAC filter order
        gpac_order = fir_order(fs, seq_len, low_hz, cycle=3)
        gpac_length = gpac_order + 1
        
        # TensorPAC filter order (if available)
        if TENSORPAC_AVAILABLE:
            tp_order = tp_fir_order(fs, seq_len, low_hz, cycle=3)
            tp_length = tp_order + 1
        else:
            tp_order = "N/A"
            tp_length = "N/A"
        
        print(f"{low_hz:>5}-{high_hz:<5} Hz | {gpac_order:^12} | {gpac_length:^12} | {tp_order:^12} | {tp_length:^12}")
    
    print()
    
    # Test amplitude filters (cycle=6)
    print("AMPLITUDE FILTERS (cycle=6):")
    print("-" * 60)
    print(f"{'Freq Range':^15} | {'gPAC Order':^12} | {'gPAC Length':^12} | {'TP Order':^12} | {'TP Length':^12}")
    print("-" * 60)
    
    for low_hz, high_hz in amp_freqs:
        # gPAC filter order
        gpac_order = fir_order(fs, seq_len, low_hz, cycle=6)
        gpac_length = gpac_order + 1
        
        # TensorPAC filter order (if available)
        if TENSORPAC_AVAILABLE:
            tp_order = tp_fir_order(fs, seq_len, low_hz, cycle=6)
            tp_length = tp_order + 1
        else:
            tp_order = "N/A"
            tp_length = "N/A"
        
        print(f"{low_hz:>5}-{high_hz:<5} Hz | {gpac_order:^12} | {gpac_length:^12} | {tp_order:^12} | {tp_length:^12}")
    
    print()
    
    # Check the actual filter coefficients
    print("ACTUAL FILTER COEFFICIENTS CHECK:")
    print("-" * 60)
    
    # Test one specific case
    low_hz, high_hz = 6.0, 12.0
    cycle = 3
    
    # gPAC filter
    gpac_filter = design_filter_tensorpac(seq_len, fs, low_hz, high_hz, cycle=cycle)
    print(f"gPAC filter for {low_hz}-{high_hz} Hz (cycle={cycle}):")
    print(f"  Shape: {gpac_filter.shape}")
    print(f"  Length: {len(gpac_filter)}")
    print(f"  Non-zero coefficients: {torch.sum(gpac_filter != 0).item()}")
    
    # Create PAC object to see what it actually uses
    pac_model = gpac.PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=10,
        amp_start_hz=60.0,
        amp_end_hz=120.0,
        amp_n_bands=10,
    )
    
    print(f"\nPAC model filter info:")
    print(f"  Phase bands: {pac_model.pha_n_bands}")
    print(f"  Amplitude bands: {pac_model.amp_n_bands}")
    print(f"  Phase cycle parameter: {pac_model.filter_cycle_pha}")
    print(f"  Amplitude cycle parameter: {pac_model.filter_cycle_amp}")
    
    # Check the actual filter lengths by looking at the combined filter
    if hasattr(pac_model, 'comb_band_pass_filter'):
        cbpf = pac_model.comb_band_pass_filter
        print(f"\nActual filter kernels:")
        if hasattr(cbpf, 'pha_kernels'):
            print(f"  Phase filter kernels shape: {cbpf.pha_kernels.shape}")
            print(f"  Phase filter length: {cbpf.pha_kernels.shape[-1]}")
        if hasattr(cbpf, 'amp_kernels'):
            print(f"  Amplitude filter kernels shape: {cbpf.amp_kernels.shape}")
            print(f"  Amplitude filter length: {cbpf.amp_kernels.shape[-1]}")
    
    # Compare with TensorPAC if available
    if TENSORPAC_AVAILABLE:
        print(f"\nTensorPAC comparison:")
        pac_tp = Pac(
            idpac=(2, 0, 0),
            f_pha=np.array([6, 12]),
            f_amp=np.array([60, 120]),
            cycle=(3, 6),
        )
        
        # Generate dummy signal to trigger filter creation
        dummy_signal = np.random.randn(seq_len)
        try:
            # This will create the filters internally
            _ = pac_tp.filter(fs, dummy_signal, njobs=1)
            
            # Try to access filter info if possible
            if hasattr(pac_tp, '_pha_filter'):
                print(f"  TensorPAC phase filter info available")
            if hasattr(pac_tp, '_amp_filter'):
                print(f"  TensorPAC amplitude filter info available")
        except Exception as e:
            print(f"  Could not access TensorPAC filter details: {e}")


def check_frequency_resolution():
    """Check the frequency resolution of both implementations."""
    print("\n" + "=" * 80)
    print("FREQUENCY RESOLUTION ANALYSIS")
    print("=" * 80)
    
    fs = 512.0
    seq_len = 2560
    
    # Calculate theoretical frequency resolution
    freq_resolution = fs / seq_len
    print(f"Theoretical frequency resolution (fs/N): {freq_resolution:.4f} Hz")
    print(f"This is the minimum resolvable frequency difference\n")
    
    # Check how cycle parameter affects filter length
    print("Filter length formula: order = cycle × (fs // f_low)")
    print("Filter length = order + 1\n")
    
    for f_low in [2.0, 4.0, 6.0, 8.0]:
        print(f"For f_low = {f_low} Hz:")
        for cycle in [3, 6]:
            order = cycle * (fs // f_low)
            length = order + 1
            # Effective frequency resolution of the filter
            eff_resolution = fs / length
            print(f"  cycle={cycle}: order={order}, length={length}, effective resolution={eff_resolution:.4f} Hz")
    
    print("\nNOTE: Longer filters (higher cycle count) provide better frequency selectivity")
    print("but require more computation and can introduce more delay/artifacts.")


if __name__ == "__main__":
    compare_filter_lengths()
    check_frequency_resolution()