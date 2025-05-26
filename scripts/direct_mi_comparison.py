#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Direct comparison of MI values between gPAC and TensorPAC

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tensorpac_source'))

import numpy as np
import torch
from tensorpac import Pac
from tensorpac.methods import modulation_index

# Import gPAC components directly to avoid init issues
from gpac._BandPassFilter import BandPassFilter
from gpac._Hilbert import Hilbert
from gpac._ModulationIndex import ModulationIndex

def compare_mi_calculation():
    """Compare MI calculation step by step."""
    
    print("="*80)
    print("DIRECT MI COMPARISON: gPAC vs TensorPAC")
    print("="*80)
    
    # Create PAC test signal
    np.random.seed(42)
    torch.manual_seed(42)
    
    fs = 1000.0
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    
    # Create signal with known PAC
    pha_freq = 5.0
    amp_freq = 50.0
    coupling_strength = 0.7
    
    pha_signal = np.sin(2 * np.pi * pha_freq * t)
    amp_signal = (1 + coupling_strength * pha_signal) * np.sin(2 * np.pi * amp_freq * t)
    
    print(f"Test signal: {pha_freq} Hz phase modulates {amp_freq} Hz amplitude")
    print(f"Coupling strength: {coupling_strength}")
    
    # TENSORPAC PROCESSING
    print("\n" + "-"*40)
    print("TENSORPAC PROCESSING")
    print("-"*40)
    
    # Define frequency bands
    pha_band = [[3, 7]]
    amp_band = [[40, 60]]
    
    # Method 1: filterfit (all-in-one)
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=pha_band, f_amp=amp_band, dcomplex='hilbert')
    mi_tensorpac_auto = pac_tp.filterfit(fs, amp_signal.reshape(1, -1))
    print(f"TensorPAC filterfit MI: {mi_tensorpac_auto.item():.6f}")
    
    # Method 2: Manual extraction
    pha_tp = pac_tp.filter(fs, amp_signal.reshape(1, -1), ftype='phase')
    amp_tp = pac_tp.filter(fs, amp_signal.reshape(1, -1), ftype='amplitude')
    
    print(f"\nExtracted values:")
    print(f"  Phase shape: {pha_tp.shape}")
    print(f"  Phase range: [{pha_tp.min():.3f}, {pha_tp.max():.3f}]")
    print(f"  Amplitude shape: {amp_tp.shape}")
    print(f"  Amplitude mean: {amp_tp.mean():.6f}")
    
    # Manual MI calculation
    mi_tensorpac_manual = modulation_index(pha_tp, amp_tp, n_bins=18)
    print(f"\nManual MI calculation: {mi_tensorpac_manual.item():.6f}")
    
    # GPAC PROCESSING
    print("\n" + "-"*40)
    print("GPAC PROCESSING")
    print("-"*40)
    
    # Convert to torch tensors
    signal_torch = torch.tensor(amp_signal, dtype=torch.float32).reshape(1, 1, -1)
    
    # Step 1: BandPass filtering
    bp_filter = BandPassFilter(
        pha_bands=torch.tensor([[3.0, 7.0]]),
        amp_bands=torch.tensor([[40.0, 60.0]]),
        fs=fs,
        seq_len=len(amp_signal),
        v01_mode=False  # Use scipy-compatible mode
    )
    
    filtered = bp_filter(signal_torch)
    print(f"Filtered shape: {filtered.shape}")  # (batch, bands, time)
    
    # Step 2: Hilbert transform
    # Split filtered into phase and amplitude bands
    n_pha_bands = 1
    pha_filtered = filtered[:, :n_pha_bands, :]  # Shape: (1, 1, 2000)
    amp_filtered = filtered[:, n_pha_bands:, :]   # Shape: (1, 1, 2000)
    
    hilbert_module = Hilbert(seq_len=len(amp_signal))
    
    # Process phase and amplitude separately
    pha_analytic = hilbert_module(pha_filtered.unsqueeze(0).unsqueeze(0))  # Add dims
    amp_analytic = hilbert_module(amp_filtered.unsqueeze(0).unsqueeze(0))  # Add dims
    
    print(f"Phase analytic shape: {pha_analytic.shape}")
    print(f"Amp analytic shape: {amp_analytic.shape}")
    
    # Extract phase and amplitude
    # Shape: (B, C, F, Seg, Time, 2) where last dim is [phase, amplitude]
    pha_gpac = pha_analytic[0, 0, 0, 0, :, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Phase values
    amp_gpac = amp_analytic[0, 0, 0, 0, :, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Amplitude values
    
    print(f"\nExtracted values:")
    print(f"  Phase shape: {pha_gpac.shape}")
    print(f"  Phase range: [{pha_gpac.min():.3f}, {pha_gpac.max():.3f}]")
    print(f"  Amplitude shape: {amp_gpac.shape}")
    print(f"  Amplitude mean: {amp_gpac.mean():.6f}")
    
    # Step 3: Modulation Index
    mi_module = ModulationIndex(n_bins=18)
    result = mi_module(pha_gpac, amp_gpac)
    mi_gpac = result['mi'].item()
    
    print(f"\ngPAC MI: {mi_gpac:.6f}")
    
    # COMPARISON
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"TensorPAC MI: {mi_tensorpac_auto.item():.6f}")
    print(f"gPAC MI: {mi_gpac:.6f}")
    print(f"Ratio (gPAC/TensorPAC): {mi_gpac/mi_tensorpac_auto.item():.6f}")
    print(f"Scale difference: {1/(mi_gpac/mi_tensorpac_auto.item()):.2f}x")
    
    # Compare intermediate values
    print("\nIntermediate value comparison:")
    print(f"Phase range - TensorPAC: [{pha_tp.min():.3f}, {pha_tp.max():.3f}]")
    print(f"Phase range - gPAC: [{pha_gpac.min().item():.3f}, {pha_gpac.max().item():.3f}]")
    print(f"Amplitude mean - TensorPAC: {amp_tp.mean():.6f}")
    print(f"Amplitude mean - gPAC: {amp_gpac.mean().item():.6f}")
    
    # Test with v01 mode
    print("\n" + "-"*40)
    print("TESTING V01 MODE")
    print("-"*40)
    
    bp_filter_v01 = BandPassFilter(
        pha_bands=torch.tensor([[3.0, 7.0]]),
        amp_bands=torch.tensor([[40.0, 60.0]]),
        fs=fs,
        seq_len=len(amp_signal),
        v01_mode=True  # Use v01 depthwise convolution
    )
    
    filtered_v01 = bp_filter_v01(signal_torch)
    
    # Split and process separately
    pha_filtered_v01 = filtered_v01[:, :n_pha_bands, :]
    amp_filtered_v01 = filtered_v01[:, n_pha_bands:, :]
    
    pha_analytic_v01 = hilbert_module(pha_filtered_v01.unsqueeze(0).unsqueeze(0))
    amp_analytic_v01 = hilbert_module(amp_filtered_v01.unsqueeze(0).unsqueeze(0))
    
    pha_v01 = pha_analytic_v01[0, 0, 0, 0, :, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    amp_v01 = amp_analytic_v01[0, 0, 0, 0, :, 1].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    result_v01 = mi_module(pha_v01, amp_v01)
    mi_v01 = result_v01['mi'].item()
    
    print(f"gPAC MI (v01 mode): {mi_v01:.6f}")
    print(f"Ratio v01/TensorPAC: {mi_v01/mi_tensorpac_auto.item():.6f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_mi_calculation()