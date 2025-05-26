#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test amplitude normalization differences

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tensorpac_source'))

import numpy as np
import torch
from tensorpac import Pac
from scipy.signal import hilbert
import gpac

def test_simple_signal():
    """Test with a very simple signal to understand amplitude differences."""
    
    print("="*80)
    print("AMPLITUDE NORMALIZATION TEST")
    print("="*80)
    
    # Create a simple test signal
    fs = 1000.0
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    
    # Pure 10 Hz signal with known amplitude
    signal = 2.0 * np.sin(2 * np.pi * 10 * t)
    
    print(f"\nTest signal: 10 Hz sine wave, amplitude = 2.0")
    print(f"Signal shape: {signal.shape}")
    print(f"Signal min/max: {signal.min():.3f} / {signal.max():.3f}")
    
    # Test 1: Direct Hilbert transform
    print("\n1. DIRECT HILBERT TRANSFORM:")
    print("-" * 40)
    analytic = hilbert(signal)
    direct_amp = np.abs(analytic)
    direct_phase = np.angle(analytic)
    print(f"Amplitude mean: {direct_amp.mean():.6f}")
    print(f"Amplitude std: {direct_amp.std():.6f}")
    print(f"Amplitude min/max: {direct_amp.min():.6f} / {direct_amp.max():.6f}")
    
    # Test 2: TensorPAC filter method
    print("\n2. TENSORPAC FILTER METHOD:")
    print("-" * 40)
    
    # Create a PAC object with a single band around 10 Hz
    pac_obj = Pac(f_pha=[[8, 12]], f_amp=[[8, 12]], dcomplex='hilbert')
    
    # Filter for amplitude
    amp_tensorpac = pac_obj.filter(fs, signal.reshape(1, -1), ftype='amplitude')
    print(f"TensorPAC amplitude shape: {amp_tensorpac.shape}")
    print(f"Amplitude mean: {amp_tensorpac.mean():.6f}")
    print(f"Amplitude std: {amp_tensorpac.std():.6f}")
    print(f"Amplitude min/max: {amp_tensorpac.min():.6f} / {amp_tensorpac.max():.6f}")
    
    # Filter for phase
    pha_tensorpac = pac_obj.filter(fs, signal.reshape(1, -1), ftype='phase')
    print(f"\nPhase min/max: {pha_tensorpac.min():.6f} / {pha_tensorpac.max():.6f}")
    
    # Test 3: gPAC approach
    print("\n3. GPAC APPROACH:")
    print("-" * 40)
    
    # Create filter and Hilbert modules
    pha_bands = torch.tensor([[8.0, 12.0]])
    amp_bands = torch.tensor([[8.0, 12.0]])
    
    # Process through gPAC modules
    signal_tensor = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, -1)
    
    # BandPassFilter
    bp_filter = gpac._BandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=signal.shape[0]
    )
    filtered = bp_filter(signal_tensor)
    print(f"Filtered shape: {filtered.shape}")
    
    # Hilbert transform
    hilbert_module = gpac._Hilbert(seq_len=signal.shape[0])
    analytic_gpac = hilbert_module(filtered)
    
    # Extract amplitude (assuming it's the second band)
    amp_gpac = analytic_gpac[0, 1, :, 1].numpy()  # [batch, band, time, pha/amp]
    pha_gpac = analytic_gpac[0, 0, :, 0].numpy()
    
    print(f"gPAC amplitude mean: {amp_gpac.mean():.6f}")
    print(f"gPAC amplitude std: {amp_gpac.std():.6f}")
    print(f"gPAC amplitude min/max: {amp_gpac.min():.6f} / {amp_gpac.max():.6f}")
    print(f"\nPhase min/max: {pha_gpac.min():.6f} / {pha_gpac.max():.6f}")
    
    # Compare ratios
    print("\n4. AMPLITUDE RATIOS:")
    print("-" * 40)
    print(f"TensorPAC / Direct Hilbert: {amp_tensorpac.mean() / direct_amp.mean():.6f}")
    print(f"gPAC / Direct Hilbert: {amp_gpac.mean() / direct_amp.mean():.6f}")
    print(f"gPAC / TensorPAC: {amp_gpac.mean() / amp_tensorpac.mean():.6f}")
    
    # Test with actual PAC calculation
    print("\n5. PAC VALUES:")
    print("-" * 40)
    
    # Create a signal with PAC
    pha_freq = 5.0
    amp_freq = 50.0
    pha_signal = np.sin(2 * np.pi * pha_freq * t)
    amp_signal = (1 + 0.5 * pha_signal) * np.sin(2 * np.pi * amp_freq * t)
    
    # TensorPAC
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=[[3, 7]], f_amp=[[40, 60]], dcomplex='hilbert')
    mi_tensorpac = pac_tp.filterfit(fs, amp_signal.reshape(1, -1))
    print(f"TensorPAC MI: {mi_tensorpac[0, 0]:.6f}")
    
    # gPAC
    pac_gpac = gpac._PAC(
        seq_len=len(amp_signal),
        fs=fs,
        pha_start_hz=3,
        pha_end_hz=7,
        pha_n_bands=1,
        amp_start_hz=40,
        amp_end_hz=60,
        amp_n_bands=1,
        mi_n_bins=18
    )
    
    result_gpac = pac_gpac(torch.tensor(amp_signal, dtype=torch.float32).reshape(1, 1, 1, -1))
    mi_gpac = result_gpac['mi'][0, 0, 0, 0].item()
    print(f"gPAC MI: {mi_gpac:.6f}")
    print(f"Ratio (gPAC/TensorPAC): {mi_gpac / mi_tensorpac[0, 0]:.6f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_simple_signal()