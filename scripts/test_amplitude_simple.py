#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simple test to understand amplitude differences

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tensorpac_source'))

import numpy as np
from tensorpac import Pac
from scipy.signal import hilbert

def test_amplitude_extraction():
    """Compare how TensorPAC and scipy extract amplitude."""
    
    print("="*60)
    print("SIMPLE AMPLITUDE EXTRACTION TEST")
    print("="*60)
    
    # Create test signal
    fs = 1000.0
    t = np.arange(1000) / fs
    
    # 10 Hz sine with amplitude 2.0
    signal = 2.0 * np.sin(2 * np.pi * 10 * t)
    
    print(f"\nTest: 10 Hz sine, amplitude = 2.0")
    
    # Method 1: Direct scipy hilbert
    print("\n1. Direct scipy.hilbert:")
    analytic = hilbert(signal)
    amp_scipy = np.abs(analytic)
    print(f"   Mean amplitude: {amp_scipy.mean():.6f}")
    print(f"   Expected: 2.0")
    
    # Method 2: TensorPAC filter
    print("\n2. TensorPAC filter(ftype='amplitude'):")
    pac = Pac(f_pha=[[8, 12]], f_amp=[[8, 12]])
    amp_tp = pac.filter(fs, signal.reshape(1, -1), ftype='amplitude')
    print(f"   Mean amplitude: {amp_tp.mean():.6f}")
    print(f"   Shape: {amp_tp.shape}")
    
    # Method 3: TensorPAC filter then manual extraction
    print("\n3. TensorPAC filter(keepfilt=True) then abs:")
    filtered = pac.filter(fs, signal.reshape(1, -1), ftype='amplitude', keepfilt=True)
    amp_manual = np.abs(hilbert(filtered[0, 0]))
    print(f"   Mean amplitude: {amp_manual.mean():.6f}")
    
    # Now test with PAC signal
    print("\n" + "="*60)
    print("PAC SIGNAL TEST")
    print("="*60)
    
    # Create PAC signal
    pha_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz phase
    amp_signal = (1 + 0.5 * pha_signal) * np.sin(2 * np.pi * 50 * t)  # 50 Hz modulated
    
    # Test MI calculation
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=[[3, 7]], f_amp=[[40, 60]])
    mi = pac_tp.filterfit(fs, amp_signal.reshape(1, -1))
    
    print(f"\nTensorPAC MI value: {mi.item():.6f}")
    
    # Check intermediate values
    pha = pac_tp.filter(fs, amp_signal.reshape(1, -1), ftype='phase')
    amp = pac_tp.filter(fs, amp_signal.reshape(1, -1), ftype='amplitude')
    
    print(f"\nExtracted values:")
    print(f"Phase range: [{pha.min():.3f}, {pha.max():.3f}]")
    print(f"Amplitude mean: {amp.mean():.6f}")
    print(f"Amplitude std: {amp.std():.6f}")
    
    # Now let's manually calculate MI using TensorPAC's method
    from tensorpac.methods import modulation_index
    mi_manual = modulation_index(pha, amp, n_bins=18)
    print(f"\nManual MI calculation: {mi_manual.item():.6f}")
    print(f"Matches filterfit: {np.allclose(mi, mi_manual)}")

if __name__ == "__main__":
    test_amplitude_extraction()