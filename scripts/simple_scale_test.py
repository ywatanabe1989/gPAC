#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simplified test to find the scale difference

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tensorpac_source'))

import numpy as np
import torch
from tensorpac import Pac

def simple_scale_test():
    """Simple test to identify scale difference."""
    
    print("="*60)
    print("SIMPLE SCALE TEST")
    print("="*60)
    
    # Create simple PAC signal
    fs = 1000.0
    t = np.arange(2000) / fs
    
    # 5 Hz phase, 50 Hz amplitude
    phase_signal = np.sin(2 * np.pi * 5 * t)
    amp_carrier = np.sin(2 * np.pi * 50 * t)
    signal = (1 + 0.5 * phase_signal) * amp_carrier
    
    # TensorPAC calculation
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=[[3, 7]], f_amp=[[40, 60]])
    mi_tp = pac_tp.filterfit(fs, signal.reshape(1, -1))
    
    print(f"TensorPAC MI: {mi_tp.item():.6f}")
    
    # Now let's use gPAC's full PAC module
    from gpac._PAC import PAC
    
    pac_gpac = PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=3,
        pha_end_hz=7,
        pha_n_bands=1,
        amp_start_hz=40,
        amp_end_hz=60,
        amp_n_bands=1,
        mi_n_bins=18,
        filtfilt_mode=False,  # Try without filtfilt first
        v01_mode=False
    )
    
    signal_torch = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, 1, -1)
    result = pac_gpac(signal_torch)
    mi_gpac = result['mi'][0, 0, 0, 0].item()
    
    print(f"gPAC MI: {mi_gpac:.6f}")
    print(f"Ratio (gPAC/TensorPAC): {mi_gpac/mi_tp.item():.6f}")
    print(f"Scale factor: {mi_tp.item()/mi_gpac:.2f}x")
    
    # Try with different settings
    print("\n" + "-"*60)
    print("Testing different configurations:")
    print("-"*60)
    
    # Test 1: With filtfilt mode
    pac_gpac_ff = PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=3,
        pha_end_hz=7,
        pha_n_bands=1,
        amp_start_hz=40,
        amp_end_hz=60,
        amp_n_bands=1,
        filtfilt_mode=True,
        v01_mode=False
    )
    
    result_ff = pac_gpac_ff(signal_torch)
    mi_ff = result_ff['mi'][0, 0, 0, 0].item()
    print(f"\nWith filtfilt=True: {mi_ff:.6f} (ratio: {mi_ff/mi_tp.item():.6f})")
    
    # Test 2: With v01 mode
    pac_gpac_v01 = PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=3,
        pha_end_hz=7,
        pha_n_bands=1,
        amp_start_hz=40,
        amp_end_hz=60,
        amp_n_bands=1,
        filtfilt_mode=True,
        v01_mode=True
    )
    
    result_v01 = pac_gpac_v01(signal_torch)
    mi_v01 = result_v01['mi'][0, 0, 0, 0].item()
    print(f"With v01_mode=True: {mi_v01:.6f} (ratio: {mi_v01/mi_tp.item():.6f})")
    
    # Test with stronger coupling
    print("\n" + "-"*60)
    print("Testing with stronger coupling:")
    print("-"*60)
    
    signal_strong = (1 + 0.9 * phase_signal) * amp_carrier
    
    mi_tp_strong = pac_tp.filterfit(fs, signal_strong.reshape(1, -1))
    print(f"\nTensorPAC MI (strong): {mi_tp_strong.item():.6f}")
    
    signal_strong_torch = torch.tensor(signal_strong, dtype=torch.float32).reshape(1, 1, 1, -1)
    result_strong = pac_gpac(signal_strong_torch)
    mi_gpac_strong = result_strong['mi'][0, 0, 0, 0].item()
    
    print(f"gPAC MI (strong): {mi_gpac_strong:.6f}")
    print(f"Ratio: {mi_gpac_strong/mi_tp_strong.item():.6f}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Consistent scale difference: ~{mi_tp.item()/mi_gpac:.1f}x")
    print("="*60)

if __name__ == "__main__":
    simple_scale_test()