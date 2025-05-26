#!/usr/bin/env python3
"""Proper comparison between gPAC and TensorPAC accounting for frequency band differences."""

import numpy as np
import torch
from scipy.stats import pearsonr
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gpac import calculate_pac
from tensorpac_source.tensorpac import Pac

def compare_with_explicit_bands():
    """Compare gPAC and TensorPAC using explicit, matching frequency bands."""
    
    # Parameters
    fs = 256
    duration = 5
    n_samples = int(fs * duration)
    
    # Generate test signal with known PAC
    t = np.linspace(0, duration, n_samples, False)
    phase_freq = 10  # Hz
    amp_freq = 80    # Hz
    
    # Create coupled signal
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * phase_freq * t)
    amp_signal = amp_mod * np.sin(2 * np.pi * amp_freq * t)
    signal = phase_signal + amp_signal
    
    print("Comparison using explicit, matching frequency bands")
    print("=" * 60)
    
    # Define matching frequency bands
    n_pha = 10
    n_amp = 10
    
    # Create frequency vectors
    pha_vec = np.linspace(2, 20, n_pha + 1)
    amp_vec = np.linspace(60, 160, n_amp + 1)
    
    # Convert to band pairs for TensorPAC
    f_pha_bands = np.c_[pha_vec[:-1], pha_vec[1:]]
    f_amp_bands = np.c_[amp_vec[:-1], amp_vec[1:]]
    
    print(f"Frequency configuration:")
    print(f"  Phase: {n_pha} bands from {pha_vec[0]:.1f} to {pha_vec[-1]:.1f} Hz")
    print(f"  Amplitude: {n_amp} bands from {amp_vec[0]:.1f} to {amp_vec[-1]:.1f} Hz")
    print()
    
    # 1. gPAC calculation
    print("1. Running gPAC...")
    signal_torch = torch.from_numpy(signal.reshape(1, 1, 1, -1)).float()
    
    pac_gpac, pha_freqs_gpac, amp_freqs_gpac = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_pha,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=n_amp
    )
    
    pac_gpac_np = pac_gpac.cpu().numpy().squeeze()
    print(f"   Shape: {pac_gpac_np.shape}")
    print(f"   Max: {pac_gpac_np.max():.4f}")
    print(f"   Mean: {pac_gpac_np.mean():.4f}")
    
    # 2. TensorPAC calculation with explicit bands
    print("\n2. Running TensorPAC with explicit bands...")
    pac_obj = Pac(
        idpac=(2, 0, 0),  # Tort MI method
        f_pha=f_pha_bands,
        f_amp=f_amp_bands,
        dcomplex='hilbert',
        cycle=(3, 6)
    )
    
    signal_2d = signal.reshape(1, -1)
    pac_tensorpac = pac_obj.filterfit(fs, signal_2d, n_perm=0).squeeze()
    
    print(f"   Shape: {pac_tensorpac.shape}")
    print(f"   Max: {pac_tensorpac.max():.4f}")
    print(f"   Mean: {pac_tensorpac.mean():.4f}")
    
    # 3. Calculate correlation
    if pac_gpac_np.shape == pac_tensorpac.shape:
        corr, p_value = pearsonr(pac_gpac_np.flatten(), pac_tensorpac.flatten())
        print(f"\n📊 Correlation: r={corr:.3f} (p={p_value:.3e})")
        
        # Value ratio
        ratio = pac_gpac_np.max() / pac_tensorpac.max()
        print(f"📊 Max value ratio (gPAC/TensorPAC): {ratio:.3f}")
    else:
        print("\n❌ Shape mismatch - cannot compute correlation")
    
    # 4. Find peaks
    gpac_peak = np.unravel_index(pac_gpac_np.argmax(), pac_gpac_np.shape)
    tensorpac_peak = np.unravel_index(pac_tensorpac.argmax(), pac_tensorpac.shape)
    
    print(f"\n🎯 Peak locations:")
    print(f"   gPAC: phase band {gpac_peak[0]} ({pha_freqs_gpac[gpac_peak[0]]:.1f} Hz), "
          f"amp band {gpac_peak[1]} ({amp_freqs_gpac[gpac_peak[1]]:.1f} Hz)")
    print(f"   TensorPAC: phase band {tensorpac_peak[0]} ({f_pha_bands[tensorpac_peak[0]].mean():.1f} Hz), "
          f"amp band {tensorpac_peak[1]} ({f_amp_bands[tensorpac_peak[1]].mean():.1f} Hz)")
    
    # 5. Test with v01 mode
    print("\n" + "=" * 60)
    print("Testing with v01 mode (depthwise convolution)")
    print("=" * 60)
    
    pac_gpac_v01, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_pha,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=n_amp,
        v01_mode=True  # Use v01 mode
    )
    
    pac_gpac_v01_np = pac_gpac_v01.cpu().numpy().squeeze()
    print(f"gPAC v01 mode:")
    print(f"   Max: {pac_gpac_v01_np.max():.4f}")
    print(f"   Mean: {pac_gpac_v01_np.mean():.4f}")
    
    if pac_gpac_v01_np.shape == pac_tensorpac.shape:
        corr_v01, p_value_v01 = pearsonr(pac_gpac_v01_np.flatten(), pac_tensorpac.flatten())
        print(f"\n📊 Correlation with v01 mode: r={corr_v01:.3f} (p={p_value_v01:.3e})")
        
        # Compare to standard mode
        if corr_v01 > corr:
            print(f"✅ v01 mode improves correlation by {(corr_v01 - corr):.3f}")
        else:
            print(f"❌ v01 mode decreases correlation by {(corr - corr_v01):.3f}")

if __name__ == "__main__":
    compare_with_explicit_bands()