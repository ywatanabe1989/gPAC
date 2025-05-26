#!/usr/bin/env python3
"""Check current correlation between gPAC and TensorPAC."""

import numpy as np
import torch
from scipy.stats import pearsonr
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gpac import calculate_pac
from tensorpac_source.tensorpac import Pac

def main():
    # Parameters
    fs = 256
    duration = 5
    n_samples = int(fs * duration)
    
    # Generate test signal with known PAC
    t = np.linspace(0, duration, n_samples, False)
    phase_freq = 10  # Hz
    amp_freq = 80    # Hz
    
    # Phase signal
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    
    # Amplitude modulation (0.5 + 0.5 * phase for 100% modulation)
    amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * phase_freq * t)
    
    # Amplitude signal
    amp_signal = amp_mod * np.sin(2 * np.pi * amp_freq * t)
    
    # Combined signal
    signal = phase_signal + amp_signal
    signal = signal.reshape(1, 1, -1)  # (n_epochs, n_channels, n_times)
    
    print("Testing gPAC vs TensorPAC correlation...")
    print(f"Signal shape: {signal.shape}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {duration} s")
    print(f"Phase frequency: {phase_freq} Hz")
    print(f"Amplitude frequency: {amp_freq} Hz")
    print()
    
    # gPAC calculation
    try:
        signal_torch = torch.from_numpy(signal).float()
        result = calculate_pac(
            signal_torch,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=10,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=10
        )
        # calculate_pac returns tuple: (pac_values, pha_freqs, amp_freqs)
        pac_gpac, pha_freqs, amp_freqs = result
        pac_gpac_np = pac_gpac.cpu().numpy()
        print(f"✅ gPAC output shape: {pac_gpac_np.shape}")
        print(f"   gPAC max value: {pac_gpac_np.max():.4f}")
        print(f"   gPAC mean value: {pac_gpac_np.mean():.4f}")
    except Exception as e:
        print(f"❌ gPAC failed: {e}")
        return
    
    # TensorPAC calculation
    try:
        p_obj = Pac(
            idpac=(2, 0, 0),  # Tort method
            f_pha=(2, 20, 10),
            f_amp=(60, 160, 10),
            dcomplex='hilbert',
            cycle=(3, 6),
            width=7
        )
        
        # Compute PAC
        # TensorPAC expects 3D array (n_epochs, n_times, n_channels) or 2D
        signal_2d = signal.squeeze(1)  # Remove channel dimension to get (1, n_times)
        pac_tensorpac = p_obj.filterfit(fs, signal_2d, n_perm=0).squeeze()
        
        print(f"✅ TensorPAC output shape: {pac_tensorpac.shape}")
        print(f"   TensorPAC max value: {pac_tensorpac.max():.4f}")
        print(f"   TensorPAC mean value: {pac_tensorpac.mean():.4f}")
    except Exception as e:
        print(f"❌ TensorPAC failed: {e}")
        return
    
    # Flatten and compute correlation
    pac_gpac_flat = pac_gpac_np.flatten()
    pac_tensorpac_flat = pac_tensorpac.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(pac_gpac_flat) | np.isnan(pac_tensorpac_flat))
    pac_gpac_flat = pac_gpac_flat[mask]
    pac_tensorpac_flat = pac_tensorpac_flat[mask]
    
    if len(pac_gpac_flat) > 0:
        corr, p_value = pearsonr(pac_gpac_flat, pac_tensorpac_flat)
        print(f"\n📊 Correlation: r={corr:.3f} (p={p_value:.3e})")
        
        # Check for compatibility layer
        try:
            from src.gpac._calculate_gpac_tensorpac_compat import calculate_pac_tensorpac_compat
            result_compat = calculate_pac_tensorpac_compat(
                signal_torch,
                fs=fs,
                pha_start_hz=2,
                pha_end_hz=20,
                pha_n_bands=10,
                amp_start_hz=60,
                amp_end_hz=160,
                amp_n_bands=10
            )
            # Assuming it returns same format as calculate_pac
            if isinstance(result_compat, tuple):
                pac_compat = result_compat[0]
            else:
                pac_compat = result_compat
            pac_compat_np = pac_compat.cpu().numpy() if hasattr(pac_compat, 'cpu') else pac_compat.numpy()
            pac_compat_flat = pac_compat_np.flatten()[mask]
            corr_compat, p_compat = pearsonr(pac_compat_flat, pac_tensorpac_flat)
            print(f"\n📊 With compatibility layer: r={corr_compat:.3f} (p={p_compat:.3e})")
            print(f"   Compat max value: {pac_compat_np.max():.4f}")
            print(f"   Scaling factor: {pac_compat_np.max() / pac_gpac_np.max():.2f}x")
        except ImportError:
            print("\n⚠️  Compatibility layer not found")
    else:
        print("\n❌ No valid values for correlation")
    
    # Find peak locations
    gpac_peak_idx = np.unravel_index(pac_gpac_np.argmax(), pac_gpac_np.shape)
    tensorpac_peak_idx = np.unravel_index(pac_tensorpac.argmax(), pac_tensorpac.shape)
    
    print(f"\n🎯 Peak locations:")
    print(f"   gPAC peak at phase band {gpac_peak_idx[-2]}, amp band {gpac_peak_idx[-1]}")
    print(f"   TensorPAC peak at phase band {tensorpac_peak_idx[-2]}, amp band {tensorpac_peak_idx[-1]}")
    
    # Check if v01 implementation exists
    if os.path.exists('/home/ywatanabe/proj/gPAC/src/gpac/v01/_CombinedBandPassFilter_v01_working.py'):
        print("\n✅ v01 implementation found - this was reported to have high correlation")
    else:
        print("\n⚠️  v01 implementation not found")

if __name__ == "__main__":
    main()