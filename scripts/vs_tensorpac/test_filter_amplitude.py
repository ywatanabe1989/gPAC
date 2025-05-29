#!/usr/bin/env python3
"""Test to compare filter outputs between gPAC and TensorPAC."""

import numpy as np
import torch
from scipy.signal import filtfilt
from tensorpac.spectral import fir_order, fir1
from tensorpac import Pac

# Import gPAC components
import sys
sys.path.append('/home/ywatanabe/proj/gPAC/src')
from gpac._BandPassFilter import BandPassFilter
from gpac._Hilbert import Hilbert


def test_filter_amplitude_comparison():
    """Compare filter outputs between implementations."""
    # Create test signal with known PAC
    np.random.seed(42)
    sf = 1000  # Hz
    n_times = 2000
    t = np.linspace(0, 2, n_times)
    
    # Phase signal (low frequency)
    f_pha = [4, 8]  # Hz
    phase_signal = np.sin(2 * np.pi * 6 * t)
    
    # Amplitude signal (high frequency)
    f_amp = [60, 80]  # Hz
    carrier = np.sin(2 * np.pi * 70 * t)
    
    # Create PAC by modulating amplitude with phase
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 6 * t))
    amp_signal = modulation * carrier
    
    # Combined signal
    signal = phase_signal + amp_signal
    signal_batch = signal[np.newaxis, :]  # (1, n_times)
    
    print("Test Signal Statistics:")
    print(f"Signal mean: {signal.mean():.6f}")
    print(f"Signal std: {signal.std():.6f}")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    print()
    
    # 1. TensorPAC filtering and amplitude extraction
    print("=== TensorPAC Processing ===")
    # Filter for amplitude using TensorPAC's approach
    cycle = 6
    forder = fir_order(sf, n_times, f_amp[0], cycle=cycle)
    b, a = fir1(forder, np.array(f_amp) / (sf / 2.))
    
    print(f"Filter order: {forder}")
    print(f"Filter coefficients b shape: {b.shape}")
    print(f"Filter coefficient a: {a}")
    
    # Apply filter
    filtered_tensorpac = filtfilt(b, a, signal, padlen=forder, axis=-1)
    
    print(f"Filtered signal mean: {filtered_tensorpac.mean():.6f}")
    print(f"Filtered signal std: {filtered_tensorpac.std():.6f}")
    print(f"Filtered signal range: [{filtered_tensorpac.min():.3f}, {filtered_tensorpac.max():.3f}]")
    
    # Apply Hilbert
    from scipy.signal import hilbert
    analytic_tensorpac = hilbert(filtered_tensorpac)
    amp_tensorpac = np.abs(analytic_tensorpac)
    
    print(f"TensorPAC amplitude mean: {amp_tensorpac.mean():.6f}")
    print(f"TensorPAC amplitude std: {amp_tensorpac.std():.6f}")
    print(f"TensorPAC amplitude range: [{amp_tensorpac.min():.3f}, {amp_tensorpac.max():.3f}]")
    print()
    
    # 2. gPAC filtering and amplitude extraction
    print("=== gPAC Processing ===")
    # Create filter
    gpac_filter = BandPassFilter(
        f_band=f_amp,
        fs=sf,
        cycle=cycle,
        tensorpac_compat=True  # Use TensorPAC compatible mode
    )
    
    signal_torch = torch.tensor(signal_batch, dtype=torch.float32)
    filtered_gpac = gpac_filter(signal_torch)
    
    print(f"Filtered signal mean: {filtered_gpac.mean().item():.6f}")
    print(f"Filtered signal std: {filtered_gpac.std().item():.6f}")
    print(f"Filtered signal range: [{filtered_gpac.min().item():.3f}, {filtered_gpac.max().item():.3f}]")
    
    # Apply Hilbert
    gpac_hilbert = Hilbert(seq_len=n_times)
    hilbert_output = gpac_hilbert(filtered_gpac)
    amp_gpac = hilbert_output[0, :, 1].numpy()
    
    print(f"gPAC amplitude mean: {amp_gpac.mean():.6f}")
    print(f"gPAC amplitude std: {amp_gpac.std():.6f}")
    print(f"gPAC amplitude range: [{amp_gpac.min():.3f}, {amp_gpac.max():.3f}]")
    print()
    
    # Compare
    print("=== Comparison ===")
    print(f"Filtered signal ratio (gPAC/TensorPAC): {filtered_gpac.std().item() / filtered_tensorpac.std():.6f}")
    print(f"Amplitude ratio (gPAC/TensorPAC): {amp_gpac.mean() / amp_tensorpac.mean():.6f}")
    print()
    
    # Check filter coefficients
    print("=== Filter Coefficient Check ===")
    print(f"TensorPAC b sum: {b.sum():.6f}")
    print(f"TensorPAC b norm: {np.linalg.norm(b):.6f}")
    
    # Also test with Pac class directly
    print("\n=== Direct TensorPAC Pac class test ===")
    pac_obj = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert')
    amp_pac_class = pac_obj.filter(sf, signal_batch, ftype='amplitude', n_jobs=1)
    print(f"Pac class amplitude shape: {amp_pac_class.shape}")
    print(f"Pac class amplitude mean: {amp_pac_class.mean():.6f}")
    print(f"Pac class amplitude std: {amp_pac_class.std():.6f}")
    
    return {
        'filtered_tensorpac': filtered_tensorpac,
        'filtered_gpac': filtered_gpac.numpy(),
        'amp_tensorpac': amp_tensorpac,
        'amp_gpac': amp_gpac,
        'amp_pac_class': amp_pac_class
    }


if __name__ == "__main__":
    results = test_filter_amplitude_comparison()