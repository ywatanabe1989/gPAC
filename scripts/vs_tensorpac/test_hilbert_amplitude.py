#!/usr/bin/env python3
"""Test to compare Hilbert transform amplitude extraction between gPAC and TensorPAC."""

import numpy as np
import torch
from scipy.signal import hilbert
from scipy import fftpack

# Import gPAC Hilbert
import sys
sys.path.append('/home/ywatanabe/proj/gPAC/src')
from gpac._Hilbert import Hilbert


def tensorpac_hilbertm(x):
    """TensorPAC's Hilbert implementation."""
    n_pts = x.shape[-1]
    fc = fftpack.helper.next_fast_len(n_pts)
    return hilbert(x, fc, axis=-1)[..., 0:n_pts]


def test_hilbert_comparison():
    """Compare Hilbert transforms on a test signal."""
    # Create test signal
    np.random.seed(42)
    n_times = 1000
    t = np.linspace(0, 1, n_times)
    
    # Create a simple amplitude-modulated signal
    carrier_freq = 100  # Hz
    modulating_freq = 10  # Hz
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * modulating_freq * t))
    signal = modulation * carrier
    
    # Add batch dimension for gPAC
    signal_batch = signal[np.newaxis, :]  # (1, n_times)
    
    print("Test Signal Statistics:")
    print(f"Signal shape: {signal.shape}")
    print(f"Signal mean: {signal.mean():.6f}")
    print(f"Signal std: {signal.std():.6f}")
    print(f"Signal min/max: {signal.min():.6f} / {signal.max():.6f}")
    print()
    
    # 1. TensorPAC Hilbert (using scipy)
    analytic_tensorpac = tensorpac_hilbertm(signal)
    amp_tensorpac = np.abs(analytic_tensorpac)
    phase_tensorpac = np.angle(analytic_tensorpac)
    
    print("TensorPAC Hilbert Results:")
    print(f"Amplitude mean: {amp_tensorpac.mean():.6f}")
    print(f"Amplitude std: {amp_tensorpac.std():.6f}")
    print(f"Amplitude min/max: {amp_tensorpac.min():.6f} / {amp_tensorpac.max():.6f}")
    print()
    
    # 2. gPAC Hilbert
    gpac_hilbert = Hilbert(seq_len=n_times)
    signal_torch = torch.tensor(signal_batch, dtype=torch.float32)
    output_gpac = gpac_hilbert(signal_torch)
    phase_gpac = output_gpac[0, :, 0].numpy()
    amp_gpac = output_gpac[0, :, 1].numpy()
    
    print("gPAC Hilbert Results:")
    print(f"Amplitude mean: {amp_gpac.mean():.6f}")
    print(f"Amplitude std: {amp_gpac.std():.6f}")
    print(f"Amplitude min/max: {amp_gpac.min():.6f} / {amp_gpac.max():.6f}")
    print()
    
    # 3. Direct scipy hilbert for comparison
    analytic_scipy = hilbert(signal)
    amp_scipy = np.abs(analytic_scipy)
    
    print("Direct scipy.hilbert Results:")
    print(f"Amplitude mean: {amp_scipy.mean():.6f}")
    print(f"Amplitude std: {amp_scipy.std():.6f}")
    print(f"Amplitude min/max: {amp_scipy.min():.6f} / {amp_scipy.max():.6f}")
    print()
    
    # Compare amplitudes
    print("Amplitude Ratios:")
    print(f"gPAC/TensorPAC ratio: {amp_gpac.mean() / amp_tensorpac.mean():.6f}")
    print(f"gPAC/scipy ratio: {amp_gpac.mean() / amp_scipy.mean():.6f}")
    print(f"TensorPAC/scipy ratio: {amp_tensorpac.mean() / amp_scipy.mean():.6f}")
    print()
    
    # Check if gPAC is missing scaling
    print("Checking for missing 2x factor:")
    amp_gpac_scaled = amp_gpac * 2.0
    print(f"gPAC*2/TensorPAC ratio: {amp_gpac_scaled.mean() / amp_tensorpac.mean():.6f}")
    print()
    
    # Manual FFT-based Hilbert to debug
    print("Manual FFT-based Hilbert (matching gPAC implementation):")
    # Replicate gPAC's approach
    xf = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n_times)
    
    # Create Heaviside step
    heaviside_u = np.zeros_like(freqs)
    if n_times % 2 == 0:
        heaviside_u[1:n_times//2] = 1.0
        heaviside_u[n_times//2] = 0.5
    else:
        heaviside_u[1:(n_times-1)//2+1] = 1.0
    heaviside_u[0] = 0.5
    
    # Apply Heaviside (multiply by 2)
    analytic_f = xf * (2.0 * heaviside_u)
    analytic_t = np.fft.ifft(analytic_f)
    amp_manual = np.abs(analytic_t)
    
    print(f"Manual amplitude mean: {amp_manual.mean():.6f}")
    print(f"Manual/TensorPAC ratio: {amp_manual.mean() / amp_tensorpac.mean():.6f}")
    
    return {
        'tensorpac': amp_tensorpac,
        'gpac': amp_gpac,
        'scipy': amp_scipy,
        'manual': amp_manual
    }


if __name__ == "__main__":
    results = test_hilbert_comparison()