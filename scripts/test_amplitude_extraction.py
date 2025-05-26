#!/usr/bin/env python3
"""Test amplitude extraction differences between gPAC and TensorPAC."""

import sys
import os
import numpy as np
import torch
from scipy import signal

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def test_amplitude_extraction():
    """Compare amplitude extraction methods."""
    
    # Create a simple test signal
    fs = 512
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    
    # Signal with known amplitude modulation
    carrier_freq = 80.0  # Hz
    modulation_freq = 10.0  # Hz
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
    signal_np = envelope * carrier
    
    print("="*60)
    print("AMPLITUDE EXTRACTION COMPARISON")
    print("="*60)
    print(f"Signal: {carrier_freq} Hz carrier with {modulation_freq} Hz modulation")
    print(f"Expected envelope range: [0.5, 1.5]")
    print(f"Signal range: [{signal_np.min():.3f}, {signal_np.max():.3f}]")
    
    # Method 1: scipy Hilbert (reference)
    print("\n1. SciPy Hilbert Transform:")
    analytic_scipy = signal.hilbert(signal_np)
    amplitude_scipy = np.abs(analytic_scipy)
    phase_scipy = np.angle(analytic_scipy)
    
    print(f"  Amplitude mean: {amplitude_scipy.mean():.6f}")
    print(f"  Amplitude std: {amplitude_scipy.std():.6f}")
    print(f"  Amplitude range: [{amplitude_scipy.min():.6f}, {amplitude_scipy.max():.6f}]")
    
    # Method 2: gPAC Hilbert
    print("\n2. gPAC Hilbert Transform:")
    from gpac._Hilbert import Hilbert
    
    # Prepare tensor
    signal_tensor = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Initialize and apply Hilbert
    hilbert_module = Hilbert(seq_len=len(signal_np))
    analytic_gpac = hilbert_module(signal_tensor)
    
    # Extract amplitude and phase
    # gPAC returns (batch, channel, time, 2) where last dim is [phase, amplitude]
    phase_gpac = analytic_gpac[..., 0].squeeze().numpy()
    amplitude_gpac = analytic_gpac[..., 1].squeeze().numpy()
    
    print(f"  Amplitude mean: {amplitude_gpac.mean():.6f}")
    print(f"  Amplitude std: {amplitude_gpac.std():.6f}")
    print(f"  Amplitude range: [{amplitude_gpac.min():.6f}, {amplitude_gpac.max():.6f}]")
    
    # Compare
    print("\n3. Comparison:")
    amp_correlation = np.corrcoef(amplitude_scipy, amplitude_gpac)[0, 1]
    print(f"  Amplitude correlation: {amp_correlation:.6f}")
    print(f"  Amplitude ratio (gPAC/scipy): {amplitude_gpac.mean() / amplitude_scipy.mean():.6f}")
    print(f"  Max amplitude ratio: {amplitude_gpac.max() / amplitude_scipy.max():.6f}")
    
    # Check if there's a systematic scaling
    scaling_factor = amplitude_scipy.mean() / amplitude_gpac.mean()
    print(f"\n4. Scaling analysis:")
    print(f"  Mean scaling factor: {scaling_factor:.6f}")
    print(f"  Is this ~22x? {abs(scaling_factor - 22) < 5}")
    
    # Test with pure sine wave (no modulation)
    print("\n5. Test with pure sine wave:")
    pure_sine = np.sin(2 * np.pi * 50 * t)
    
    # scipy
    analytic_pure_scipy = signal.hilbert(pure_sine)
    amp_pure_scipy = np.abs(analytic_pure_scipy)
    
    # gPAC
    pure_tensor = torch.tensor(pure_sine, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    analytic_pure_gpac = hilbert_module(pure_tensor)
    amp_pure_gpac = analytic_pure_gpac[..., 1].squeeze().numpy()
    
    print(f"  SciPy amplitude: {amp_pure_scipy.mean():.6f}")
    print(f"  gPAC amplitude: {amp_pure_gpac.mean():.6f}")
    print(f"  Ratio: {amp_pure_gpac.mean() / amp_pure_scipy.mean():.6f}")
    
    return amplitude_scipy, amplitude_gpac

if __name__ == "__main__":
    test_amplitude_extraction()