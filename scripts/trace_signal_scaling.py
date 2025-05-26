#!/usr/bin/env python3
"""Trace signal scaling through gPAC vs TensorPAC pipelines."""

import sys
import os
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
tensorpac_path = os.path.join(os.path.dirname(__file__), '../tensorpac_source')
sys.path.insert(0, tensorpac_path)

def trace_signal_scaling():
    """Compare signal processing step by step."""
    
    # Parameters
    fs = 512
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Create test signal with known PAC
    phase_freq = 10.0
    amp_freq = 80.0
    modulation_depth = 0.8
    
    # Phase signal
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    
    # Amplitude signal with modulation
    carrier = np.sin(2 * np.pi * amp_freq * t)
    envelope = 1.0 + modulation_depth * np.sin(2 * np.pi * phase_freq * t)
    amp_signal = envelope * carrier
    
    # Combined signal
    signal = phase_signal + 0.5 * amp_signal
    
    print("="*60)
    print("SIGNAL SCALING TRACE")
    print("="*60)
    print(f"Test signal: {phase_freq} Hz phase, {amp_freq} Hz amplitude")
    print(f"Signal stats - mean: {signal.mean():.6f}, std: {signal.std():.6f}")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Test single frequency band filtering
    print("\n1. FILTER A SINGLE BAND (10 Hz):")
    
    # Import filter design
    from gpac._tensorpac_fir1 import design_filter_tensorpac
    
    # Design filter for phase band
    filter_coef = design_filter_tensorpac(
        n_samples, fs, low_hz=8.0, high_hz=12.0, cycle=3
    )
    print(f"  Filter length: {len(filter_coef)}")
    print(f"  Filter sum: {filter_coef.sum():.6f}")
    print(f"  Filter max: {filter_coef.max():.6f}")
    
    # Apply filter using conv1d (gPAC style)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.tensor(filter_coef, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    filtered_gpac = torch.nn.functional.conv1d(signal_tensor, kernel_tensor, padding='same')
    filtered_gpac_np = filtered_gpac.squeeze().numpy()
    
    print(f"\n  gPAC filtered signal:")
    print(f"    Mean: {filtered_gpac_np.mean():.6f}")
    print(f"    Std: {filtered_gpac_np.std():.6f}")
    print(f"    Range: [{filtered_gpac_np.min():.3f}, {filtered_gpac_np.max():.3f}]")
    
    # Apply filter using scipy (TensorPAC style)
    from scipy import signal as scipy_signal
    
    # TensorPAC uses filtfilt
    from tensorpac.spectral import spectral
    from tensorpac.methods.meth_pac import _kl_hr
    
    # Check if TensorPAC does any preprocessing
    print("\n2. CHECK TENSORPAC PREPROCESSING:")
    
    # Look for normalization in TensorPAC
    try:
        # TensorPAC's Pac class
        from tensorpac import Pac
        
        # Create simple PAC object
        pac_obj = Pac(idpac=(2, 0, 0), f_pha=[[8, 12]], f_amp=[[78, 82]])
        
        # Check the filter method
        print("  Checking TensorPAC's filter method...")
        
        # TensorPAC expects (n_epochs, n_times)
        signal_2d = signal.reshape(1, -1)
        
        # Get the xphase and xamp from TensorPAC
        # This requires looking at the filterfit method
        pac_obj._phast_meth = 'hilbert'
        pac_obj._cycle = (3, 6)
        
        # Filter for phase
        sf_pha = pac_obj.filter(fs, signal_2d, 'phase')
        print(f"  TensorPAC phase filtered shape: {sf_pha.shape}")
        print(f"  Phase filtered range: [{sf_pha.min():.3f}, {sf_pha.max():.3f}]")
        
        # Filter for amplitude  
        sf_amp = pac_obj.filter(fs, signal_2d, 'amplitude')
        print(f"  TensorPAC amp filtered shape: {sf_amp.shape}")
        print(f"  Amp filtered range: [{sf_amp.min():.3f}, {sf_amp.max():.3f}]")
        
    except Exception as e:
        print(f"  Error accessing TensorPAC internals: {e}")
    
    # The key insight
    print("\n3. KEY INSIGHT:")
    print("  The 22x difference suggests:")
    print("  - Different filter gain/normalization")
    print("  - Different amplitude extraction method")
    print("  - Possible sqrt(2) or pi factors")
    print(f"  - Check if 22 ≈ 2π² ≈ {2 * np.pi**2:.1f}")
    
    # Test amplitude extraction difference
    print("\n4. AMPLITUDE EXTRACTION TEST:")
    
    # Create analytic signal
    analytic_scipy = scipy_signal.hilbert(filtered_gpac_np)
    amp_scipy = np.abs(analytic_scipy)
    
    print(f"  Amplitude from Hilbert:")
    print(f"    Mean: {amp_scipy.mean():.6f}")
    print(f"    Max: {amp_scipy.max():.6f}")
    
    # Check if there's a systematic factor
    print(f"\n  Possible scaling factors:")
    print(f"    sqrt(2) = {np.sqrt(2):.6f}")
    print(f"    pi = {np.pi:.6f}")
    print(f"    2*pi = {2*np.pi:.6f}")
    print(f"    Filter length / 2 = {len(filter_coef)/2:.1f}")
    
    return filtered_gpac_np

if __name__ == "__main__":
    trace_signal_scaling()