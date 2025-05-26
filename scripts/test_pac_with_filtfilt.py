#!/usr/bin/env python3
"""Test PAC calculation with explicit filtfilt mode to improve TensorPAC correlation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import torch
import gpac

# Try to import TensorPAC
try:
    tensorpac_path = os.path.join(os.path.dirname(__file__), '../tensorpac_source')
    sys.path.insert(0, tensorpac_path)
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("TensorPAC not available")

def test_pac_correlation():
    """Test PAC with explicit filtfilt settings."""
    
    # Generate simple test signal
    fs = 1000.0
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Create PAC signal: 10 Hz phase modulating 80 Hz amplitude
    phase_signal = np.sin(2 * np.pi * 10 * t)
    amplitude_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * t)
    amplitude_signal = amplitude_envelope * np.sin(2 * np.pi * 80 * t)
    signal = phase_signal + amplitude_signal
    
    # Add to batch format
    signal_batch = signal.reshape(1, 1, -1)  # (batch=1, channel=1, time)
    
    print("="*60)
    print("PAC CORRELATION TEST WITH FILTFILT")
    print("="*60)
    print(f"Signal: 10 Hz phase modulating 80 Hz amplitude")
    print(f"Duration: {duration}s, Sampling rate: {fs} Hz")
    
    # Test 1: gPAC with explicit filtfilt
    print("\n1. Testing gPAC with filtfilt_mode=True:")
    pac_values_filtfilt, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=torch.tensor(signal_batch, dtype=torch.float32),
        fs=fs,
        pha_start_hz=4.0,
        pha_end_hz=16.0,
        pha_n_bands=6,
        amp_start_hz=60.0,
        amp_end_hz=100.0,
        amp_n_bands=6,
        filtfilt_mode=True,  # Explicitly enable filtfilt
        edge_mode='reflect',  # Match scipy.signal.filtfilt
        n_perm=None
    )
    
    print(f"  PAC shape: {pac_values_filtfilt.shape}")
    print(f"  PAC range: [{pac_values_filtfilt.min():.4f}, {pac_values_filtfilt.max():.4f}]")
    print(f"  Max PAC value: {pac_values_filtfilt.max():.4f}")
    
    # Find peak
    max_idx = np.unravel_index(pac_values_filtfilt.argmax(), pac_values_filtfilt.shape)
    peak_pha = pha_freqs[max_idx[-2]]
    peak_amp = amp_freqs[max_idx[-1]]
    print(f"  Peak at: {peak_pha:.1f} Hz (phase) - {peak_amp:.1f} Hz (amplitude)")
    
    # Test 2: gPAC without filtfilt for comparison
    print("\n2. Testing gPAC with filtfilt_mode=False:")
    pac_values_no_filtfilt, _, _ = gpac.calculate_pac(
        signal=torch.tensor(signal_batch, dtype=torch.float32),
        fs=fs,
        pha_start_hz=4.0,
        pha_end_hz=16.0,
        pha_n_bands=6,
        amp_start_hz=60.0,
        amp_end_hz=100.0,
        amp_n_bands=6,
        filtfilt_mode=False,
        n_perm=None
    )
    
    print(f"  PAC range: [{pac_values_no_filtfilt.min():.4f}, {pac_values_no_filtfilt.max():.4f}]")
    print(f"  Max PAC value: {pac_values_no_filtfilt.max():.4f}")
    
    # Compare
    corr = np.corrcoef(pac_values_filtfilt.flatten(), pac_values_no_filtfilt.flatten())[0, 1]
    print(f"\n3. Filtfilt vs No-filtfilt correlation: {corr:.4f}")
    
    # Test 3: TensorPAC if available
    if TENSORPAC_AVAILABLE:
        print("\n4. Testing TensorPAC:")
        
        # Create frequency bands
        pha_bands = [[4 + i*2, 4 + (i+1)*2] for i in range(6)]
        amp_bands = [[60 + i*40/6, 60 + (i+1)*40/6] for i in range(6)]
        
        # Initialize TensorPAC
        pac_obj = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, dcomplex='hilbert')
        
        # Calculate PAC (TensorPAC expects n_epochs x n_times)
        signal_2d = signal.reshape(1, -1)
        tensorpac_values = pac_obj.filterfit(fs, signal_2d)
        
        print(f"  PAC shape: {tensorpac_values.shape}")
        print(f"  PAC range: [{tensorpac_values.min():.4f}, {tensorpac_values.max():.4f}]")
        print(f"  Max PAC value: {tensorpac_values.max():.4f}")
        
        # Compare with gPAC
        gpac_vals = pac_values_filtfilt[0, 0].numpy()  # Remove batch and channel dims
        corr_filtfilt = np.corrcoef(gpac_vals.flatten(), tensorpac_values.flatten())[0, 1]
        
        gpac_vals_no = pac_values_no_filtfilt[0, 0].numpy()
        corr_no_filtfilt = np.corrcoef(gpac_vals_no.flatten(), tensorpac_values.flatten())[0, 1]
        
        print(f"\n5. Correlation with TensorPAC:")
        print(f"  gPAC (filtfilt=True):  {corr_filtfilt:.4f}")
        print(f"  gPAC (filtfilt=False): {corr_no_filtfilt:.4f}")
        
        if corr_filtfilt > corr_no_filtfilt:
            print(f"\n✅ filtfilt=True improves correlation by {corr_filtfilt - corr_no_filtfilt:.4f}")
        else:
            print(f"\n❌ filtfilt=False is better by {corr_no_filtfilt - corr_filtfilt:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_pac_correlation()