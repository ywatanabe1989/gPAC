#!/usr/bin/env python3
"""Test with correct parameters matching v01 working version."""

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

def test_with_matching_params():
    """Test with parameters matching the working v01 implementation."""
    
    # Use the same parameters as in the working version
    fs = 512.0  # Changed from 1000 to 512
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Generate test signal with PAC
    phase_freq = 10.0
    amp_freq = 80.0
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    
    # Create amplitude modulation
    modulation_depth = 0.5
    amplitude_envelope = 1 + modulation_depth * np.sin(2 * np.pi * phase_freq * t)
    amplitude_signal = amplitude_envelope * np.sin(2 * np.pi * amp_freq * t)
    
    # Combine signals
    signal = phase_signal + 0.5 * amplitude_signal
    signal = signal / np.std(signal)  # Normalize
    
    print("="*60)
    print("PAC COMPARISON WITH CORRECT PARAMETERS")
    print("="*60)
    print(f"Sampling rate: {fs} Hz (matching v01 working version)")
    print(f"Signal: {phase_freq} Hz phase modulating {amp_freq} Hz amplitude")
    print(f"Duration: {duration}s, {n_samples} samples")
    
    # Parameters matching the scripts
    pha_freqs = (2, 20)
    amp_freqs = (60, 160)
    pha_n_bands = 50  # hres
    amp_n_bands = 30  # mres
    
    # Test gPAC with default settings
    print("\n1. Testing gPAC:")
    
    # Convert to tensor
    signal_batch = torch.tensor(signal.reshape(1, 1, -1), dtype=torch.float32)
    
    # Run gPAC (uses filtfilt_mode=False by default in current implementation)
    pac_values, pha_freqs_gpac, amp_freqs_gpac = gpac.calculate_pac(
        signal=signal_batch,
        fs=fs,
        pha_start_hz=pha_freqs[0],
        pha_end_hz=pha_freqs[1],
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_freqs[0],
        amp_end_hz=amp_freqs[1],
        amp_n_bands=amp_n_bands,
        n_perm=None,
        filter_cycle=3  # Default for phase
    )
    
    print(f"  PAC shape: {pac_values.shape}")
    print(f"  PAC range: [{pac_values.min():.6f}, {pac_values.max():.6f}]")
    
    # Find peak
    pac_numpy = pac_values[0, 0].cpu().numpy()
    max_idx = np.unravel_index(pac_numpy.argmax(), pac_numpy.shape)
    peak_pha = pha_freqs_gpac[max_idx[0]]
    peak_amp = amp_freqs_gpac[max_idx[1]]
    print(f"  Peak at: {peak_pha:.1f} Hz (phase) - {peak_amp:.1f} Hz (amplitude)")
    print(f"  Peak value: {pac_numpy.max():.6f}")
    
    # Test TensorPAC if available
    if TENSORPAC_AVAILABLE:
        print("\n2. Testing TensorPAC:")
        
        # Create frequency vectors matching gPAC
        pha_vec = np.linspace(pha_freqs[0], pha_freqs[1], pha_n_bands + 1)
        amp_vec = np.linspace(amp_freqs[0], amp_freqs[1], amp_n_bands + 1)
        
        # Convert to band format
        pha_bands = [[pha_vec[i], pha_vec[i+1]] for i in range(pha_n_bands)]
        amp_bands = [[amp_vec[i], amp_vec[i+1]] for i in range(amp_n_bands)]
        
        # Initialize TensorPAC
        pac_obj = Pac(
            idpac=(2, 0, 0),  # Modulation Index
            f_pha=pha_bands,
            f_amp=amp_bands,
            dcomplex='hilbert',
            cycle=(3, 6),  # cycle_pha=3, cycle_amp=6
            width=12,  # As per TensorPAC defaults
        )
        
        # Calculate PAC
        signal_2d = signal.reshape(1, -1)
        tensorpac_values = pac_obj.filterfit(fs, signal_2d)
        
        print(f"  PAC shape: {tensorpac_values.shape}")
        print(f"  PAC range: [{tensorpac_values.min():.6f}, {tensorpac_values.max():.6f}]")
        
        # Find peak
        max_idx_tp = np.unravel_index(tensorpac_values.argmax(), tensorpac_values.shape)
        pha_mids = np.array([np.mean(band) for band in pha_bands])
        amp_mids = np.array([np.mean(band) for band in amp_bands])
        peak_pha_tp = pha_mids[max_idx_tp[1]]
        peak_amp_tp = amp_mids[max_idx_tp[2]]
        print(f"  Peak at: {peak_pha_tp:.1f} Hz (phase) - {peak_amp_tp:.1f} Hz (amplitude)")
        print(f"  Peak value: {tensorpac_values.max():.6f}")
        
        # Compare - handle shape difference
        # TensorPAC shape: (n_amp, n_pha, n_trials)
        # gPAC shape: (n_pha, n_amp)
        tensorpac_vals = tensorpac_values[:, :, 0].T  # Transpose to match gPAC
        print(f"  Reshaped TensorPAC: {tensorpac_vals.shape}")
        print(f"  gPAC shape: {pac_numpy.shape}")
        
        # Ensure same shape
        if tensorpac_vals.shape != pac_numpy.shape:
            print(f"  ⚠️  Shape mismatch! Cannot compare directly.")
            correlation = 0.0
        else:
            correlation = np.corrcoef(pac_numpy.flatten(), tensorpac_vals.flatten())[0, 1]
        
        print(f"\n3. Correlation: {correlation:.4f}")
        
        # Detailed comparison
        print(f"\n4. Detailed comparison:")
        print(f"  Shape match: {pac_numpy.shape == tensorpac_vals.shape}")
        print(f"  Max value ratio: {pac_numpy.max() / tensorpac_values.max():.4f}")
        print(f"  Mean value ratio: {pac_numpy.mean() / tensorpac_values.mean():.4f}")
        
        if correlation < 0.5:
            print(f"\n⚠️  Low correlation detected!")
            print(f"  This suggests a fundamental difference in implementation")
            print(f"  Check filter parameters and MI calculation")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_with_matching_params()