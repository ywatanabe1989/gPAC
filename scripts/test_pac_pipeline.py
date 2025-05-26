#!/usr/bin/env python3
"""Test the full PAC pipeline to find where the 22x scaling occurs."""

import sys
import os
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def test_pac_pipeline():
    """Trace through PAC pipeline to find scaling issue."""
    
    # Parameters matching the comparison
    fs = 512
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Create PAC signal
    phase_freq = 10.0
    amp_freq = 80.0
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    amplitude_envelope = 1 + 0.5 * np.sin(2 * np.pi * phase_freq * t)
    amplitude_signal = amplitude_envelope * np.sin(2 * np.pi * amp_freq * t)
    signal = phase_signal + 0.5 * amplitude_signal
    signal = signal / np.std(signal)
    
    print("="*60)
    print("PAC PIPELINE TRACING")
    print("="*60)
    
    # Import modules
    from gpac._BandPassFilter import BandPassFilter
    from gpac._Hilbert import Hilbert
    from gpac._ModulationIndex import ModulationIndex
    
    # Prepare signal
    signal_tensor = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, -1)
    print(f"Input signal shape: {signal_tensor.shape}")
    print(f"Input signal range: [{signal_tensor.min():.3f}, {signal_tensor.max():.3f}]")
    
    # Step 1: Bandpass filtering
    print("\n1. BANDPASS FILTERING:")
    
    # Single band around our frequencies
    pha_bands = torch.tensor([[8., 12.]])  # Around 10 Hz
    amp_bands = torch.tensor([[70., 90.]])  # Around 80 Hz
    
    # Create filter
    bp_filter = BandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=n_samples,
        filtfilt_mode=False  # Simple filtering first
    )
    
    # Apply filter
    filtered = bp_filter(signal_tensor)
    print(f"  Filtered shape: {filtered.shape}")
    print(f"  Phase band output range: [{filtered[0,0,0].min():.6f}, {filtered[0,0,0].max():.6f}]")
    print(f"  Amp band output range: [{filtered[0,0,1].min():.6f}, {filtered[0,0,1].max():.6f}]")
    
    # Step 2: Hilbert transform
    print("\n2. HILBERT TRANSFORM:")
    hilbert = Hilbert(seq_len=n_samples)
    
    # Reshape for Hilbert (expects 4D or 5D)
    filtered_4d = filtered.permute(0, 1, 2, 3)  # Already correct shape
    analytic = hilbert(filtered_4d)
    
    print(f"  Analytic shape: {analytic.shape}")
    phase_data = analytic[..., 0]
    amp_data = analytic[..., 1]
    
    print(f"  Phase band amplitude: [{amp_data[0,0,0].min():.6f}, {amp_data[0,0,0].max():.6f}]")
    print(f"  Amp band amplitude: [{amp_data[0,0,1].min():.6f}, {amp_data[0,0,1].max():.6f}]")
    
    # Step 3: Extract phase and amplitude for MI
    print("\n3. PHASE/AMPLITUDE EXTRACTION:")
    # Get phase from phase band
    phase = phase_data[0, 0, 0]  # First (and only) phase band
    # Get amplitude from amplitude band  
    amplitude = amp_data[0, 0, 1]  # Second band (amplitude)
    
    print(f"  Phase range: [{phase.min():.3f}, {phase.max():.3f}]")
    print(f"  Amplitude range: [{amplitude.min():.6f}, {amplitude.max():.6f}]")
    print(f"  Amplitude mean: {amplitude.mean():.6f}")
    
    # Step 4: MI calculation
    print("\n4. MODULATION INDEX:")
    mi_calc = ModulationIndex(n_bins=18)
    
    # Prepare for MI (expects specific shape)
    phase_5d = phase.reshape(1, 1, 1, 1, -1)
    amp_5d = amplitude.reshape(1, 1, 1, 1, -1)
    
    result = mi_calc(phase_5d, amp_5d)
    mi_value = result['mi'].item()
    
    print(f"  MI value: {mi_value:.6f}")
    print(f"  Expected range: [0, 2] where 2=no coupling")
    
    # Check amplitude probability distribution
    amp_prob = result['amp_prob'].squeeze()
    print(f"\n5. AMPLITUDE DISTRIBUTION:")
    print(f"  Amp prob shape: {amp_prob.shape}")
    print(f"  Amp prob sum: {amp_prob.sum():.6f}")
    print(f"  Amp prob range: [{amp_prob.min():.6f}, {amp_prob.max():.6f}]")
    
    # The issue might be in the amplitude values going into MI
    print(f"\n6. DIAGNOSIS:")
    print(f"  If amplitude values are ~22x smaller than expected,")
    print(f"  check the filter gain or normalization")
    
    return mi_value

if __name__ == "__main__":
    test_pac_pipeline()