#!/usr/bin/env python3
"""Test TensorPAC filter behavior to understand the 22x difference."""

import sys
import os
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
tensorpac_path = os.path.join(os.path.dirname(__file__), '../tensorpac_source')
sys.path.insert(0, tensorpac_path)

try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("TensorPAC not available")

def test_filter_behavior():
    """Test what TensorPAC's filter actually returns."""
    
    if not TENSORPAC_AVAILABLE:
        print("TensorPAC not available, skipping test")
        return
    
    # Create test signal
    fs = 512
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Pure 10 Hz sine wave
    signal = np.sin(2 * np.pi * 10 * t)
    signal_2d = signal.reshape(1, -1)
    
    print("="*60)
    print("TENSORPAC FILTER BEHAVIOR TEST")
    print("="*60)
    print(f"Test signal: 10 Hz sine wave")
    print(f"Expected amplitude: 1.0")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Create PAC object
    pac = Pac(idpac=(2, 0, 0), f_pha=[[8, 12]], f_amp=[[8, 12]])
    
    # Test 1: Filter for phase
    print("\n1. Filter with 'phase' mode:")
    phase_output = pac.filter(fs, signal_2d, 'phase')
    print(f"  Output shape: {phase_output.shape}")
    print(f"  Output range: [{phase_output.min():.3f}, {phase_output.max():.3f}]")
    print(f"  Is this phase angles? {phase_output.min() < -3 and phase_output.max() > 3}")
    
    # Test 2: Filter for amplitude  
    print("\n2. Filter with 'amplitude' mode:")
    amp_output = pac.filter(fs, signal_2d, 'amplitude')
    print(f"  Output shape: {amp_output.shape}")
    print(f"  Output range: [{amp_output.min():.3f}, {amp_output.max():.3f}]")
    print(f"  Mean amplitude: {amp_output.mean():.6f}")
    print(f"  Is this envelope? {amp_output.min() >= 0}")
    
    # Test 3: Compare with manual filtering + Hilbert
    print("\n3. Manual filtering + Hilbert (gPAC style):")
    
    # Import gPAC modules
    from gpac._tensorpac_fir1 import design_filter_tensorpac
    from scipy import signal as scipy_signal
    
    # Design same filter
    filter_coef = design_filter_tensorpac(n_samples, fs, low_hz=8, high_hz=12, cycle=3)
    
    # Apply filter
    filtered = scipy_signal.filtfilt(filter_coef, 1, signal)
    
    # Apply Hilbert
    analytic = scipy_signal.hilbert(filtered)
    manual_phase = np.angle(analytic)
    manual_amp = np.abs(analytic)
    
    print(f"  Filtered signal range: [{filtered.min():.3f}, {filtered.max():.3f}]")
    print(f"  Manual phase range: [{manual_phase.min():.3f}, {manual_phase.max():.3f}]")
    print(f"  Manual amplitude mean: {manual_amp.mean():.6f}")
    
    # Compare
    print("\n4. Comparison:")
    print(f"  TensorPAC amp mean / Manual amp mean = {amp_output.mean() / manual_amp.mean():.6f}")
    
    # Test with modulated signal
    print("\n5. Test with PAC signal:")
    # Create signal with PAC
    carrier = np.sin(2 * np.pi * 80 * t)
    modulation = 1 + 0.8 * np.sin(2 * np.pi * 10 * t)
    pac_signal = np.sin(2 * np.pi * 10 * t) + modulation * carrier
    pac_signal_2d = pac_signal.reshape(1, -1)
    
    # Update PAC object for correct frequencies
    pac2 = Pac(idpac=(2, 0, 0), f_pha=[[8, 12]], f_amp=[[70, 90]])
    
    # Get phase and amplitude
    phase_pac = pac2.filter(fs, pac_signal_2d, 'phase')
    amp_pac = pac2.filter(fs, pac_signal_2d, 'amplitude')
    
    print(f"  Phase output shape: {phase_pac.shape}")
    print(f"  Amplitude output shape: {amp_pac.shape}")
    print(f"  Amplitude mean: {amp_pac.mean():.6f}")
    
    # Calculate PAC value
    pac_value = pac2.fit(phase_pac, amp_pac)
    print(f"  PAC value from separated: {pac_value[0,0,0]:.6f}")
    
    # Compare with direct filterfit
    pac_direct = pac2.filterfit(fs, pac_signal_2d)
    print(f"  PAC value from filterfit: {pac_direct[0,0,0]:.6f}")
    
    print("\n6. KEY FINDING:")
    print("  TensorPAC's filter() returns processed phase/amplitude, not just filtered signal")
    print("  This explains why the values are different!")
    
    return amp_output.mean(), manual_amp.mean()

if __name__ == "__main__":
    test_filter_behavior()