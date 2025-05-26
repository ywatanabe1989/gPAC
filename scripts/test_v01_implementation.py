#!/usr/bin/env python3
"""Test v01 implementation against current to check correlation improvements."""

import sys
import numpy as np
import torch

# Add paths
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src/gpac/v01')

# Test if we can import the v01 module
try:
    from _CombinedBandPassFilter_v01_working import CombinedBandPassFilter as CombinedBandPassFilter_v01
    print("✅ Successfully imported v01 CombinedBandPassFilter")
    
    # Generate test signal
    fs = 512
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Define frequency bands
    pha_bands = torch.tensor([[4., 8.], [8., 12.], [12., 16.]])
    amp_bands = torch.tensor([[60., 80.], [80., 100.]])
    
    # Test v01 filter
    filter_v01 = CombinedBandPassFilter_v01(
        pha_bands=pha_bands,
        amp_bands=amp_bands, 
        fs=fs,
        seq_len=len(signal),
        filtfilt_mode=True,
        edge_mode='reflect'
    )
    
    output_v01 = filter_v01(signal_tensor)
    print(f"v01 output shape: {output_v01.shape}")
    print(f"v01 output stats - min: {output_v01.min():.6f}, max: {output_v01.max():.6f}, mean: {output_v01.mean():.6f}")
    
except Exception as e:
    print(f"❌ Failed to test v01 implementation: {e}")
    import traceback
    traceback.print_exc()

# Now test current implementation
try:
    from gpac._BandPassFilter import BandPassFilter
    print("\n✅ Successfully imported current BandPassFilter")
    
    # Test current filter with phase bands
    filter_current = BandPassFilter(
        bands=pha_bands[:1],  # Just test first band
        fs=fs,
        seq_len=len(signal),
        filtfilt_mode=True,
        edge_mode='reflect'
    )
    
    output_current = filter_current(signal_tensor)
    print(f"Current output shape: {output_current.shape}")
    print(f"Current output stats - min: {output_current.min():.6f}, max: {output_current.max():.6f}, mean: {output_current.mean():.6f}")
    
except Exception as e:
    print(f"❌ Failed to test current implementation: {e}")
    import traceback
    traceback.print_exc()