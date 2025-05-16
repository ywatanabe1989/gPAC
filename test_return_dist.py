#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test return_dist functionality

import torch
import sys
import os

# Add the project root directory to the path to ensure we can import gpac
sys.path.insert(0, os.path.abspath('.'))
import gpac

def test_return_dist():
    """Test that return_dist parameter works correctly."""
    # Create test signal (batch, channels, segments, time)
    signal = torch.randn(1, 1, 1, 1000)
    
    # Parameters
    fs = 1000
    n_perm = 10
    
    # Test without return_dist (default=False)
    result_no_dist, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=signal,
        fs=fs,
        pha_start_hz=5.0,
        pha_end_hz=15.0,
        pha_n_bands=3,
        amp_start_hz=80.0,
        amp_end_hz=120.0,
        amp_n_bands=2,
        n_perm=n_perm,
        return_dist=False,
    )
    
    print(f"Without return_dist: Shape of PAC result: {result_no_dist.shape}")
    
    # Test with return_dist=True
    result_with_dist, surrogate_dist, pha_freqs_2, amp_freqs_2 = gpac.calculate_pac(
        signal=signal,
        fs=fs,
        pha_start_hz=5.0,
        pha_end_hz=15.0,
        pha_n_bands=3,
        amp_start_hz=80.0,
        amp_end_hz=120.0,
        amp_n_bands=2,
        n_perm=n_perm,
        return_dist=True,
    )
    
    print(f"With return_dist: Shape of PAC result: {result_with_dist.shape}, Shape of surrogate distribution: {surrogate_dist.shape}")
    
    # Verify they have the same shape
    assert result_no_dist.shape == result_with_dist.shape
    
    # Verify surrogate_dist has expected shape (n_perm, batch, channels, pha_n_bands, amp_n_bands)
    expected_shape = (n_perm, 1, 1, 3, 2)
    assert surrogate_dist.shape == expected_shape
    
    print("All tests passed!")

if __name__ == "__main__":
    test_return_dist()