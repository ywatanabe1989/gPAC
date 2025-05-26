#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for TensorPAC FIR filter implementation.
Tests mirror the source code structure and functionality.
"""

import pytest
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from gpac._tensorpac_fir1 import fir_order, n_odd_fcn, n_even_fcn
except ImportError:
    # Alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.gpac._tensorpac_fir1 import fir_order, n_odd_fcn, n_even_fcn

# Import design_filter_tensorpac separately due to @torch_fn decorator
try:
    from gpac._tensorpac_fir1 import design_filter_tensorpac
except ImportError:
    from src.gpac._tensorpac_fir1 import design_filter_tensorpac


def check_nan_inf(h):
    """Helper to check for NaN/Inf in both numpy and torch arrays."""
    if isinstance(h, torch.Tensor):
        return not torch.isnan(h).any() and not torch.isinf(h).any()
    else:
        return not np.any(np.isnan(h)) and not np.any(np.isinf(h))


def to_numpy(h):
    """Convert to numpy array if needed."""
    if isinstance(h, torch.Tensor):
        return h.detach().cpu().numpy()
    return h


class TestFirOrder:
    """Test fir_order function."""
    
    def test_basic_filter_order(self):
        """Test basic filter order calculation."""
        fs = 1000  # Hz
        sizevec = 2000  # samples
        flow = 10  # Hz
        cycle = 3
        
        order = fir_order(fs, sizevec, flow, cycle)
        
        # Expected: cycle * (fs // flow) = 3 * (1000 // 10) = 300
        assert order == 300
        
    def test_filter_order_size_limit(self):
        """Test filter order when signal is too short."""
        fs = 1000
        sizevec = 500  # Short signal
        flow = 10
        cycle = 3
        
        order = fir_order(fs, sizevec, flow, cycle)
        
        # Should be limited by signal size: (500 - 1) // 3 = 166
        assert order == 166
        
    def test_filter_order_none_cycle(self):
        """Test filter order with None cycle."""
        fs = 1000
        sizevec = 2000
        flow = 10
        cycle = None
        
        order = fir_order(fs, sizevec, flow, cycle)
        
        # Expected: 3 * fix(fs / flow) = 3 * fix(100) = 300
        assert order == 300
        
    def test_filter_order_various_frequencies(self):
        """Test filter order with various frequency combinations."""
        test_cases = [
            (512, 1024, 4, 3, 384),    # Normal case
            (256, 512, 8, 3, 96),      # Lower fs
            (1000, 100, 10, 3, 33),    # Very short signal
            (2000, 4000, 50, 3, 120),  # High frequency
        ]
        
        for fs, sizevec, flow, cycle, expected in test_cases:
            order = fir_order(fs, sizevec, flow, cycle)
            assert order == expected


class TestNOddFcn:
    """Test n_odd_fcn for odd-length filters."""
    
    def test_basic_odd_filter(self):
        """Test basic odd-length filter design."""
        # Simple bandpass parameters
        f = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 1.0])
        o = np.array([0, 0, 1, 1, 0, 0])
        w = np.array([1.0, 1.0, 1.0])
        l = 10  # Half length
        
        h = n_odd_fcn(f, o, w, l)
        
        # Check output properties
        assert len(h) == 2 * l + 1  # Odd length
        assert h[l] == h.max()  # Center tap should be maximum
        assert np.allclose(h[:l], h[l+1:][::-1])  # Symmetric
        
    def test_lowpass_odd_filter(self):
        """Test lowpass filter design."""
        f = np.array([0.0, 0.2, 0.2, 1.0])
        o = np.array([1, 1, 0, 0])
        w = np.array([1.0, 1.0])
        l = 15
        
        h = n_odd_fcn(f, o, w, l)
        
        assert len(h) == 31
        assert check_nan_inf(h)
        
    def test_highpass_odd_filter(self):
        """Test highpass filter design."""
        f = np.array([0.0, 0.3, 0.3, 1.0])
        o = np.array([0, 0, 1, 1])
        w = np.array([1.0, 1.0])
        l = 20
        
        h = n_odd_fcn(f, o, w, l)
        
        assert len(h) == 41
        # Highpass filters have alternating signs
        assert h[20] != 0  # Center tap


class TestNEvenFcn:
    """Test n_even_fcn for even-length filters."""
    
    def test_basic_even_filter(self):
        """Test basic even-length filter design."""
        f = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 1.0])
        o = np.array([0, 0, 1, 1, 0, 0])
        w = np.array([1.0, 1.0, 1.0])
        l = 10
        
        h = n_even_fcn(f, o, w, l)
        
        # Check output properties
        assert len(h) == 2 * (l + 1)  # Even length
        assert np.allclose(h[:l+1], h[l+1:][::-1])  # Symmetric
        
    def test_bandstop_even_filter(self):
        """Test bandstop filter design."""
        f = np.array([0.0, 0.2, 0.2, 0.3, 0.3, 1.0])
        o = np.array([1, 1, 0, 0, 1, 1])
        w = np.array([1.0, 1.0, 1.0])
        l = 12
        
        h = n_even_fcn(f, o, w, l)
        
        assert len(h) == 26
        assert check_nan_inf(h)
        
    def test_multiband_even_filter(self):
        """Test multiband filter design."""
        f = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 1.0])
        o = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        l = 15
        
        h = n_even_fcn(f, o, w, l)
        
        assert len(h) == 32
        assert h.sum() != 0  # Should have some passband


class TestDesignFilterTensorpac:
    """Test the main filter design function."""
    
    def test_basic_bandpass_filter(self):
        """Test basic bandpass filter design."""
        sig_len = 2000
        fs = 1000
        low_hz = 10
        high_hz = 20
        cycle = 3
        
        h = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, cycle)
        
        # Check basic properties  
        assert isinstance(h, torch.Tensor) or isinstance(h, np.ndarray)
        assert len(h) > 0
        assert check_nan_inf(h)
        
    def test_various_filter_designs(self):
        """Test various filter configurations."""
        test_cases = [
            # (fs, sizevec, flow, fhigh, cycle)
            (1024, 512, 4, 8, 3),      # Narrow band
            (2000, 1000, 60, 80, 6),   # High frequency
            (512, 256, 2, 10, 3),      # Low frequency
            (4000, 2000, 100, 200, 6), # Very high frequency
        ]
        
        for sig_len, fs, low_hz, high_hz, cycle in test_cases:
            h = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, cycle)
            
            # All filters should be valid
            assert isinstance(h, torch.Tensor) or isinstance(h, np.ndarray)
            assert len(h) > 0
            assert check_nan_inf(h)
            
    def test_filter_symmetry(self):
        """Test that filters are symmetric."""
        sig_len = 2000
        fs = 1000
        low_hz = 10
        high_hz = 50
        cycle = 3
        
        h = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, cycle)
        h_np = to_numpy(h)
        
        # Check symmetry
        mid = len(h_np) // 2
        if len(h_np) % 2 == 1:  # Odd length
            assert np.allclose(h_np[:mid], h_np[mid+1:][::-1])
        else:  # Even length  
            assert np.allclose(h_np[:mid], h_np[mid:][::-1])
            
    def test_filter_edge_cases(self):
        """Test edge cases in filter design."""
        # Very short signal
        h1 = design_filter_tensorpac(100, 1000, 10, 20, 3)
        assert len(h1) < 100
        
        # High frequency relative to fs
        h2 = design_filter_tensorpac(512, 256, 100, 120, 3)
        assert len(h2) > 0
        
        # Very narrow band
        h3 = design_filter_tensorpac(2000, 1000, 10, 11, 3)
        assert len(h3) > 0
        
    def test_filter_normalization(self):
        """Test filter normalization."""
        sig_len = 2000
        fs = 1000
        
        # Lowpass filter (0-50 Hz)
        h_low = design_filter_tensorpac(sig_len, fs, None, 50, 3)
        
        # Bandpass filter (10-50 Hz)  
        h_band = design_filter_tensorpac(sig_len, fs, 10, 50, 3)
        
        h_low_np = to_numpy(h_low)
        h_band_np = to_numpy(h_band)
        
        # Both should have reasonable magnitude
        assert 0 < h_low_np.sum() < 2.0  # Roughly normalized
        assert -1.0 < h_band_np.sum() < 1.0
        
    def test_cycle_parameter_effect(self):
        """Test effect of cycle parameter on filter length."""
        sig_len = 4000
        fs = 1000
        low_hz = 10
        high_hz = 20
        
        # Different cycle values
        h3 = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, 3)
        h6 = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, 6)
        h9 = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, 9)
        
        # Higher cycles should give longer filters
        assert len(h3) < len(h6) < len(h9)
        
    def test_compatibility_with_scipy(self):
        """Test that output format is compatible with scipy filters."""
        sig_len = 2000
        fs = 1000
        low_hz = 10
        high_hz = 50
        
        h = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, 3)
        
        # Should be 1D array suitable for convolution
        assert h.ndim == 1
        h_np = to_numpy(h)
        assert h_np.dtype in [np.float32, np.float64]
        
        # Test that it can be used for filtering
        test_signal = np.random.randn(sig_len)
        filtered = np.convolve(test_signal, h_np, mode='same')
        assert filtered.shape == test_signal.shape
        assert not np.any(np.isnan(filtered))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])