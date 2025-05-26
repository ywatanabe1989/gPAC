#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 19:59:10 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/tests/gpac/test__tensorpac_fir1.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/test__tensorpac_fir1.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""
Tests for TensorPAC FIR filter implementation.
Tests mirror the source code structure and functionality.
"""

import pytest
import numpy as np
import torch

import sys
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
        assert not np.any(np.isnan(h))
        assert not np.any(np.isinf(h))

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
        assert not np.any(np.isnan(h))

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
        assert not np.any(np.isnan(h))
        assert not np.any(np.isinf(h))

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
            assert not np.any(np.isnan(h))

    def test_filter_symmetry(self):
        """Test that filters are symmetric."""
        sig_len = 2000
        fs = 1000
        low_hz = 10
        high_hz = 50
        cycle = 3

        h = design_filter_tensorpac(sig_len, fs, low_hz, high_hz, cycle)

        # Check symmetry
        mid = len(h) // 2
        if len(h) % 2 == 1:  # Odd length
            assert np.allclose(h[:mid], h[mid+1:][::-1])
        else:  # Even length
            assert np.allclose(h[:mid], h[mid:][::-1])

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

        # Both should have reasonable magnitude
        assert 0 < h_low.sum() < 2.0  # Roughly normalized
        assert -1.0 < h_band.sum() < 1.0

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
        if isinstance(h, torch.Tensor):
            h_np = h.numpy()
        else:
            h_np = h
        assert h_np.dtype in [np.float32, np.float64]

        # Test that it can be used for filtering
        test_signal = np.random.randn(sig_len)
        filtered = np.convolve(test_signal, h_np, mode='same')
        assert filtered.shape == test_signal.shape
        assert not np.any(np.isnan(filtered))

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_tensorpac_fir1.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-25 11:55:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/gPAC/src/gpac/_tensorpac_fir1.py
# # ----------------------------------------
# """
# TensorPAC-compatible FIR filter implementation for gPAC.
# 
# This module provides TensorPAC's custom fir1 implementation
# to ensure exact compatibility for fair comparisons.
# """
# 
# import numpy as np
# import torch
# from ._decorators import torch_fn
# 
# 
# def fir_order(fs, sizevec, flow, cycle=3):
#     """
#     Calculate filter order using TensorPAC's method.
#     
#     Parameters
#     ----------
#     fs : float
#         Sampling frequency
#     sizevec : int
#         Signal length
#     flow : float
#         Lower frequency bound
#     cycle : int
#         Number of cycles
#         
#     Returns
#     -------
#     int
#         Filter order
#     """
#     if cycle is None:
#         filtorder = 3 * np.fix(fs / flow)
#     else:
#         filtorder = cycle * (fs // flow)
#         
#         if (sizevec < 3 * filtorder):
#             filtorder = (sizevec - 1) // 3
#     
#     return int(filtorder)
# 
# 
# def n_odd_fcn(f, o, w, l):
#     """Odd case for TensorPAC fir1."""
#     # Variables :
#     b0 = 0
#     m = np.array(range(int(l + 1)))
#     k = m[1:len(m)]
#     b = np.zeros(k.shape)
# 
#     # Run Loop :
#     for s in range(0, len(f), 2):
#         m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
#         b1 = o[s] - m * f[s]
#         b0 = b0 + (b1 * (f[s + 1] - f[s]) + m / 2 * (
#             f[s + 1] * f[s + 1] - f[s] * f[s])) * abs(
#             np.square(w[round((s + 1) / 2)]))
#         b = b + (m / (4 * np.pi * np.pi) * (
#             np.cos(2 * np.pi * k * f[s + 1]) - np.cos(2 * np.pi * k * f[s])
#         ) / (k * k)) * abs(np.square(w[round((s + 1) / 2)]))
#         b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
#             s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
#             np.square(w[round((s + 1) / 2)]))
# 
#     b = np.insert(b, 0, b0)
#     a = (np.square(w[0])) * 4 * b
#     a[0] = a[0] / 2
#     aud = np.flipud(a[1:len(a)]) / 2
#     a2 = np.insert(aud, len(aud), a[0])
#     h = np.concatenate((a2, a[1:] / 2))
# 
#     return h
# 
# 
# def n_even_fcn(f, o, w, l):
#     """Even case for TensorPAC fir1."""
#     # Variables :
#     k = np.array(range(0, int(l) + 1, 1)) + 0.5
#     b = np.zeros(k.shape)
# 
#     # # Run Loop :
#     for s in range(0, len(f), 2):
#         m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
#         b1 = o[s] - m * f[s]
#         b = b + (m / (4 * np.pi * np.pi) * (np.cos(2 * np.pi * k * f[
#             s + 1]) - np.cos(2 * np.pi * k * f[s])) / (
#             k * k)) * abs(np.square(w[round((s + 1) / 2)]))
#         b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
#             s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
#             np.square(w[round((s + 1) / 2)]))
# 
#     a = (np.square(w[0])) * 4 * b
#     h = 0.5 * np.concatenate((np.flipud(a), a))
# 
#     return h
# 
# 
# def firls(n, f, o):
#     """TensorPAC's firls implementation."""
#     # Variables definition :
#     w = np.ones(round(len(f) / 2))
#     n += 1
#     f /= 2
#     lo = (n - 1) / 2
# 
#     nodd = bool(n % 2)
# 
#     if nodd:  # Odd case
#         h = n_odd_fcn(f, o, w, lo)
#     else:  # Even case
#         h = n_even_fcn(f, o, w, lo)
# 
#     return h
# 
# 
# def fir1(n, wn):
#     """
#     TensorPAC's fir1 implementation.
#     
#     Parameters
#     ----------
#     n : int
#         Filter order
#     wn : array_like
#         Normalized frequency boundaries [low, high] / (fs/2)
#         
#     Returns
#     -------
#     b : array_like
#         Filter coefficients
#     a : float
#         Always 1 for FIR filters
#     """
#     # Variables definition :
#     nbands = len(wn) + 1
#     ff = np.array((0, wn[0], wn[0], wn[1], wn[1], 1))
# 
#     f0 = np.mean(ff[2:4])
#     lo = n + 1
# 
#     mags = np.array(range(nbands)).reshape(1, -1) % 2
#     aa = np.ravel(np.tile(mags, (2, 1)), order='F')
# 
#     # Get filter coefficients :
#     h = firls(lo - 1, ff, aa)
# 
#     # Apply a window to coefficients :
#     wind = np.hamming(lo)
#     b = h * wind
#     c = np.exp(-1j * 2 * np.pi * (f0 / 2) * np.array(range(lo)))
#     b /= abs(c @ b)
# 
#     return b, 1
# 
# 
# @torch_fn
# def design_filter_tensorpac(sig_len, fs, low_hz=None, high_hz=None, cycle=3, is_bandstop=False):
#     """
#     Design a filter using TensorPAC's fir1 implementation.
#     
#     This is a drop-in replacement for gPAC's design_filter function
#     that uses TensorPAC's custom fir1 implementation instead of scipy.firwin.
#     
#     Parameters
#     ----------
#     sig_len : int
#         Signal length
#     fs : float
#         Sampling frequency
#     low_hz : float, optional
#         Lower frequency bound
#     high_hz : float, optional
#         Upper frequency bound
#     cycle : int
#         Number of cycles (default: 3)
#     is_bandstop : bool
#         Whether to create a bandstop filter (not implemented)
#         
#     Returns
#     -------
#     torch.Tensor
#         Filter coefficients
#     """
#     if is_bandstop:
#         raise NotImplementedError("Bandstop filters not implemented in TensorPAC mode")
#     
#     fs_f = float(fs)
#     if fs_f <= 0:
#         raise ValueError("fs must be positive.")
#     
#     nyq = fs_f / 2.0
#     low_hz_f = float(low_hz) if low_hz is not None else None
#     high_hz_f = float(high_hz) if high_hz is not None else None
#     
#     # Validate inputs
#     if low_hz_f is None or high_hz_f is None:
#         raise ValueError("Both low_hz and high_hz must be provided for bandpass filter")
#     
#     if not (0 < low_hz_f < nyq and 0 < high_hz_f < nyq):
#         raise ValueError(f"Frequencies must be > 0 and < Nyquist ({nyq}).")
#     
#     if low_hz_f >= high_hz_f:
#         raise ValueError(f"Low frequency {low_hz_f} must be < high {high_hz_f}.")
#     
#     # Calculate filter order using TensorPAC's method
#     filter_order = fir_order(fs_f, sig_len, low_hz_f, cycle=cycle)
#     
#     # Get filter coefficients using TensorPAC's fir1
#     wn = np.array([low_hz_f, high_hz_f]) / nyq
#     b_coeff, a_coeff = fir1(filter_order, wn)
#     
#     # Convert to torch tensor
#     h_np_contiguous = np.ascontiguousarray(b_coeff, dtype=np.float32)
#     return torch.from_numpy(h_np_contiguous)
# 
# 
# # Export the main function
# __all__ = ['design_filter_tensorpac', 'fir_order', 'fir1']
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_tensorpac_fir1.py
# --------------------------------------------------------------------------------
