#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 18:27:19 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_filter/_StaticBandpassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/_filter/_StaticBandpassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SincNet-style static bandpass filter with fixed frequency boundaries.
Follows the same sinc-based approach as DifferentiableBandPassFilter but without learnable parameters.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ._BaseFilter1D import BaseFilter1D


class StaticBandPassFilter(BaseFilter1D):
    """
    Static bandpass filter using SincNet-style sinc functions.

    This implementation uses the same sinc-based filtering as the
    differentiable version but with fixed (non-learnable) frequency bands.
    """

    def __init__(
        self,
        bands,
        fs,
        seq_len,
        filter_length=251,  # Should be odd
        window="hamming",  # 'hamming' or 'rectangular'
        fp16=False,
        padding_mode="zeros",  # Added for compatibility
    ):
        super().__init__(fp16=fp16)

        # Ensure filter length is odd
        if filter_length % 2 == 0:
            filter_length += 1

        self.fp16 = fp16
        self.fs = fs
        self.filter_length = filter_length
        self.window_type = window
        self.padding_mode = padding_mode

        # Ensure bands shape and validate
        assert bands.ndim == 2, "Bands should be 2D tensor with shape (n_bands, 2)"
        assert bands.shape[1] == 2, "Each band should have [low_hz, high_hz]"

        # Check bands definitions
        nyq = fs / 2.0
        bands = torch.clip(bands, 0.1, nyq - 1)
        for low, high in bands:
            assert (
                0 < low < high < nyq
            ), f"Invalid band: [{low}, {high}] Hz (Nyquist: {nyq} Hz)"

        # Store bands
        self.bands = bands
        self.n_filters = len(bands)

        # Create window function
        self.register_buffer("window", self._get_window(filter_length, window))

        # Prepare kernels
        kernels = self.init_kernels(seq_len, fs, bands)
        if fp16:
            kernels = kernels.half()
        self.register_buffer("kernels", kernels)

    @staticmethod
    def _get_window(length, window_type):
        """Get window function."""
        if window_type == "hamming":
            window = torch.hamming_window(length)
        elif window_type == "rectangular":
            window = torch.ones(length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        return window

    def _create_filters_vectorized(self, bands, fs, filter_length):
        """Create all filters at once - no loops, no scipy!"""
        n_filters = len(bands)

        # Time axis
        n = torch.arange(filter_length, dtype=torch.float32) - (filter_length - 1) / 2
        n = n.unsqueeze(0)  # (1, filter_length)

        # Frequencies
        low_hz = bands[:, 0:1]  # (n_filters, 1)
        high_hz = bands[:, 1:2]  # (n_filters, 1)

        # Normalized frequencies
        low_freq = low_hz * 2 / fs
        high_freq = high_hz * 2 / fs

        # Vectorized sinc - handle n=0 case
        eps = 1e-8
        n_safe = torch.where(torch.abs(n) < eps, eps, n)

        # Compute sinc for all filters at once
        low_arg = np.pi * low_freq * n
        high_arg = np.pi * high_freq * n

        low_sinc = torch.sin(low_arg) / (np.pi * n_safe)
        high_sinc = torch.sin(high_arg) / (np.pi * n_safe)

        # Handle n=0
        low_sinc = torch.where(torch.abs(n) < eps, low_freq, low_sinc)
        high_sinc = torch.where(torch.abs(n) < eps, high_freq, high_sinc)

        # Scale by frequency
        n_low = low_freq * low_sinc
        n_high = high_freq * high_sinc

        # Bandpass = highpass - lowpass
        band_pass = (n_high - n_low) * self.window.unsqueeze(0)

        # Normalize
        energy = torch.sqrt(torch.sum(band_pass**2, dim=1, keepdim=True))
        band_pass = band_pass / (energy + eps)

        return band_pass

    def init_kernels(self, seq_len, fs, bands):
        """Initialize sinc-based kernels for all bands - vectorized."""
        # Use vectorized implementation
        kernels = self._create_filters_vectorized(bands, fs, self.filter_length)

        # Ensure all filters have the same length
        kernels = self.ensure_even_len(kernels)

        return kernels

    @staticmethod
    def ensure_even_len(x):
        """Ensure the filter length is even for symmetric padding."""
        if x.shape[-1] % 2 == 0:
            return x
        else:
            return x[..., :-1]

    def get_filter_info(self):
        """Get information about the filter banks."""
        info = {
            "n_filters": self.n_filters,
            "filter_length": self.filter_length,
            "fs": self.fs,
            "window_type": self.window_type,
            "bands": self.bands.cpu().numpy()
            if hasattr(self.bands, "cpu")
            else self.bands,
        }
        return info


# EOF
