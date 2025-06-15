#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 13:29:27 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/core/_BandPassFilters/_StaticBandPassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/core/_BandPassFilters/_StaticBandPassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Static bandpass filter with frequency band generation
Following field-standard bandwidths for neuroscience
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, lfilter


class StaticBandPassFilter(nn.Module):
    """
    Static bandpass filter with automatic or manual frequency band specification.

    Supports two modes:
    1. Automatic generation with strict boundary enforcement
    2. Manual band specification for precise control

    Field-standard bandwidths (when auto-generating):
    - Phase bands: [f - f/4, f + f/4] (bandwidth = f/2)
    - Amplitude bands: [f - f/8, f + f/8] (bandwidth = f/4)

    WARNING: Manual bands take precedence over range parameters.
    If both are provided, manual bands will be used and range parameters ignored.
    """

    def __init__(
        self,
        fs: float,
        pha_range_hz: Optional[Tuple[float, float]] = (2, 20),
        amp_range_hz: Optional[Tuple[float, float]] = (60, 160),
        pha_n_bands: Optional[int] = 10,
        amp_n_bands: Optional[int] = 10,
        pha_bands_hz: Optional[List[List[float]]] = None,
        amp_bands_hz: Optional[List[List[float]]] = None,
        n_cycles: int = 4,
        spacing: str = "linear",
        fp16: bool = False,
    ):
        """
        Initialize static bandpass filter.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        pha_range_hz : tuple, optional
            Phase frequency range (min, max) in Hz. Default: (2, 20). Ignored if pha_bands_hz provided.
        amp_range_hz : tuple, optional
            Amplitude frequency range (min, max) in Hz. Default: (60, 160). Ignored if amp_bands_hz provided.
        pha_n_bands : int, optional
            Number of phase filters. Default: 10. Ignored if pha_bands_hz provided.
        amp_n_bands : int, optional
            Number of amplitude filters. Default: 10. Ignored if amp_bands_hz provided.
        pha_bands_hz : list of lists, optional
            Manual phase bands [[low1, high1], [low2, high2], ...].
            Takes precedence over pha_range_hz and pha_n_bands.
        amp_bands_hz : list of lists, optional
            Manual amplitude bands [[low1, high1], [low2, high2], ...].
            Takes precedence over amp_range_hz and amp_n_bands.
        n_cycles : int
            Filter order (number of cycles for bandwidth calculation). Default: 4.
        spacing : str
            'log' for logarithmic spacing, 'linear' for linear spacing. Default: 'linear'.
        fp16 : bool
            Use half precision for memory efficiency. Default: False.

        Raises
        ------
        ValueError
            If band frequencies exceed Nyquist frequency or are invalid
        """
        super().__init__()
        self.fs = float(fs)
        self.n_cycles = int(n_cycles)
        self.spacing = spacing
        self.fp16 = fp16
        self.nyq = self.fs / 2

        # Priority warning for conflicting parameters
        self._check_parameter_conflicts(
            pha_range_hz,
            pha_n_bands,
            pha_bands_hz,
            amp_range_hz,
            amp_n_bands,
            amp_bands_hz,
        )

        # Device management
        if self.fp16:
            self.register_parameter(
                "_dummy", nn.Parameter(torch.zeros(1, dtype=torch.half))
            )
        else:
            self.register_parameter("_dummy", nn.Parameter(torch.zeros(1)))

        # Handle direct band specification vs range generation
        if pha_bands_hz is not None:
            self.pha_bands_hz = torch.tensor(pha_bands_hz, dtype=torch.float32)
            self.pha_n_bands = len(pha_bands_hz)
            self.pha_center_freqs = (
                self.pha_bands_hz[:, 0] + self.pha_bands_hz[:, 1]
            ) / 2
        else:
            self.pha_bands_hz, self.pha_center_freqs = self._generate_bands_strict(
                pha_range_hz[0], pha_range_hz[1], pha_n_bands, "phase"
            )
            self.pha_n_bands = pha_n_bands

        if amp_bands_hz is not None:
            self.amp_bands_hz = torch.tensor(amp_bands_hz, dtype=torch.float32)
            self.amp_n_bands = len(amp_bands_hz)
            self.amp_center_freqs = (
                self.amp_bands_hz[:, 0] + self.amp_bands_hz[:, 1]
            ) / 2
        else:
            self.amp_bands_hz, self.amp_center_freqs = self._generate_bands_strict(
                amp_range_hz[0],
                amp_range_hz[1],
                amp_n_bands,
                "amplitude",
            )
            self.amp_n_bands = amp_n_bands

        # Validate all bands against Nyquist
        self._validate_nyquist(self.pha_bands_hz)
        self._validate_nyquist(self.amp_bands_hz)

        # Extract frequency components for compatibility
        self.pha_low = self.pha_bands_hz[:, 0]
        self.pha_high = self.pha_bands_hz[:, 1]
        self.amp_low = self.amp_bands_hz[:, 0]
        self.amp_high = self.amp_bands_hz[:, 1]

        # Build filter bank
        filter_bank = self._build_filter_bank()
        self.register_buffer("filter_kernels", filter_bank)

        # Precompute padding as int
        self._padding = self.filter_kernels.shape[-1] // 2
        self.n_filters = len(self.pha_bands_hz) + len(self.amp_bands_hz)

    def _check_parameter_conflicts(
        self,
        pha_range_hz,
        pha_n_bands,
        pha_bands_hz,
        amp_range_hz,
        amp_n_bands,
        amp_bands_hz,
    ):
        """Check for parameter conflicts and issue warnings."""
        if pha_bands_hz is not None and (
            pha_range_hz is not None or pha_n_bands is not None
        ):
            warnings.warn(
                "Both pha_bands_hz and range parameters provided. "
                "Manual pha_bands_hz takes precedence - range parameters ignored.",
                UserWarning,
            )

        if amp_bands_hz is not None and (
            amp_range_hz is not None or amp_n_bands is not None
        ):
            warnings.warn(
                "Both amp_bands_hz and range parameters provided. "
                "Manual amp_bands_hz takes precedence - range parameters ignored.",
                UserWarning,
            )

    def _generate_bands_strict(
        self, f_min: float, f_max: float, n_filters: int, band_type: str
    ):
        """
        Generate frequency bands with strict boundary enforcement.

        Parameters
        ----------
        f_min : float
            Minimum frequency (strict lower bound)
        f_max : float
            Maximum frequency (strict upper bound)
        n_filters : int
            Number of filters to generate
        band_type : str
            'phase' or 'amplitude' for bandwidth calculation

        Returns
        -------
        bands : torch.Tensor
            Shape (n_filters, 2) with [low, high] frequencies
        center_freqs : torch.Tensor
            Center frequencies of each band
        """
        if self.spacing == "log":
            center_freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_filters)
        else:
            center_freqs = np.linspace(f_min, f_max, n_filters)

        # Calculate bandwidth using field standards
        if band_type == "phase":
            bandwidths = center_freqs / 2.0
        else:
            bandwidths = center_freqs / 4.0

        # Calculate band edges
        low_freqs = center_freqs - bandwidths / 2
        high_freqs = center_freqs + bandwidths / 2

        # Enforce strict boundaries
        low_freqs = np.maximum(low_freqs, f_min)
        high_freqs = np.minimum(high_freqs, f_max)

        # Ensure minimum valid frequency
        low_freqs = np.maximum(low_freqs, 0.5)

        bands = torch.stack(
            [
                torch.tensor(low_freqs, dtype=torch.float32),
                torch.tensor(high_freqs, dtype=torch.float32),
            ],
            dim=1,
        )

        center_freqs = torch.tensor(center_freqs, dtype=torch.float32)
        return bands.type_as(self._dummy), center_freqs.type_as(self._dummy)

    def _validate_nyquist(self, bands):
        """
        Validate all bands against Nyquist frequency.

        Parameters
        ----------
        bands : torch.Tensor
            Frequency bands to validate

        Raises
        ------
        ValueError
            If any band frequency exceeds Nyquist frequency
        """
        if bands.max() >= self.nyq:
            invalid_bands = torch.where(bands >= self.nyq)
            raise ValueError(
                f"Band frequencies {bands[invalid_bands]} exceed Nyquist frequency {self.nyq} Hz"
            )

    def _calc_filter_length(self, freqs):
        """
        Calculate optimal filter length based on lowest frequency.

        Parameters
        ----------
        freqs : torch.Tensor
            Frequencies to base calculation on

        Returns
        -------
        int
            Odd filter length for symmetric filter
        """
        filter_length = int((self.n_cycles * self.fs) / freqs.min())
        return filter_length + (1 - filter_length % 2)

    def _design_fir_filter(self, low_freq, high_freq, filter_length):
        """
        Design FIR bandpass filter using Butterworth approximation.

        Parameters
        ----------
        low_freq : float
            Lower cutoff frequency
        high_freq : float
            Upper cutoff frequency
        filter_length : int
            Length of FIR filter

        Returns
        -------
        numpy.ndarray
            Normalized FIR filter coefficients
        """
        b, a = butter(
            self.n_cycles,
            [float(low_freq) / self.nyq, float(high_freq) / self.nyq],
            btype="band",
        )
        impulse = np.zeros(filter_length)
        impulse[filter_length // 2] = 1.0
        fir_kernel = lfilter(b, a, impulse)

        # May corrupted...
        # Normalize
        fir_kernel = fir_kernel / np.max(np.abs(fir_kernel))

        return torch.from_numpy(fir_kernel).float().type_as(self._dummy)

    def _build_filter_bank(self):
        """
        Build concatenated filter bank for grouped convolution.

        Returns
        -------
        torch.Tensor
            Filter kernels shaped (n_filters, 1, filter_length)
        """
        all_low_freqs = torch.cat([self.pha_low, self.amp_low])
        filter_length = self._calc_filter_length(all_low_freqs)
        kernels = []

        # Phase filters
        for low, high in zip(self.pha_low, self.pha_high):
            fir_kernel = self._design_fir_filter(low, high, filter_length)
            # kernels.append(torch.from_numpy(fir_kernel))
            kernels.append(fir_kernel)

        # Amplitude filters
        for low, high in zip(self.amp_low, self.amp_high):
            fir_kernel = self._design_fir_filter(low, high, filter_length)
            # kernels.append(torch.from_numpy(fir_kernel))
            kernels.append(fir_kernel)

        return torch.stack(kernels).unsqueeze(1).type_as(self._dummy)

    # def forward(self, x):
    #     """
    #     Apply zero-phase bandpass filtering to input signals.

    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Input signal of shape (..., seq_len)

    #     Returns
    #     -------
    #     torch.Tensor
    #         Filtered signals of shape (..., n_filters, seq_len)
    #     """
    #     original_shape = x.shape
    #     seq_len = original_shape[-1]
    #     x_flat = x.view(-1, seq_len)

    #     if self.fp16:
    #         x_flat = x_flat.half()
    #     x_flat = x_flat.to(self.filter_kernels.device)

    #     x_viewed = x_flat.unsqueeze(1)
    #     x_expanded = x_viewed.expand(-1, self.n_filters, -1)

    #     # Forward filtering
    #     filtered = F.conv1d(
    #         x_expanded,
    #         self.filter_kernels,
    #         groups=self.n_filters,
    #         padding=self._padding,
    #     )

    #     # Zero-phase filtering (forward-backward)
    #     filtered_backward = F.conv1d(
    #         filtered.flip(-1),
    #         self.filter_kernels,
    #         groups=self.n_filters,
    #         padding=self._padding,
    #     ).flip(-1)

    #     new_shape = original_shape[:-1] + (self.n_filters, seq_len)
    #     return filtered_backward.reshape(new_shape)

    @torch._dynamo.disable
    def forward(self, x):
        """
        Apply zero-phase bandpass filtering to input signals.
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (..., seq_len)
        Returns
        -------
        torch.Tensor
            Filtered signals of shape (..., n_filters, seq_len)
        """
        original_shape = x.shape
        seq_len = original_shape[-1]
        x_flat = x.view(-1, seq_len)

        x_flat = x_flat.to(self.filter_kernels.device)
        if self.fp16:
            x_flat = x_flat.half()

        x_viewed = x_flat.unsqueeze(1)
        x_expanded = x_viewed.expand(-1, self.n_filters, -1)

        # Forward filtering
        filtered = F.conv1d(
            x_expanded,
            self.filter_kernels,
            groups=self.n_filters,
            padding=self._padding,
        )

        # Zero-phase filtering (forward-backward)
        filtered_backward = F.conv1d(
            filtered.flip(-1),
            self.filter_kernels,
            groups=self.n_filters,
            padding=self._padding,
        ).flip(-1)

        if self.fp16:
            filtered_backward = filtered_backward.float()

        new_shape = original_shape[:-1] + (self.n_filters, seq_len)
        return filtered_backward.reshape(new_shape)

    @property
    def info(self):
        """
        Return comprehensive filter configuration.

        Returns
        -------
        dict
            Filter configuration dictionary
        """
        return {
            "pha_bands_hz": self.pha_bands_hz,
            "amp_bands_hz": self.amp_bands_hz,
            "pha_center_freqs": self.pha_center_freqs,
            "amp_center_freqs": self.amp_center_freqs,
            "fs": self.fs,
            "n_cycles": self.n_cycles,
            "n_filters": self.n_filters,
            "pha_n_bands": self.pha_n_bands,
            "amp_n_bands": self.amp_n_bands,
            "filter_length": self.filter_kernels.shape[-1],
            "spacing": self.spacing,
        }


# EOF
