#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 09:15:11 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/core/_BandPassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/core/_BandPassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ._BandPassFilters._PooledBandPassFilter import PooledBandPassFilter
from ._BandPassFilters._StaticBandPassFilter import StaticBandPassFilter


class BandPassFilter(nn.Module):
    """
    Unified bandpass filter supporting static and learnable modes.
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
        trainable: bool = False,
        pha_n_pool_ratio: Optional[float] = 2.0,
        amp_n_pool_ratio: Optional[float] = 2.0,
        temperature: float = 1.0,
        hard_selection: bool = False,
    ):
        """
        Initialize unified bandpass filter.
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        pha_range_hz : tuple, optional
            Phase frequency range (min, max) in Hz. Ignored if pha_bands_hz provided.
        amp_range_hz : tuple, optional
            Amplitude frequency range (min, max) in Hz. Ignored if amp_bands_hz provided.
        pha_n_bands : int, optional
            Number of phase filters. Ignored if pha_bands_hz provided.
        amp_n_bands : int, optional
            Number of amplitude filters. Ignored if amp_bands_hz provided.
        pha_bands_hz : list of lists, optional
            Manual phase bands [[low1, high1], [low2, high2], ...].
            Takes precedence over range parameters.
        amp_bands_hz : list of lists, optional
            Manual amplitude bands [[low1, high1], [low2, high2], ...].
            Takes precedence over range parameters.
        n_cycles : int
            Filter order (number of cycles for bandwidth calculation)
        spacing : str
            'log' for logarithmic spacing, 'linear' for linear spacing
        fp16 : bool
            Use half precision for memory efficiency
        trainable : bool
            Whether to use learnable (PooledBandPassFilter) or static filter
        pha_n_pool_ratio : float, optional
            Ratio of pool size to number of bands for phase filters (only for trainable mode)
        amp_n_pool_ratio : float, optional
            Ratio of pool size to number of bands for amplitude filters (only for trainable mode)
        temperature : float
            Gumbel Softmax temperature (only for trainable mode)
        hard_selection : bool
            Use hard selection in forward pass (only for trainable mode)
        Raises
        ------
        ValueError
            If parameters are invalid or exceed Nyquist frequency
        """
        super().__init__()
        # Validate sampling frequency
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got fs={fs}")
        self.fs = fs
        self.trainable = trainable
        self.spacing = spacing
        nyquist = fs / 2

        # Check for parameter conflicts
        self._check_parameter_conflicts(
            pha_range_hz,
            pha_n_bands,
            pha_bands_hz,
            amp_range_hz,
            amp_n_bands,
            amp_bands_hz,
        )

        # Validate frequency ranges if provided
        if pha_range_hz is not None and pha_range_hz[1] > nyquist:
            raise ValueError(
                f"Phase frequency {pha_range_hz[1]} Hz exceeds Nyquist frequency {nyquist} Hz"
            )
        if amp_range_hz is not None and amp_range_hz[1] > nyquist:
            raise ValueError(
                f"Amplitude frequency {amp_range_hz[1]} Hz exceeds Nyquist frequency {nyquist} Hz"
            )

        # Store parameters for compatibility
        self.pha_range_hz = pha_range_hz
        self.amp_range_hz = amp_range_hz
        self.pha_n_bands = pha_n_bands if pha_bands_hz is None else len(pha_bands_hz)
        self.amp_n_bands = amp_n_bands if amp_bands_hz is None else len(amp_bands_hz)
        self.pha_n_pool_ratio = pha_n_pool_ratio
        self.amp_n_pool_ratio = amp_n_pool_ratio

        # Create appropriate filter
        if trainable:
            self.filter = PooledBandPassFilter(
                fs=fs,
                pha_range_hz=pha_range_hz,
                amp_range_hz=amp_range_hz,
                pha_n_bands=pha_n_bands,
                amp_n_bands=amp_n_bands,
                pha_bands_hz=pha_bands_hz,
                amp_bands_hz=amp_bands_hz,
                pha_n_pool_ratio=pha_n_pool_ratio,
                amp_n_pool_ratio=amp_n_pool_ratio,
                n_cycles=n_cycles,
                temperature=temperature,
                hard_selection=hard_selection,
                spacing=spacing,
                fp16=fp16,
            )
        else:
            self.filter = StaticBandPassFilter(
                fs=fs,
                pha_range_hz=pha_range_hz,
                amp_range_hz=amp_range_hz,
                pha_n_bands=pha_n_bands,
                amp_n_bands=amp_n_bands,
                pha_bands_hz=pha_bands_hz,
                amp_bands_hz=amp_bands_hz,
                n_cycles=n_cycles,
                spacing=spacing,
                fp16=fp16,
            )

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

    def forward(self, x):
        """
        Apply bandpass filtering.
        Parameters
        ----------
        x : torch.Tensor
            Input signal with shape (..., seq_len)
        Returns
        -------
        torch.Tensor
            Filtered signal with shape (..., n_bands_total, seq_len)
            where n_bands_total = n_bands_pha + n_bands_amp
        """
        return self.filter(x)

    def get_regularization_loss(self):
        """
        Get regularization loss for learnable filters.
        Returns
        -------
        torch.Tensor
            Regularization loss (zero for static filters)
        """
        if self.trainable and hasattr(self.filter, "get_regularization_loss"):
            return self.filter.get_regularization_loss()
        else:
            return torch.tensor(0.0)

    def get_selection_weights(self):
        """
        Get selection weights for learnable filters.
        Returns
        -------
        tuple or (None, None)
            (phase_weights, amplitude_weights) for learnable filters, (None, None) for static
        """
        if self.trainable and hasattr(self.filter, "get_selection_weights"):
            return self.filter.get_selection_weights()
        else:
            return None, None

    @property
    def info(self):
        """
        Return comprehensive filter configuration.
        Returns
        -------
        dict
            Filter configuration dictionary with center frequencies
        """
        base_info = self.filter.info
        base_info["trainable"] = self.trainable
        return base_info

    @property
    def pha_mids(self):
        """Phase band center frequencies."""
        return self.info["pha_center_freqs"]

    @property
    def amp_mids(self):
        """Amplitude band center frequencies."""
        return self.info["amp_center_freqs"]

    @property
    def pha_bands_hz(self):
        """Phase frequency bands."""
        return self.filter.pha_bands_hz

    @property
    def amp_bands_hz(self):
        """Amplitude frequency bands."""
        return self.filter.amp_bands_hz


# EOF
