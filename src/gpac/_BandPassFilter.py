#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 19:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/_BandPassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Unified BandPassFilter module that combines static and differentiable filtering modes.
Provides a single interface for both trainable and non-trainable bandpass filtering.
"""

import torch
import torch.nn as nn

from ._Filters._DifferentiableBandpassFilter import DifferentiableBandPassFilter
from ._Filters._StaticBandpassFilter import StaticBandPassFilter


class BandPassFilter(nn.Module):
    """
    Unified bandpass filter that switches between static and differentiable modes.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence
    fs : float
        Sampling frequency
    pha_start_hz : float
        Start frequency for phase bands
    pha_end_hz : float
        End frequency for phase bands
    pha_n_bands : int
        Number of phase bands
    amp_start_hz : float
        Start frequency for amplitude bands
    amp_end_hz : float
        End frequency for amplitude bands
    amp_n_bands : int
        Number of amplitude bands
    fp16 : bool
        Whether to use half precision
    trainable : bool
        Whether to use trainable (differentiable) filters
    """

    def __init__(
        self,
        seq_len,
        fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=50,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=30,
        fp16=False,
        trainable=False,
        padding_mode="reflect",
    ):
        super().__init__()

        # Validate parameters
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got seq_len={seq_len}")
        if pha_start_hz >= pha_end_hz:
            raise ValueError(
                f"Invalid phase frequency range: {pha_start_hz} >= {pha_end_hz}"
            )
        if amp_start_hz >= amp_end_hz:
            raise ValueError(
                f"Invalid amplitude frequency range: {amp_start_hz} >= {amp_end_hz}"
            )
        if pha_n_bands <= 0:
            raise ValueError(
                f"Number of bands must be positive, got pha_n_bands={pha_n_bands}"
            )
        if amp_n_bands <= 0:
            raise ValueError(
                f"Number of bands must be positive, got amp_n_bands={amp_n_bands}"
            )
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got fs={fs}")

        # Check Nyquist frequency
        nyquist = fs / 2
        if pha_end_hz > nyquist:
            raise ValueError(
                f"Phase frequency {pha_end_hz} Hz exceeds Nyquist frequency {nyquist} Hz"
            )
        if amp_end_hz > nyquist:
            raise ValueError(
                f"Amplitude frequency {amp_end_hz} Hz exceeds Nyquist frequency {nyquist} Hz"
            )

        self.seq_len = seq_len
        self.fs = fs
        self.fp16 = fp16
        self.trainable = trainable
        self.pha_n_bands = pha_n_bands
        self.amp_n_bands = amp_n_bands
        self.pha_start_hz = pha_start_hz
        self.pha_end_hz = pha_end_hz
        self.amp_start_hz = amp_start_hz
        self.amp_end_hz = amp_end_hz
        self.padding_mode = padding_mode

        if trainable:
            # Use differentiable bandpass filter with SincNet style
            self.filter = DifferentiableBandPassFilter(
                sig_len=seq_len,
                fs=fs,
                pha_low_hz=pha_start_hz,
                pha_high_hz=pha_end_hz,
                pha_n_bands=pha_n_bands,
                amp_low_hz=amp_start_hz,
                amp_high_hz=amp_end_hz,
                amp_n_bands=amp_n_bands,
                filter_length=251,
                min_band_hz=1,
                min_low_hz=1,
                init_scale="mel",
                window="hamming",
                normalization="std",
                fp16=fp16,
            )
            # Get initial band centers for compatibility
            bands_info = self.filter.get_filter_banks()
            self.pha_mids = bands_info["pha_bands"][:, :].mean(dim=1)
            self.amp_mids = bands_info["amp_bands"][:, :].mean(dim=1)
        else:
            # Use static bandpass filter
            # First calculate the bands
            pha_bands = self._calc_bands_pha(pha_start_hz, pha_end_hz, pha_n_bands)
            amp_bands = self._calc_bands_amp(amp_start_hz, amp_end_hz, amp_n_bands)
            all_bands = torch.vstack([pha_bands, amp_bands])

            self.filter = StaticBandPassFilter(
                bands=all_bands,
                fs=fs,
                seq_len=seq_len,
                fp16=fp16,
                padding_mode=padding_mode,
            )
            # Store mid frequencies for PAC
            self.pha_mids = pha_bands.mean(-1)
            self.amp_mids = amp_bands.mean(-1)

    def forward(self, x, edge_len=0):
        """
        Apply bandpass filtering.

        Parameters
        ----------
        x : torch.Tensor
            Input signal with shape (batch_size, n_segments, seq_len)
        edge_len : int
            Number of samples to remove from edges

        Returns
        -------
        torch.Tensor
            Filtered signal with shape (batch_size, n_segments, n_bands, seq_len)
        """
        return self.filter(x, edge_len=edge_len)

    @staticmethod
    def _calc_bands_pha(start_hz=2, end_hz=20, n_bands=100):
        """Calculate phase frequency bands."""
        start_hz = start_hz if start_hz is not None else 2
        end_hz = end_hz if end_hz is not None else 20
        mid_hz = torch.linspace(start_hz, end_hz, n_bands)
        return torch.cat(
            (
                mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 4.0,
                mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 4.0,
            ),
            dim=1,
        )

    @staticmethod
    def _calc_bands_amp(start_hz=30, end_hz=160, n_bands=100):
        """Calculate amplitude frequency bands."""
        start_hz = start_hz if start_hz is not None else 30
        end_hz = end_hz if end_hz is not None else 160
        mid_hz = torch.linspace(start_hz, end_hz, n_bands)
        return torch.cat(
            (
                mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 8.0,
                mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 8.0,
            ),
            dim=1,
        )

    def get_filter_banks(self):
        """Get current filter bank parameters (for trainable filters)."""
        if hasattr(self.filter, "get_filter_banks"):
            return self.filter.get_filter_banks()
        else:
            # For static filters, return the fixed bands
            pha_bands = self._calc_bands_pha(
                getattr(self, "pha_start_hz", 2),
                getattr(self, "pha_end_hz", 20),
                self.pha_n_bands,
            )
            amp_bands = self._calc_bands_amp(
                getattr(self, "amp_start_hz", 60),
                getattr(self, "amp_end_hz", 160),
                self.amp_n_bands,
            )
            return {"pha_bands": pha_bands, "amp_bands": amp_bands}

    def get_regularization_loss(self, lambda_overlap=0.01, lambda_bandwidth=0.01):
        """Get regularization loss for trainable filters."""
        if hasattr(self.filter, "get_regularization_loss"):
            return self.filter.get_regularization_loss(lambda_overlap, lambda_bandwidth)
        else:
            return {"total": torch.tensor(0.0)}

    def constrain_parameters(self):
        """Apply constraints to trainable parameters."""
        if hasattr(self.filter, "constrain_parameters"):
            self.filter.constrain_parameters()


# EOF
