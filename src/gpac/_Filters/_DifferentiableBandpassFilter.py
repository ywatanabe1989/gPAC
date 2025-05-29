#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 18:49:51 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_filter/_DifferentiableBandpassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/_filter/_DifferentiableBandpassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SincNet-style differentiable bandpass filter with learnable frequency boundaries.
Based on: Ravanelli & Bengio, "Speaker Recognition from raw waveform with SincNet" (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._BaseFilter1D import BaseFilter1D


class DifferentiableBandPassFilter(BaseFilter1D):
    """
    SincNet-style learnable bandpass filter.

    This implementation uses parametric sinc functions with learnable
    low and high cutoff frequencies for each band.
    """

    def __init__(
        self,
        sig_len,
        fs,
        pha_low_hz=2,
        pha_high_hz=20,
        pha_n_bands=30,
        amp_low_hz=80,
        amp_high_hz=160,
        amp_n_bands=50,
        filter_length=251,  # Should be odd
        min_band_hz=1,  # Minimum bandwidth in Hz
        min_low_hz=1,  # Minimum low frequency in Hz
        init_scale="mel",  # 'mel' or 'linear' initialization
        window="hamming",  # 'hamming' or 'rectangular'
        normalization="std",  # 'std', 'freq_weighted', or 'bandwidth'
        fp16=False,
    ):
        super().__init__(fp16=fp16)

        # Ensure filter length is odd
        if filter_length % 2 == 0:
            filter_length += 1

        # Store parameters
        self.sig_len = sig_len
        self.fs = fs
        self.filter_length = filter_length
        self.min_band_hz = min_band_hz
        self.min_low_hz = min_low_hz
        self.window_type = window
        self.normalization = normalization
        self.fp16 = fp16

        # Frequency bounds
        self.pha_low_hz = pha_low_hz
        self.pha_high_hz = pha_high_hz
        self.pha_n_bands = pha_n_bands
        self.amp_low_hz = amp_low_hz
        self.amp_high_hz = amp_high_hz
        self.amp_n_bands = amp_n_bands

        # Total number of filters
        self.n_filters = pha_n_bands + amp_n_bands

        # Initialize learnable parameters
        self._init_parameters(init_scale)

        # Create window function
        self.register_buffer("window", self._get_window(filter_length, window))

        # Initialize kernels (will be updated in forward pass)
        self.register_buffer("kernels", torch.zeros(self.n_filters, filter_length))

    def _init_parameters(self, init_scale):
        """Initialize learnable frequency parameters."""
        # Initialize phase bands
        if init_scale == "mel":
            pha_low_mel = self._hz_to_mel(self.pha_low_hz)
            pha_high_mel = self._hz_to_mel(self.pha_high_hz)
            pha_freqs_mel = torch.linspace(
                pha_low_mel, pha_high_mel, self.pha_n_bands + 1
            )
            pha_freqs_hz = self._mel_to_hz(pha_freqs_mel)
        else:  # linear
            pha_freqs_hz = torch.linspace(
                self.pha_low_hz, self.pha_high_hz, self.pha_n_bands + 1
            )

        # Initialize amplitude bands
        if init_scale == "mel":
            amp_low_mel = self._hz_to_mel(self.amp_low_hz)
            amp_high_mel = self._hz_to_mel(self.amp_high_hz)
            amp_freqs_mel = torch.linspace(
                amp_low_mel, amp_high_mel, self.amp_n_bands + 1
            )
            amp_freqs_hz = self._mel_to_hz(amp_freqs_mel)
        else:  # linear
            amp_freqs_hz = torch.linspace(
                self.amp_low_hz, self.amp_high_hz, self.amp_n_bands + 1
            )

        # Create learnable parameters for low frequencies
        pha_low_freqs = pha_freqs_hz[:-1]
        amp_low_freqs = amp_freqs_hz[:-1]
        all_low_freqs = torch.cat([pha_low_freqs, amp_low_freqs])
        self.low_hz_ = nn.Parameter(all_low_freqs)

        # Create learnable parameters for bandwidths
        pha_bandwidths = pha_freqs_hz[1:] - pha_freqs_hz[:-1]
        amp_bandwidths = amp_freqs_hz[1:] - amp_freqs_hz[:-1]
        all_bandwidths = torch.cat([pha_bandwidths, amp_bandwidths])
        self.band_hz_ = nn.Parameter(all_bandwidths)

    @staticmethod
    def _hz_to_mel(hz):
        """Convert frequency in Hz to mel scale."""
        if isinstance(hz, (int, float)):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel):
        """Convert mel scale to frequency in Hz."""
        if isinstance(mel, (int, float)):
            mel = torch.tensor(mel, dtype=torch.float32)
        return 700 * (10 ** (mel / 2595) - 1)

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

    def _compute_filters(self):
        """Compute sinc bandpass filters with current parameters."""
        # Get constrained frequencies
        low_hz = torch.abs(self.low_hz_) + self.min_low_hz
        high_hz = torch.abs(self.band_hz_) + self.min_band_hz + low_hz

        # Ensure we don't exceed Nyquist frequency
        nyq = self.fs / 2.0
        high_hz = torch.min(high_hz, torch.tensor(nyq - 1).to(high_hz.device))

        # Time axis (centered at 0)
        n = torch.arange(self.filter_length).float() - (self.filter_length - 1) / 2
        n = n.to(low_hz.device)

        # Compute sinc filters
        filters = []
        for i in range(self.n_filters):
            low_freq = low_hz[i] * 2 / self.fs
            high_freq = high_hz[i] * 2 / self.fs

            # Handle the special case at n=0 to avoid division by zero
            n_low = low_freq * torch.sinc(low_freq * n)
            n_high = high_freq * torch.sinc(high_freq * n)

            # Bandpass = highpass - lowpass
            band_pass = n_high - n_low

            # Apply window
            band_pass = band_pass * self.window

            # Frequency-dependent normalization
            # Higher frequencies have naturally higher variance, so we compensate
            center_freq = (low_hz[i] + high_hz[i]) / 2
            bandwidth = high_hz[i] - low_hz[i]

            # Method 1: Standard deviation normalization (preserves relative energy)
            if self.normalization == "std":
                band_pass = band_pass / torch.sqrt(torch.sum(band_pass**2))

            # Method 2: Frequency-weighted normalization (compensates for frequency-dependent energy)
            elif self.normalization == "freq_weighted":
                # Lower frequencies get less normalization, higher frequencies get more
                freq_weight = torch.sqrt(
                    center_freq / 100.0
                )  # Adjust 100.0 based on your frequency range
                band_pass = band_pass / (
                    torch.sqrt(torch.sum(band_pass**2)) * freq_weight
                )

            # Method 3: Bandwidth-aware normalization
            elif self.normalization == "bandwidth":
                # Normalize by bandwidth to ensure equal contribution
                band_pass = band_pass / (
                    torch.sqrt(torch.sum(band_pass**2)) * torch.sqrt(bandwidth)
                )

            # Additional gain correction to ensure unity passband gain
            # This is applied after the chosen normalization method
            import numpy as np
            from scipy import signal as scipy_signal

            w, h = scipy_signal.freqz(
                band_pass.detach().cpu().numpy(), worN=8192, fs=self.fs
            )

            # Find peak gain in passband (use peak instead of average for better normalization)
            low_hz_val = low_hz[i].item()
            high_hz_val = high_hz[i].item()
            center_freq = (low_hz_val + high_hz_val) / 2
            # Define passband more narrowly around center
            passband_mask = (w >= low_hz_val) & (w <= high_hz_val)
            if np.any(passband_mask):
                passband_gains = np.abs(h[passband_mask])
                # Use peak gain for normalization
                peak_passband_gain = np.max(passband_gains)
                if peak_passband_gain > 0:
                    band_pass = band_pass / peak_passband_gain

            filters.append(band_pass)

        return torch.stack(filters)

    def forward(self, x, t=None, edge_len=0):
        """Apply bandpass filtering with current learnable parameters."""
        # Update kernels with current parameters
        self.kernels = self._compute_filters()

        if self.fp16:
            self.kernels = self.kernels.half()

        # Apply filtering using parent class method
        return super().forward(x=x, t=t, edge_len=edge_len)

    def get_filter_banks(self):
        """Get current filter bank parameters for inspection."""
        with torch.no_grad():
            low_hz = torch.abs(self.low_hz_) + self.min_low_hz
            high_hz = torch.abs(self.band_hz_) + self.min_band_hz + low_hz

            # Ensure we don't exceed Nyquist
            nyq = self.fs / 2.0
            high_hz = torch.min(high_hz, torch.tensor(nyq - 1))

            # Split into phase and amplitude bands
            pha_low = low_hz[: self.pha_n_bands]
            pha_high = high_hz[: self.pha_n_bands]
            amp_low = low_hz[self.pha_n_bands :]
            amp_high = high_hz[self.pha_n_bands :]

            return {
                "pha_bands": torch.stack([pha_low, pha_high], dim=1),
                "amp_bands": torch.stack([amp_low, amp_high], dim=1),
                "all_bands": torch.stack([low_hz, high_hz], dim=1),
            }

    def init_kernels(self):
        """Initialize kernels (required by parent class)."""
        # This is called by parent __init__, but we handle kernel
        # initialization in forward() for differentiability
        pass

    def get_regularization_loss(self, lambda_overlap=0.01, lambda_bandwidth=0.01):
        """
        Compute regularization losses to control filter behavior.

        Args:
            lambda_overlap: Weight for overlap penalty (encourages non-overlapping bands)
            lambda_bandwidth: Weight for bandwidth penalty (prevents too narrow/wide bands)

        Returns:
            Dictionary with individual losses and total regularization loss
        """
        # Get current bands
        low_hz = torch.abs(self.low_hz_) + self.min_low_hz
        high_hz = torch.abs(self.band_hz_) + self.min_band_hz + low_hz

        # Ensure we don't exceed Nyquist
        nyq = self.fs / 2.0
        high_hz = torch.min(high_hz, torch.tensor(nyq - 1).to(high_hz.device))

        losses = {}

        # 1. Overlap penalty: Penalize overlapping adjacent bands
        overlap_loss = 0
        for i in range(self.n_filters - 1):
            # Check if band i overlaps with band i+1
            overlap = torch.relu(high_hz[i] - low_hz[i + 1])
            overlap_loss += overlap
        losses["overlap"] = overlap_loss * lambda_overlap

        # 2. Bandwidth regularization: Prevent too narrow or too wide bands
        bandwidths = high_hz - low_hz

        # Penalize very narrow bands (less than min_band_hz)
        narrow_penalty = torch.sum(torch.relu(self.min_band_hz - bandwidths))

        # Penalize very wide bands (optional, based on frequency range)
        # Phase bands shouldn't be too wide (e.g., > 10 Hz)
        # Amplitude bands can be wider (e.g., > 50 Hz)
        pha_bands = bandwidths[: self.pha_n_bands]
        amp_bands = bandwidths[self.pha_n_bands :]

        max_pha_bandwidth = 10.0  # Hz
        max_amp_bandwidth = 50.0  # Hz

        wide_pha_penalty = torch.sum(torch.relu(pha_bands - max_pha_bandwidth))
        wide_amp_penalty = torch.sum(torch.relu(amp_bands - max_amp_bandwidth))

        bandwidth_loss = narrow_penalty + 0.1 * (wide_pha_penalty + wide_amp_penalty)
        losses["bandwidth"] = bandwidth_loss * lambda_bandwidth

        # 3. Frequency ordering: Ensure bands are ordered by frequency
        order_loss = 0
        for i in range(self.n_filters - 1):
            # Penalize if band i+1 starts before band i
            order_violation = torch.relu(low_hz[i] - low_hz[i + 1])
            order_loss += order_violation
        losses["order"] = order_loss * 0.1  # Fixed weight for ordering

        # Total regularization loss
        losses["total"] = losses["overlap"] + losses["bandwidth"] + losses["order"]

        return losses

    def constrain_parameters(self):
        """
        Apply hard constraints to parameters after optimization step.
        Can be called after optimizer.step() to ensure valid parameters.
        """
        with torch.no_grad():
            # Ensure positive frequencies
            self.low_hz_.data = torch.abs(self.low_hz_.data)
            self.band_hz_.data = torch.abs(self.band_hz_.data)

            # Ensure minimum bandwidth
            self.band_hz_.data = torch.clamp(self.band_hz_.data, min=self.min_band_hz)

            # Ensure bands don't exceed Nyquist
            nyq = self.fs / 2.0
            max_high = nyq - 1

            for i in range(self.n_filters):
                potential_high = self.low_hz_.data[i] + self.band_hz_.data[i]
                if potential_high > max_high:
                    # Reduce bandwidth to fit within Nyquist
                    self.band_hz_.data[i] = max_high - self.low_hz_.data[i]


# EOF
