#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .._tensorpac_fir1 import design_filter_tensorpac


class BaseBandPassFilter(nn.Module, ABC):
    """
    Abstract base class for bandpass filters.
    Contains common filtering logic extracted from BandPassFilter.
    """

    def __init__(
        self,
        fs,
        seq_len,
        fp16=False,
        filtfilt_mode=False,
        edge_mode=None,
    ):
        super().__init__()
        self.fs = fs
        self.seq_len = seq_len
        self.fp16 = fp16
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode
        self.padlen = 0

    @abstractmethod
    def get_bands(self):
        """
        Get frequency bands for filtering.
        Must return (pha_bands, amp_bands, cycle_pha, cycle_amp).
        """
        pass

    def _create_kernels(self, pha_bands, amp_bands, cycle_pha, cycle_amp):
        """Create filter kernels from band specifications."""
        # Create phase filters with cycle_pha
        pha_filters = []
        for ll, hh in pha_bands:
            kernel = design_filter_tensorpac(
                self.seq_len, self.fs, low_hz=ll, high_hz=hh, cycle=cycle_pha
            )
            pha_filters.append(kernel)

        # Create amplitude filters with cycle_amp
        amp_filters = []
        for ll, hh in amp_bands:
            kernel = design_filter_tensorpac(
                self.seq_len, self.fs, low_hz=ll, high_hz=hh, cycle=cycle_amp
            )
            amp_filters.append(kernel)

        # Combine all filters
        all_filters = pha_filters + amp_filters

        if not all_filters:
            raise ValueError("No filters created. Check band specifications.")

        # Find max length for padding
        max_len = max(f.shape[0] for f in all_filters)

        # Pad filters to same length
        padded_filters = []
        for f in all_filters:
            pad_needed = max_len - f.shape[0]
            if pad_needed > 0:
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
                f_padded = torch.nn.functional.pad(f, (pad_left, pad_right))
            else:
                f_padded = f
            padded_filters.append(f_padded)

        # Stack all filters
        kernels = torch.stack(padded_filters)
        if self.fp16:
            kernels = kernels.half()

        # Calculate padlen for edge handling if requested
        if self.edge_mode:
            self.padlen = max(len(f) for f in all_filters) - 1
        else:
            self.padlen = 0

        return kernels

    def _initialize_kernels(self):
        """Initialize kernels based on band specifications."""
        pha_bands, amp_bands, cycle_pha, cycle_amp = self.get_bands()
        kernels = self._create_kernels(pha_bands, amp_bands, cycle_pha, cycle_amp)
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filtering."""
        # Ensure kernels are initialized
        if not hasattr(self, 'kernels'):
            self._initialize_kernels()

        # x shape: (batch*channel*segment, 1, time)

        # Apply edge padding if requested
        if self.edge_mode and self.padlen > 0:
            x = torch.nn.functional.pad(
                x, (self.padlen, self.padlen), mode=self.edge_mode
            )

        if self.filtfilt_mode:
            # Apply sequential filtfilt-style zero-phase filtering
            x_expanded = x.expand(-1, len(self.kernels), -1)
            kernels_expanded = self.kernels.unsqueeze(1)

            # First forward pass
            filtered = torch.nn.functional.conv1d(
                x_expanded,
                kernels_expanded,
                padding="same",
                groups=len(self.kernels),
            )

            # Second pass on time-reversed signal
            filtered = torch.nn.functional.conv1d(
                filtered.flip(-1),
                kernels_expanded,
                padding="same",
                groups=len(self.kernels),
            ).flip(-1)

        else:
            # Standard single-pass filtering
            filtered = torch.nn.functional.conv1d(
                x,
                self.kernels.unsqueeze(1),
                padding="same",
                groups=1,
            )

        # Remove edge padding if it was applied
        if self.edge_mode and self.padlen > 0:
            filtered = filtered[:, :, self.padlen : -self.padlen]

        # Add extra dimension to match expected output
        filtered = filtered.unsqueeze(1)

        return filtered