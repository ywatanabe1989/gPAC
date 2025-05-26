#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 22:10:09 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_CombinedBandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_CombinedBandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import torch
import torch.nn as nn

from ._tensorpac_fir1 import design_filter_tensorpac


class CombinedBandPassFilter(nn.Module):
    """
    Combined filter module that applies different cycle parameters
    to phase and amplitude bands, matching TensorPAC's approach.
    """

    def __init__(
        self,
        pha_bands,
        amp_bands,
        fs,
        seq_len,
        fp16=False,
        cycle_pha=3,
        cycle_amp=6,
        filtfilt_mode=False,
        edge_mode=None,
    ):
        super().__init__()
        self.fp16 = fp16
        self.n_pha_bands = len(pha_bands)
        self.n_amp_bands = len(amp_bands)
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode

        # Create phase filters with cycle_pha
        pha_filters = []
        for ll, hh in pha_bands:
            kernel = design_filter_tensorpac(
                seq_len, fs, low_hz=ll, high_hz=hh, cycle=cycle_pha
            )
            pha_filters.append(kernel)

        # Create amplitude filters with cycle_amp
        amp_filters = []
        for ll, hh in amp_bands:
            kernel = design_filter_tensorpac(
                seq_len, fs, low_hz=ll, high_hz=hh, cycle=cycle_amp
            )
            amp_filters.append(kernel)

        # Combine all filters
        all_filters = pha_filters + amp_filters

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
        if fp16:
            kernels = kernels.half()
        self.register_buffer("kernels", kernels)

        # Calculate padlen for edge handling if requested
        if edge_mode:
            # Get the maximum filter length for padding
            self.padlen = max(len(f) for f in all_filters) - 1
        else:
            self.padlen = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filtering with combined filters."""
        # x shape: (batch*channel*segment, 1, time)

        # Apply edge padding if requested
        if self.edge_mode and self.padlen > 0:
            # Pad along time dimension
            x = torch.nn.functional.pad(
                x, (self.padlen, self.padlen), mode=self.edge_mode
            )

        if self.filtfilt_mode:
            # Apply sequential filtfilt-style zero-phase filtering
            # This matches scipy.signal.filtfilt behavior exactly

            # Expand input to match number of filters
            # x: (batch, 1, time) -> (batch, n_filters, time)
            x_expanded = x.expand(-1, len(self.kernels), -1)

            # Prepare kernels for depthwise conv
            # kernels: (n_filters, kernel_len) -> (n_filters, 1, kernel_len)
            kernels_expanded = self.kernels.unsqueeze(1)

            # First forward pass using depthwise convolution
            filtered = torch.nn.functional.conv1d(
                x_expanded,
                kernels_expanded,
                padding="same",
                groups=len(
                    self.kernels
                ),  # Each filter processes its own channel
            )

            # Second pass on time-reversed signal (backward filtering)
            filtered = torch.nn.functional.conv1d(
                filtered.flip(-1),  # Flip time dimension
                kernels_expanded,
                padding="same",
                groups=len(
                    self.kernels
                ),  # Each filter processes its own channel
            ).flip(
                -1
            )  # Flip back

            # Sequential filtering is actually faster than averaging
            # and provides exact scipy.signal.filtfilt compatibility
        else:
            # Standard single-pass filtering
            filtered = torch.nn.functional.conv1d(
                x,
                self.kernels.unsqueeze(1),  # (n_bands, 1, kernel_len)
                padding="same",
                groups=1,
            )

        # filtered shape: (batch*channel*segment, n_bands, time)

        # Remove edge padding if it was applied
        if self.edge_mode and self.padlen > 0:
            filtered = filtered[:, :, self.padlen : -self.padlen]

        # Add extra dimension to match expected output
        # Output should be: (batch*channel*segment, 1, n_bands, time)
        filtered = filtered.unsqueeze(1)

        return filtered

# EOF
