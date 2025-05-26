#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 11:15:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# ----------------------------------------

import torch
import torch.nn as nn
import numpy as np

from ._tensorpac_fir1 import design_filter_tensorpac


class BandPassFilter(nn.Module):
    """
    Bandpass filter with combined phase and amplitude filtering.
    
    This implementation uses scipy-compatible odd extension padding
    for exact filtfilt behavior when filtfilt_mode=True.
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
            x = torch.nn.functional.pad(
                x, (self.padlen, self.padlen), mode=self.edge_mode
            )

        if self.filtfilt_mode:
            # For exact scipy compatibility, we need odd extension padding
            batch_size = x.shape[0]
            seq_len = x.shape[-1]
            
            # Convert input for processing
            x_squeezed = x.squeeze(1)  # (batch, time)
            
            # Process each filter with appropriate padding
            filtered_bands = []
            
            for i, kernel in enumerate(self.kernels):
                # Calculate padlen based on filter length
                padlen = min(3 * len(kernel), seq_len - 1)
            
                # Apply odd extension padding manually for each signal
                x_padded_list = []
                for b in range(batch_size):
                    signal = x_squeezed[b]
                    
                    if padlen > 0:
                        # Create odd extension: -flipped_signal
                        left_pad = -signal[1:padlen+1].flip(0)
                        right_pad = -signal[-padlen-1:-1].flip(0) 
                        signal_padded = torch.cat([left_pad, signal, right_pad])
                    else:
                        signal_padded = signal
                    
                    x_padded_list.append(signal_padded)
            
                # Stack padded signals
                x_padded = torch.stack(x_padded_list).unsqueeze(1)  # (batch, 1, padded_time)
                
                # Apply forward and backward filtering
                kernel_3d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_len)
                filtered_band = torch.nn.functional.conv1d(x_padded, kernel_3d, padding='same')
                filtered_band = torch.nn.functional.conv1d(
                    filtered_band.flip(-1), kernel_3d, padding='same'
                ).flip(-1)
                
                # Remove padding
                if padlen > 0:
                    filtered_band = filtered_band[:, :, padlen:-padlen]
                
                filtered_bands.append(filtered_band.squeeze(1))
            
            # Stack all filtered bands
            filtered = torch.stack(filtered_bands, dim=1)  # (batch, n_bands, time)
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