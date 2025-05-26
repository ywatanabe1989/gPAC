#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 10:52:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_OptimizedBandPassFilter.py
# ----------------------------------------

import torch
import torch.nn as nn
import numpy as np
from functools import lru_cache
from typing import Tuple, Optional

from ._tensorpac_fir1 import design_filter_tensorpac


class OptimizedBandPassFilter(nn.Module):
    """
    Optimized bandpass filter with cached filter coefficients.
    
    Key optimizations:
    1. LRU cache for filter design (major bottleneck)
    2. Pre-computed FFT for frequency domain filtering
    3. Batch-optimized convolutions
    """
    
    # Class-level cache for filter coefficients
    _filter_cache = {}
    
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
        use_fft=False,  # New option for FFT-based filtering
    ):
        super().__init__()
        self.fp16 = fp16
        self.n_pha_bands = len(pha_bands)
        self.n_amp_bands = len(amp_bands)
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode
        self.use_fft = use_fft
        self.seq_len = seq_len
        self.fs = fs
        
        # Get or create cached filters
        kernels = self._get_cached_filters(
            pha_bands, amp_bands, fs, seq_len, cycle_pha, cycle_amp
        )
        
        if fp16:
            kernels = kernels.half()
        self.register_buffer("kernels", kernels)
        
        # Pre-compute FFT of kernels if using FFT mode
        if use_fft:
            self._prepare_fft_filters()
        
        # Calculate padlen for edge handling if requested
        if edge_mode:
            self.padlen = self.kernels.shape[1] - 1
        else:
            self.padlen = 0
    
    @classmethod
    def _get_cached_filters(
        cls,
        pha_bands: torch.Tensor,
        amp_bands: torch.Tensor,
        fs: float,
        seq_len: int,
        cycle_pha: int,
        cycle_amp: int,
    ) -> torch.Tensor:
        """Get filters from cache or create and cache them."""
        # Create cache key
        cache_key = (
            tuple(pha_bands.flatten().tolist()),
            tuple(amp_bands.flatten().tolist()),
            fs,
            seq_len,
            cycle_pha,
            cycle_amp,
        )
        
        if cache_key in cls._filter_cache:
            return cls._filter_cache[cache_key].clone()
        
        # Create filters (expensive operation)
        pha_filters = []
        for ll, hh in pha_bands:
            kernel = design_filter_tensorpac(
                seq_len, fs, low_hz=ll, high_hz=hh, cycle=cycle_pha
            )
            pha_filters.append(kernel)
        
        amp_filters = []
        for ll, hh in amp_bands:
            kernel = design_filter_tensorpac(
                seq_len, fs, low_hz=ll, high_hz=hh, cycle=cycle_amp
            )
            amp_filters.append(kernel)
        
        # Combine and pad filters
        all_filters = pha_filters + amp_filters
        max_len = max(f.shape[0] for f in all_filters)
        
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
        
        kernels = torch.stack(padded_filters)
        
        # Cache the result
        cls._filter_cache[cache_key] = kernels.clone()
        
        return kernels
    
    def _prepare_fft_filters(self):
        """Pre-compute FFT of filters for frequency domain filtering."""
        # Pad kernels to next power of 2 for efficient FFT
        kernel_len = self.kernels.shape[1]
        fft_size = 2 ** int(np.ceil(np.log2(self.seq_len + kernel_len - 1)))
        
        # Pad kernels
        pad_size = fft_size - kernel_len
        padded_kernels = torch.nn.functional.pad(
            self.kernels, (0, pad_size), mode='constant', value=0
        )
        
        # Compute FFT of kernels
        self.register_buffer("kernels_fft", torch.fft.rfft(padded_kernels, dim=-1))
        self.fft_size = fft_size
    
    def _fft_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply filtering in frequency domain using FFT."""
        batch_size = x.shape[0]
        n_filters = len(self.kernels)
        
        # Pad signal
        pad_size = self.fft_size - x.shape[-1]
        x_padded = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
        
        # FFT of signal
        x_fft = torch.fft.rfft(x_padded, dim=-1)
        
        # Expand for all filters
        x_fft_expanded = x_fft.unsqueeze(1).expand(-1, n_filters, -1)
        kernels_fft_expanded = self.kernels_fft.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multiply in frequency domain
        filtered_fft = x_fft_expanded * kernels_fft_expanded
        
        # IFFT back to time domain
        filtered = torch.fft.irfft(filtered_fft, n=self.fft_size, dim=-1)
        
        # Trim to original length
        filtered = filtered[:, :, :x.shape[-1]]
        
        return filtered
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optimized bandpass filtering."""
        # x shape: (batch*channel*segment, 1, time)
        
        # Apply edge padding if requested
        if self.edge_mode and self.padlen > 0:
            x = torch.nn.functional.pad(
                x, (self.padlen, self.padlen), mode=self.edge_mode
            )
        
        if self.use_fft and hasattr(self, 'kernels_fft'):
            # Use FFT-based filtering
            filtered = self._fft_filter(x.squeeze(1))
        elif self.filtfilt_mode:
            # Scipy-compatible implementation with odd extension padding
            batch_size = x.shape[0]
            seq_len = x.shape[-1]
            x_squeezed = x.squeeze(1)
            
            filtered_bands = []
            for i, kernel in enumerate(self.kernels):
                padlen = min(3 * len(kernel), seq_len - 1)
                
                x_padded_list = []
                for b in range(batch_size):
                    signal = x_squeezed[b]
                    
                    if padlen > 0:
                        left_pad = -signal[1:padlen+1].flip(0)
                        right_pad = -signal[-padlen-1:-1].flip(0) 
                        signal_padded = torch.cat([left_pad, signal, right_pad])
                    else:
                        signal_padded = signal
                    
                    x_padded_list.append(signal_padded)
                
                x_padded = torch.stack(x_padded_list).unsqueeze(1)
                
                kernel_3d = kernel.unsqueeze(0).unsqueeze(0)
                filtered_band = torch.nn.functional.conv1d(x_padded, kernel_3d, padding='same')
                filtered_band = torch.nn.functional.conv1d(
                    filtered_band.flip(-1), kernel_3d, padding='same'
                ).flip(-1)
                
                if padlen > 0:
                    filtered_band = filtered_band[:, :, padlen:-padlen]
                
                filtered_bands.append(filtered_band.squeeze(1))
            
            filtered = torch.stack(filtered_bands, dim=1)
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
    
    @classmethod
    def clear_cache(cls):
        """Clear the filter cache to free memory."""
        cls._filter_cache.clear()


# EOF