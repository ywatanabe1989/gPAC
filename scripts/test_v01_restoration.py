#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-26 11:00:00 (ywatanabe)"
# File: test_v01_restoration.py

"""
Test if restoring v01's depthwise convolution approach improves TensorPAC correlation.

This script creates a modified BandPassFilter that uses the v01 approach
and compares it with the current implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import sys
sys.path.append('./src')

from gpac._tensorpac_fir1 import design_filter_tensorpac


class BandPassFilterV01Style(nn.Module):
    """BandPassFilter using v01's depthwise convolution approach."""
    
    def __init__(self, bands, fs, seq_len, filtfilt_mode=True, edge_mode=None):
        super().__init__()
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode
        
        # Create filters
        filters = []
        for low_hz, high_hz in bands:
            kernel = design_filter_tensorpac(seq_len, fs, low_hz, high_hz)
            filters.append(kernel)
        
        # Find max length and pad
        max_len = max(f.shape[0] for f in filters)
        padded_filters = []
        for f in filters:
            pad_needed = max_len - f.shape[0]
            if pad_needed > 0:
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
                f_padded = torch.nn.functional.pad(f, (pad_left, pad_right))
            else:
                f_padded = f
            padded_filters.append(f_padded)
        
        # Stack filters
        kernels = torch.stack(padded_filters)
        self.register_buffer("kernels", kernels)
        
        # Calculate padlen
        self.padlen = max(len(f) for f in filters) - 1 if edge_mode else 0
    
    def forward(self, x):
        """Apply v01-style filtfilt using depthwise convolution."""
        # Apply edge padding if needed
        if self.edge_mode and self.padlen > 0:
            x = torch.nn.functional.pad(x, (self.padlen, self.padlen), mode=self.edge_mode)
        
        if self.filtfilt_mode:
            # V01 approach: expand input and use depthwise convolution
            x_expanded = x.expand(-1, len(self.kernels), -1)
            kernels_expanded = self.kernels.unsqueeze(1)
            
            # Forward pass
            filtered = torch.nn.functional.conv1d(
                x_expanded,
                kernels_expanded,
                padding="same",
                groups=len(self.kernels)  # Key: depthwise convolution
            )
            
            # Backward pass
            filtered = torch.nn.functional.conv1d(
                filtered.flip(-1),
                kernels_expanded,
                padding="same",
                groups=len(self.kernels)
            ).flip(-1)
        else:
            # Standard filtering
            filtered = torch.nn.functional.conv1d(
                x,
                self.kernels.unsqueeze(1),
                padding="same",
                groups=1
            )
        
        # Remove padding
        if self.edge_mode and self.padlen > 0:
            filtered = filtered[:, :, self.padlen:-self.padlen]
        
        return filtered.unsqueeze(1)


def test_v01_vs_current():
    """Compare v01-style filtering with current implementation."""
    
    # Test parameters
    fs = 512
    duration = 2.0
    seq_len = int(fs * duration)
    
    # Create test signal with known frequency components
    t = np.linspace(0, duration, seq_len)
    signal_10hz = np.sin(2 * np.pi * 10 * t)
    signal_80hz = np.sin(2 * np.pi * 80 * t)
    test_signal = signal_10hz + 0.5 * signal_80hz
    test_signal = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Define frequency bands
    phase_bands = torch.tensor([[8.0, 12.0], [12.0, 16.0]])
    amp_bands = torch.tensor([[60.0, 80.0], [80.0, 100.0]])
    all_bands = torch.cat([phase_bands, amp_bands])
    
    # Create filters
    v01_filter = BandPassFilterV01Style(all_bands, fs, seq_len, filtfilt_mode=True, edge_mode='reflect')
    
    # Import current implementation
    from gpac._BandPassFilter import BandPassFilter
    current_filter = BandPassFilter(phase_bands, amp_bands, fs, seq_len, filtfilt_mode=True, edge_mode='reflect')
    
    # Apply filters
    with torch.no_grad():
        v01_output = v01_filter(test_signal)
        current_output = current_filter(test_signal)
    
    # Compare outputs
    print("Output shapes:")
    print(f"V01 style: {v01_output.shape}")
    print(f"Current: {current_output.shape}")
    
    # Calculate correlation for each band
    print("\nCorrelations between v01 and current implementation:")
    for i in range(all_bands.shape[0]):
        v01_band = v01_output[0, 0, i, :].numpy()
        current_band = current_output[0, 0, i, :].numpy()
        
        # Calculate correlation
        correlation = np.corrcoef(v01_band, current_band)[0, 1]
        
        # Calculate RMS difference
        rms_diff = np.sqrt(np.mean((v01_band - current_band) ** 2))
        
        band_type = "Phase" if i < len(phase_bands) else "Amplitude"
        band_range = all_bands[i]
        
        print(f"{band_type} band {band_range.tolist()} Hz:")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  RMS difference: {rms_diff:.6f}")
    
    # Test with scipy for reference
    print("\nCorrelation with scipy.signal.filtfilt:")
    for i, (low, high) in enumerate(all_bands):
        # Design scipy filter
        nyq = fs / 2
        sos = signal.butter(4, [low/nyq, high/nyq], btype='band', output='sos')
        scipy_filtered = signal.sosfiltfilt(sos, test_signal.squeeze().numpy())
        
        v01_band = v01_output[0, 0, i, :].numpy()
        current_band = current_output[0, 0, i, :].numpy()
        
        v01_corr = np.corrcoef(v01_band, scipy_filtered)[0, 1]
        current_corr = np.corrcoef(current_band, scipy_filtered)[0, 1]
        
        print(f"Band {[low.item(), high.item()]} Hz:")
        print(f"  V01 vs scipy: {v01_corr:.6f}")
        print(f"  Current vs scipy: {current_corr:.6f}")


if __name__ == "__main__":
    print("Testing V01-style depthwise convolution approach...")
    print("=" * 60)
    test_v01_vs_current()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("The V01 approach uses depthwise convolution which is:")
    print("1. More efficient (all bands processed together)")
    print("2. Simpler implementation")
    print("3. May match TensorPAC's internal processing better")
    print("\nConsider adding a 'v01_mode' parameter to BandPassFilter")
    print("to allow users to choose between implementations.")

# EOF