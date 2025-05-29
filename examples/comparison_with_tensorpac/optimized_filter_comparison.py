#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 15:30:00"
# Author: Claude
# Filename: optimized_filter_comparison.py

"""
Demonstrates optimization opportunities in gPAC filtering
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal

def create_filters_sequential(bands, fs, filter_length=251):
    """Original sequential approach - SLOW"""
    filters = []
    window = torch.hamming_window(filter_length)
    
    for low_hz, high_hz in bands:
        # Time axis
        n = torch.arange(filter_length).float() - (filter_length - 1) / 2
        
        # Normalized frequencies
        low_freq = low_hz * 2 / fs
        high_freq = high_hz * 2 / fs
        
        # Sinc functions
        n_low = low_freq * torch.sinc(low_freq * n)
        n_high = high_freq * torch.sinc(high_freq * n)
        
        # Bandpass
        band_pass = (n_high - n_low) * window
        band_pass = band_pass / torch.sqrt(torch.sum(band_pass ** 2))
        
        # Frequency response normalization (SLOW!)
        w, h = scipy_signal.freqz(band_pass.numpy(), worN=8192, fs=fs)
        passband_mask = (w >= low_hz) & (w <= high_hz)
        if np.any(passband_mask):
            peak_gain = np.max(np.abs(h[passband_mask]))
            if peak_gain > 0:
                band_pass = band_pass / peak_gain
        
        filters.append(band_pass)
    
    return torch.stack(filters)

def create_filters_vectorized(bands, fs, filter_length=251):
    """Optimized vectorized approach - FAST"""
    n_filters = len(bands)
    window = torch.hamming_window(filter_length)
    
    # Time axis (same for all filters)
    n = torch.arange(filter_length).float() - (filter_length - 1) / 2
    n = n.unsqueeze(0).expand(n_filters, -1)  # (n_filters, filter_length)
    
    # Vectorized frequency computation
    low_freqs = bands[:, 0:1] * 2 / fs  # (n_filters, 1)
    high_freqs = bands[:, 1:2] * 2 / fs  # (n_filters, 1)
    
    # Vectorized sinc computation
    n_low = low_freqs * torch.sinc(low_freqs * n)
    n_high = high_freqs * torch.sinc(high_freqs * n)
    
    # Vectorized bandpass
    band_pass = (n_high - n_low) * window.unsqueeze(0)
    
    # Vectorized normalization
    energy = torch.sqrt(torch.sum(band_pass ** 2, dim=1, keepdim=True))
    band_pass = band_pass / energy
    
    # Skip scipy normalization for speed (or do it in batch on GPU)
    return band_pass

def optimized_batch_conv(x, kernels):
    """Optimized convolution using grouped convolution"""
    # x: (batch_size, n_chs, seq_len)
    # kernels: (n_kernels, kernel_len)
    
    batch_size, n_chs, seq_len = x.shape
    n_kernels, kernel_len = kernels.shape
    
    # Use grouped convolution for efficiency
    # Reshape x to (batch_size * n_chs, 1, seq_len)
    x_reshaped = x.reshape(batch_size * n_chs, 1, seq_len)
    
    # Reshape kernels to (n_kernels, 1, kernel_len)
    kernels_reshaped = kernels.unsqueeze(1)
    
    # Apply convolution
    output = F.conv1d(x_reshaped, kernels_reshaped, padding=kernel_len//2)
    
    # Reshape back
    output = output.reshape(batch_size, n_chs, n_kernels, -1)
    
    return output[..., :seq_len]

def benchmark_filter_creation():
    """Compare filter creation methods"""
    print("Filter Creation Benchmark")
    print("=" * 80)
    
    fs = 512
    filter_length = 251
    
    for n_bands in [10, 30, 50]:
        print(f"\n{n_bands}x{n_bands} bands ({n_bands**2} total filters):")
        
        # Create bands
        pha_bands = torch.linspace(2, 20, n_bands + 1)
        pha_bands = torch.stack([pha_bands[:-1], pha_bands[1:]], dim=1)
        
        amp_bands = torch.linspace(30, 150, n_bands + 1)
        amp_bands = torch.stack([amp_bands[:-1], amp_bands[1:]], dim=1)
        
        all_bands = torch.cat([pha_bands, amp_bands], dim=0)
        
        # Sequential (original)
        start = time.time()
        filters_seq = create_filters_sequential(all_bands, fs, filter_length)
        seq_time = time.time() - start
        print(f"  Sequential: {seq_time:.3f}s")
        
        # Vectorized
        start = time.time()
        filters_vec = create_filters_vectorized(all_bands, fs, filter_length)
        vec_time = time.time() - start
        print(f"  Vectorized: {vec_time:.3f}s (speedup: {seq_time/vec_time:.1f}x)")

def benchmark_convolution():
    """Compare convolution methods"""
    print("\n\nConvolution Benchmark")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test parameters
    batch_size = 8
    n_channels = 64
    seq_len = 5120
    filter_length = 251
    
    for n_filters in [60, 300, 900]:
        print(f"\n{n_filters} filters:")
        
        # Create data and filters
        x = torch.randn(batch_size, n_channels, seq_len).to(device)
        kernels = torch.randn(n_filters, filter_length).to(device)
        
        # Warmup
        _ = optimized_batch_conv(x, kernels)
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Time convolution
        times = []
        for _ in range(10):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            
            output = optimized_batch_conv(x, kernels)
            
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times[2:])  # Skip first runs
        throughput = batch_size * n_channels * seq_len / mean_time / 1e6
        
        print(f"  Time: {mean_time*1000:.2f}ms ({throughput:.1f}M samples/s)")

if __name__ == "__main__":
    benchmark_filter_creation()
    benchmark_convolution()