#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 14:55:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/sequential_filtfilt_fixed.py
# ----------------------------------------
"""
Fixed implementation of sequential filtfilt with proper dimension handling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.signal import filtfilt

import sys
sys.path.insert(0, '.')
from gpac._tensorpac_fir1 import design_filter_tensorpac


def sequential_filtfilt_torch(x, h, padlen=0):
    """
    True sequential filtfilt implementation in PyTorch.
    Mimics scipy.signal.filtfilt behavior.
    """
    # Get device
    device = x.device if hasattr(x, 'device') else 'cpu'
    
    # Ensure 3D tensors (batch, channel, time)
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)
    
    if isinstance(h, np.ndarray):
        h = torch.tensor(h, dtype=torch.float32, device=device)
    elif isinstance(h, torch.Tensor):
        h = h.to(device)
    
    if h.dim() == 1:
        h = h.flip(0).unsqueeze(0).unsqueeze(0)  # Flip for convolution
    
    # Apply padding
    if padlen > 0:
        x_padded = F.pad(x, (padlen, padlen), mode='reflect')
    else:
        x_padded = x
    
    # Forward filter
    y1 = F.conv1d(x_padded, h, padding='same')
    
    # Backward filter (filter the time-reversed signal)
    y2 = F.conv1d(y1.flip(-1), h, padding='same').flip(-1)
    
    # Remove padding
    if padlen > 0:
        y2 = y2[:, :, padlen:-padlen]
    
    return y2.squeeze(1) if y2.shape[1] == 1 else y2


def batch_sequential_filtfilt(x, filters, padlens):
    """
    Apply multiple filters to the same signal in a batched way.
    More efficient than loop.
    """
    batch_size = x.shape[0]
    n_filters = len(filters)
    time_len = x.shape[-1]
    device = x.device
    
    # Stack all filters
    max_len = max(f.shape[0] if hasattr(f, 'shape') else len(f) for f in filters)
    h_padded = []
    for f in filters:
        if isinstance(f, np.ndarray):
            f_tensor = torch.tensor(f, dtype=torch.float32, device=device).flip(0)
        elif isinstance(f, torch.Tensor):
            f_tensor = f.to(device).flip(0)
        else:
            f_tensor = torch.tensor(f, dtype=torch.float32, device=device).flip(0)
            
        if len(f_tensor) < max_len:
            pad_total = max_len - len(f_tensor)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            f_padded = F.pad(f_tensor, (pad_left, pad_right))
        else:
            f_padded = f_tensor
        h_padded.append(f_padded)
    
    h_batch = torch.stack(h_padded).unsqueeze(1)  # (n_filters, 1, kernel_len)
    
    # Expand input for all filters
    x_expanded = x.unsqueeze(1).expand(-1, n_filters, -1)  # (batch, n_filters, time)
    
    # Apply padding (use max padlen for simplicity)
    max_padlen = max(padlens)
    if max_padlen > 0:
        x_padded = F.pad(x_expanded, (max_padlen, max_padlen), mode='reflect')
    else:
        x_padded = x_expanded
    
    # Reshape for grouped convolution
    x_reshaped = x_padded.reshape(batch_size * n_filters, 1, -1)
    h_reshaped = h_batch.repeat(batch_size, 1, 1)
    
    # Forward pass
    y1 = F.conv1d(x_reshaped, h_batch, padding='same', groups=1)
    
    # Backward pass
    y2 = F.conv1d(y1.flip(-1), h_batch, padding='same', groups=1).flip(-1)
    
    # Reshape back
    y2 = y2.reshape(batch_size, n_filters, -1)
    
    # Remove padding
    if max_padlen > 0:
        y2 = y2[:, :, max_padlen:-max_padlen]
    
    return y2


def benchmark_implementations():
    """Benchmark different filtfilt implementations."""
    print("=" * 80)
    print("BENCHMARKING FILTFILT IMPLEMENTATIONS")
    print("=" * 80)
    
    # Test parameters
    fs = 512.0
    duration = 5.0
    seq_len = int(fs * duration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test signal
    t = np.linspace(0, duration, seq_len)
    signal = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)
    signal += np.random.normal(0, 0.1, len(t))
    signal_torch = torch.tensor(signal, dtype=torch.float32).to(device)
    
    # Create filters
    n_phase_bands = 10
    n_amp_bands = 10
    
    pha_filters = []
    amp_filters = []
    
    # Phase filters
    for i in range(n_phase_bands):
        f_low = 2 + i * 1.8
        f_high = f_low + 3.6
        h = design_filter_tensorpac(seq_len, fs, f_low, f_high, cycle=3)
        pha_filters.append(h)
    
    # Amplitude filters  
    for i in range(n_amp_bands):
        f_low = 60 + i * 6
        f_high = f_low + 12
        h = design_filter_tensorpac(seq_len, fs, f_low, f_high, cycle=6)
        amp_filters.append(h)
    
    all_filters = pha_filters + amp_filters
    padlens = [len(f) - 1 for f in all_filters]
    
    print(f"\nTest configuration:")
    print(f"  Device: {device}")
    print(f"  Signal length: {seq_len} samples")
    print(f"  Phase bands: {n_phase_bands}")
    print(f"  Amplitude bands: {n_amp_bands}")
    print(f"  Total filters: {len(all_filters)}")
    
    # Method 1: Current gPAC averaging
    print("\n1. Current gPAC (averaging) method:")
    
    n_runs = 50
    signal_batch = signal_torch.unsqueeze(0).unsqueeze(0)
    
    # Time averaging method
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            results_avg = []
            for h, padlen in zip(all_filters, padlens):
                if isinstance(h, np.ndarray):
                    h_tensor = torch.tensor(h, dtype=torch.float32, device=device)
                else:
                    h_tensor = h.to(device)
                h_tensor = h_tensor.flip(0).unsqueeze(0).unsqueeze(0)
                # Apply padding
                if padlen > 0:
                    x_pad = F.pad(signal_batch, (padlen, padlen), mode='reflect')
                else:
                    x_pad = signal_batch
                # Forward and backward
                y_fwd = F.conv1d(x_pad, h_tensor, padding='same')
                y_bwd = F.conv1d(x_pad.flip(-1), h_tensor, padding='same').flip(-1)
                y_avg = (y_fwd + y_bwd) / 2.0
                # Remove padding
                if padlen > 0:
                    y_avg = y_avg[:, :, padlen:-padlen]
                results_avg.append(y_avg)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_avg = (time.time() - start) / n_runs
    print(f"  Time per run: {time_avg*1000:.3f} ms")
    
    # Method 2: Sequential filtfilt (loop)
    print("\n2. Sequential filtfilt (loop):")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            results_seq = []
            for h, padlen in zip(all_filters, padlens):
                y = sequential_filtfilt_torch(signal_torch, h, padlen)
                results_seq.append(y)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_seq_loop = (time.time() - start) / n_runs
    print(f"  Time per run: {time_seq_loop*1000:.3f} ms")
    print(f"  Slowdown vs averaging: {time_seq_loop/time_avg:.2f}x")
    
    # Method 3: Batched sequential filtfilt
    print("\n3. Batched sequential filtfilt:")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            result_batch = batch_sequential_filtfilt(
                signal_torch.unsqueeze(0), all_filters, padlens
            )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_seq_batch = (time.time() - start) / n_runs
    print(f"  Time per run: {time_seq_batch*1000:.3f} ms")
    print(f"  Slowdown vs averaging: {time_seq_batch/time_avg:.2f}x")
    
    # Method 4: Scipy reference
    print("\n4. Scipy filtfilt (CPU):")
    
    signal_np = signal_torch.cpu().numpy()
    start = time.time()
    
    for _ in range(n_runs):
        results_scipy = []
        for h, padlen in zip(all_filters, padlens):
            h_np = h.cpu().numpy() if isinstance(h, torch.Tensor) else h
            y = filtfilt(h_np, 1, signal_np, padlen=padlen)
            results_scipy.append(y)
    
    time_scipy = (time.time() - start) / n_runs
    print(f"  Time per run: {time_scipy*1000:.3f} ms")
    
    # Accuracy check
    print("\n" + "=" * 60)
    print("ACCURACY CHECK:")
    print("=" * 60)
    
    # Compare one filter result
    test_idx = 5
    h_test = all_filters[test_idx]
    h_np = h_test.cpu().numpy() if isinstance(h_test, torch.Tensor) else h_test
    padlen_test = padlens[test_idx]
    
    # Scipy reference
    y_scipy = filtfilt(h_np, 1, signal_np, padlen=padlen_test)
    
    # Sequential result
    y_seq = sequential_filtfilt_torch(signal_torch, h_test, padlen_test).cpu().numpy()
    
    diff = np.abs(y_scipy - y_seq)
    print(f"Sequential vs Scipy max difference: {diff.max():.6e}")
    print(f"Sequential vs Scipy mean difference: {diff.mean():.6e}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY:")
    print("=" * 60)
    
    print(f"\nMethod                Time (ms)   Relative to Averaging")
    print("-" * 55)
    print(f"Averaging            {time_avg*1000:8.2f}          1.00x")
    print(f"Sequential (loop)    {time_seq_loop*1000:8.2f}          {time_seq_loop/time_avg:.2f}x")
    print(f"Sequential (batch)   {time_seq_batch*1000:8.2f}          {time_seq_batch/time_avg:.2f}x")
    print(f"Scipy (CPU)          {time_scipy*1000:8.2f}          {time_scipy/time_avg:.2f}x")
    
    return time_avg, time_seq_loop, time_seq_batch


def estimate_pac_impact(time_avg, time_seq):
    """Estimate impact on full PAC computation."""
    print("\n" + "=" * 60)
    print("ESTIMATED IMPACT ON PAC:")
    print("=" * 60)
    
    # Typical PAC timing breakdown (rough estimates)
    filtering_fraction = 0.3  # 30% of time
    hilbert_fraction = 0.2   # 20% of time
    mi_fraction = 0.4        # 40% of time
    other_fraction = 0.1     # 10% of time
    
    print(f"\nAssuming PAC time breakdown:")
    print(f"  Filtering: {filtering_fraction*100:.0f}%")
    print(f"  Hilbert: {hilbert_fraction*100:.0f}%")
    print(f"  MI calculation: {mi_fraction*100:.0f}%")
    print(f"  Other: {other_fraction*100:.0f}%")
    
    slowdown_factor = time_seq / time_avg
    
    # New filtering time
    new_filtering_fraction = filtering_fraction * slowdown_factor
    total_new_fraction = new_filtering_fraction + hilbert_fraction + mi_fraction + other_fraction
    
    overall_slowdown = total_new_fraction / 1.0
    
    print(f"\nWith {slowdown_factor:.1f}x slower filtering:")
    print(f"  Overall PAC slowdown: {overall_slowdown:.2f}x")
    print(f"  If PAC takes 100ms → {100*overall_slowdown:.0f}ms with sequential")
    
    if overall_slowdown < 1.5:
        print("\n✅ Sequential filtfilt is practical for PAC!")
    else:
        print("\n⚠️  Sequential filtfilt adds significant overhead")


def main():
    """Run all tests."""
    print("🚀 SEQUENTIAL FILTFILT IMPLEMENTATION TEST")
    print("=" * 80)
    
    # Run benchmarks
    time_avg, time_seq_loop, time_seq_batch = benchmark_implementations()
    
    # Estimate PAC impact
    best_seq_time = min(time_seq_loop, time_seq_batch)
    estimate_pac_impact(time_avg, best_seq_time)
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if best_seq_time < time_avg * 2:
        print("\n✅ Sequential implementation is viable!")
        print("\nImplementation plan:")
        print("1. Add filtfilt_mode options: 'averaging' (default), 'sequential'")
        print("2. Use batched implementation for efficiency")
        print("3. Document accuracy vs speed trade-off")
        print("\nExample usage:")
        print("  pac = gpac.PAC(..., filtfilt_mode='sequential')  # Exact scipy match")
        print("  pac = gpac.PAC(..., filtfilt_mode='averaging')   # Faster (default)")
    else:
        print("\n⚠️  Consider optimization strategies:")
        print("1. Custom CUDA kernel for sequential filtering")
        print("2. Hybrid approach for critical bands only")
        print("3. Keep as experimental feature")


if __name__ == "__main__":
    main()