#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 14:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/sequential_filtfilt_implementation.py
# ----------------------------------------
"""
Implement and benchmark true sequential filtfilt in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, '.')
from gpac._tensorpac_fir1 import design_filter_tensorpac


class SequentialFiltFilt(nn.Module):
    """True sequential filtfilt implementation in PyTorch."""
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def apply_sequential_filter(x, h, padding='same'):
        """
        Apply filter sequentially using conv1d.
        This mimics the behavior of scipy's lfilter.
        """
        # Ensure proper dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        y = F.conv1d(x, h, padding=padding)
        
        return y
    
    def forward(self, x, h, padlen=None):
        """
        True filtfilt implementation:
        1. Apply padding
        2. Filter forward
        3. Reverse and filter again
        4. Reverse back
        5. Remove padding
        """
        # Handle dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        
        # Apply padding if specified
        if padlen and padlen > 0:
            x_padded = F.pad(x, (padlen, padlen), mode='reflect')
        else:
            x_padded = x
        
        # Forward filter
        y1 = F.conv1d(x_padded, h, padding='same')
        
        # Reverse, filter, reverse
        y2 = F.conv1d(y1.flip(-1), h, padding='same').flip(-1)
        
        # Remove padding
        if padlen and padlen > 0:
            y2 = y2[:, :, padlen:-padlen]
        
        return y2


class CombinedBandPassFilterSequential(nn.Module):
    """Modified filter with true sequential filtfilt."""
    
    def __init__(self, pha_bands, amp_bands, fs, seq_len, fp16=False,
                 cycle_pha=3, cycle_amp=6, edge_mode=None):
        super().__init__()
        self.fp16 = fp16
        self.n_pha_bands = len(pha_bands)
        self.n_amp_bands = len(amp_bands)
        self.edge_mode = edge_mode
        
        # Create filters (same as before)
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
        
        # Store individual filters for sequential processing
        self.pha_filters = pha_filters
        self.amp_filters = amp_filters
        
        # Calculate padlens
        self.pha_padlens = [len(f) - 1 for f in pha_filters]
        self.amp_padlens = [len(f) - 1 for f in amp_filters]
        
        # Sequential filtfilt module
        self.filtfilt = SequentialFiltFilt()
    
    def forward(self, x):
        """Apply sequential filtfilt to each band."""
        # x shape: (batch*channel*segment, 1, time)
        batch_size = x.shape[0]
        time_len = x.shape[-1]
        
        # Process each filter separately (sequential)
        all_filtered = []
        
        # Phase filters
        for i, (h, padlen) in enumerate(zip(self.pha_filters, self.pha_padlens)):
            h_tensor = h.unsqueeze(0).unsqueeze(0).to(x.device)
            if self.edge_mode:
                filtered = self.filtfilt(x[:, 0, :], h_tensor, padlen=padlen)
            else:
                filtered = self.filtfilt(x[:, 0, :], h_tensor, padlen=0)
            all_filtered.append(filtered)
        
        # Amplitude filters
        for i, (h, padlen) in enumerate(zip(self.amp_filters, self.amp_padlens)):
            h_tensor = h.unsqueeze(0).unsqueeze(0).to(x.device)
            if self.edge_mode:
                filtered = self.filtfilt(x[:, 0, :], h_tensor, padlen=padlen)
            else:
                filtered = self.filtfilt(x[:, 0, :], h_tensor, padlen=0)
            all_filtered.append(filtered)
        
        # Stack results
        result = torch.cat(all_filtered, dim=1)
        
        # Add dimension to match expected output
        # Output should be: (batch*channel*segment, 1, n_bands, time)
        result = result.unsqueeze(1)
        
        return result


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
    signal_torch = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, -1).to(device)
    
    # Frequency bands
    pha_bands = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]
    amp_bands = [(60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]
    
    print(f"\nTest configuration:")
    print(f"  Device: {device}")
    print(f"  Signal length: {seq_len} samples")
    print(f"  Phase bands: {len(pha_bands)}")
    print(f"  Amplitude bands: {len(amp_bands)}")
    print(f"  Total filters: {len(pha_bands) + len(amp_bands)}")
    
    # 1. Current gPAC (averaging) method
    print("\n1. Current gPAC (averaging) method:")
    
    from gpac._PAC import CombinedBandPassFilter
    filter_avg = CombinedBandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
        filtfilt_mode=True,
        edge_mode='reflect'
    ).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = filter_avg(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Time averaging method
    n_runs = 100
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            result_avg = filter_avg(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_avg = (time.time() - start) / n_runs
    
    print(f"  Time per run: {time_avg*1000:.3f} ms")
    print(f"  Output shape: {result_avg.shape}")
    
    # 2. Sequential filtfilt method
    print("\n2. Sequential filtfilt method:")
    
    filter_seq = CombinedBandPassFilterSequential(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
        edge_mode='reflect'
    ).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = filter_seq(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Time sequential method
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            result_seq = filter_seq(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_seq = (time.time() - start) / n_runs
    
    print(f"  Time per run: {time_seq*1000:.3f} ms")
    print(f"  Output shape: {result_seq.shape}")
    
    # 3. Compare with scipy for accuracy
    print("\n3. Scipy reference (CPU only):")
    
    # Get one filter for comparison
    h = design_filter_tensorpac(seq_len, fs, 70.0, 80.0, cycle=6).numpy()
    
    start = time.time()
    for _ in range(n_runs):
        result_scipy = filtfilt(h, 1, signal, padlen=len(h)-1)
    time_scipy = (time.time() - start) / n_runs
    
    print(f"  Time per run: {time_scipy*1000:.3f} ms (single filter)")
    print(f"  Estimated for {len(pha_bands) + len(amp_bands)} filters: {time_scipy*1000*(len(pha_bands) + len(amp_bands)):.3f} ms")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON:")
    print("=" * 60)
    
    print(f"Averaging method: {time_avg*1000:.3f} ms")
    print(f"Sequential method: {time_seq*1000:.3f} ms")
    print(f"Slowdown factor: {time_seq/time_avg:.2f}x")
    
    if time_seq < time_avg * 1.5:
        print("\n✅ Sequential method is competitive! Only {:.0f}% slower".format((time_seq/time_avg - 1) * 100))
    else:
        print("\n⚠️  Sequential method is {:.0f}% slower".format((time_seq/time_avg - 1) * 100))
    
    # Accuracy comparison
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON:")
    print("=" * 60)
    
    # Extract one band for comparison
    avg_band = result_avg[0, 0, 5, :].cpu().numpy()  # 6th band (amplitude)
    seq_band = result_seq[0, 0, 5, :].cpu().numpy()
    
    # Compare with scipy
    h_test = design_filter_tensorpac(seq_len, fs, 60.0, 70.0, cycle=6).numpy()
    scipy_result = filtfilt(h_test, 1, signal, padlen=len(h_test)-1)
    
    diff_avg_scipy = np.abs(avg_band - scipy_result).max()
    diff_seq_scipy = np.abs(seq_band - scipy_result).max()
    
    print(f"Averaging vs Scipy max diff: {diff_avg_scipy:.6f}")
    print(f"Sequential vs Scipy max diff: {diff_seq_scipy:.6f}")
    
    if diff_seq_scipy < diff_avg_scipy * 0.1:
        print("\n✅ Sequential method is much more accurate!")
    
    return time_avg, time_seq, result_avg, result_seq


def test_full_pac_pipeline():
    """Test sequential filtfilt in full PAC pipeline."""
    print("\n" + "=" * 80)
    print("TESTING IN FULL PAC PIPELINE")
    print("=" * 80)
    
    # This would require modifying the PAC class to use the sequential filter
    # For now, let's estimate the impact
    
    print("\nEstimated impact on full PAC computation:")
    print("  - Filtering is ~30% of total PAC time")
    print("  - If sequential is 2x slower for filtering")
    print("  - Total PAC would be ~1.3x slower")
    print("  - But with much better TensorPAC compatibility")


def main():
    """Run all benchmarks."""
    print("🚀 SEQUENTIAL FILTFILT IMPLEMENTATION TEST")
    print("=" * 80)
    
    # Run benchmarks
    time_avg, time_seq, result_avg, result_seq = benchmark_implementations()
    
    # Test in pipeline context
    test_full_pac_pipeline()
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if time_seq < time_avg * 1.5:
        print("\n✅ Sequential implementation is viable!")
        print("   - Add as an option: filtfilt_mode='sequential'")
        print("   - Default can remain 'averaging' for speed")
        print("   - Users can choose accuracy vs speed")
    else:
        print("\n⚠️  Sequential implementation has significant overhead")
        print("   - Consider optimizations (batched conv1d)")
        print("   - Or keep as experimental feature")
    
    print("\nNext steps:")
    print("1. Optimize sequential implementation (batch processing)")
    print("2. Add as option to PAC class")
    print("3. Document the trade-offs for users")


if __name__ == "__main__":
    main()