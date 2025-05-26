#!/usr/bin/env python3
"""
Test optimized BandPassFilter performance.
"""

import torch
import time
import sys
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')

from gpac._BandPassFilter import BandPassFilter
from gpac._OptimizedBandPassFilter import OptimizedBandPassFilter

def benchmark_filters():
    """Compare original vs optimized filter performance."""
    
    # Test parameters
    fs = 512.0
    seq_len = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pha_bands = torch.tensor([[2.0, 4.0], [4.0, 8.0], [8.0, 12.0]])
    amp_bands = torch.tensor([[30.0, 50.0], [50.0, 80.0], [80.0, 120.0]])
    
    print(f"Device: {device}")
    print(f"Filters: {len(pha_bands) + len(amp_bands)} total")
    print("-" * 50)
    
    # Test initialization time
    print("\n1. Filter initialization time:")
    
    # Original filter
    start = time.time()
    filter_original = BandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
    ).to(device)
    time_original_init = (time.time() - start) * 1000
    print(f"  Original: {time_original_init:.2f} ms")
    
    # Optimized filter (first time - no cache)
    start = time.time()
    filter_optimized = OptimizedBandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
    ).to(device)
    time_optimized_init_cold = (time.time() - start) * 1000
    print(f"  Optimized (cold): {time_optimized_init_cold:.2f} ms")
    
    # Optimized filter (second time - with cache)
    start = time.time()
    filter_optimized_cached = OptimizedBandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
    ).to(device)
    time_optimized_init_warm = (time.time() - start) * 1000
    print(f"  Optimized (cached): {time_optimized_init_warm:.2f} ms")
    print(f"  Speedup: {time_original_init / time_optimized_init_warm:.2f}x")
    
    # Test FFT mode
    start = time.time()
    filter_fft = OptimizedBandPassFilter(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=seq_len,
        use_fft=True,
    ).to(device)
    time_fft_init = (time.time() - start) * 1000
    print(f"  FFT mode: {time_fft_init:.2f} ms")
    
    # Test forward pass performance
    print("\n2. Forward pass performance (avg of 100 runs):")
    
    batch_sizes = [1, 4, 16, 32]
    x_test = {bs: torch.randn(bs, 1, seq_len).to(device) for bs in batch_sizes}
    
    for bs in batch_sizes:
        x = x_test[bs]
        
        # Warmup
        for _ in range(10):
            _ = filter_original(x)
            _ = filter_optimized(x)
            _ = filter_fft(x)
        
        # Original
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = filter_original(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_original = (time.time() - start) / 100 * 1000
        
        # Optimized
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = filter_optimized(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_optimized = (time.time() - start) / 100 * 1000
        
        # FFT
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = filter_fft(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        time_fft = (time.time() - start) / 100 * 1000
        
        print(f"\n  Batch size {bs}:")
        print(f"    Original: {time_original:.2f} ms")
        print(f"    Optimized: {time_optimized:.2f} ms (speedup: {time_original/time_optimized:.2f}x)")
        print(f"    FFT mode: {time_fft:.2f} ms (speedup: {time_original/time_fft:.2f}x)")
    
    # Test accuracy
    print("\n3. Accuracy verification:")
    x = torch.randn(1, 1, seq_len).to(device)
    
    out_original = filter_original(x)
    out_optimized = filter_optimized(x)
    out_fft = filter_fft(x)
    
    diff_optimized = torch.abs(out_original - out_optimized).max()
    diff_fft = torch.abs(out_original - out_fft).max()
    
    print(f"  Max diff (optimized vs original): {diff_optimized:.6f}")
    print(f"  Max diff (FFT vs original): {diff_fft:.6f}")
    
    # Clear cache
    OptimizedBandPassFilter.clear_cache()
    print("\n✅ Cache cleared")

if __name__ == "__main__":
    print("Optimized BandPassFilter Performance Test")
    print("=" * 50)
    benchmark_filters()