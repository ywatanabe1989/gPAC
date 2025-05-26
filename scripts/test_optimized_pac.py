#!/usr/bin/env python3
"""
Test optimized PAC performance vs original.
"""

import torch
import time
import sys
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')

from gpac import calculate_pac

def benchmark_pac():
    """Compare original vs optimized PAC performance."""
    
    # Test parameters
    fs = 512.0
    duration = 2.0
    seq_len = int(fs * duration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Signal length: {seq_len} samples")
    print("-" * 50)
    
    # Generate test signal
    torch.manual_seed(42)
    signal = torch.randn(1, 1, 1, seq_len).to(device)
    
    # Test original (no optimization)
    print("\n1. Original PAC (no filter optimization):")
    start = time.time()
    pac_original, _, _ = calculate_pac(
        signal,
        fs=fs,
        pha_n_bands=10,
        amp_n_bands=10,
        device=device,
        use_optimized_filter=False
    )
    time_original = (time.time() - start) * 1000
    print(f"  Time: {time_original:.2f} ms")
    
    # Test optimized (first run - cold cache)
    print("\n2. Optimized PAC (cold cache):")
    start = time.time()
    pac_optimized_cold, _, _ = calculate_pac(
        signal,
        fs=fs,
        pha_n_bands=10,
        amp_n_bands=10,
        device=device,
        use_optimized_filter=True
    )
    time_optimized_cold = (time.time() - start) * 1000
    print(f"  Time: {time_optimized_cold:.2f} ms")
    print(f"  Speedup: {time_original/time_optimized_cold:.2f}x")
    
    # Test optimized (second run - warm cache)
    print("\n3. Optimized PAC (warm cache):")
    start = time.time()
    pac_optimized_warm, _, _ = calculate_pac(
        signal,
        fs=fs,
        pha_n_bands=10,
        amp_n_bands=10,
        device=device,
        use_optimized_filter=True
    )
    time_optimized_warm = (time.time() - start) * 1000
    print(f"  Time: {time_optimized_warm:.2f} ms")
    print(f"  Speedup: {time_original/time_optimized_warm:.2f}x")
    
    # Verify accuracy
    print("\n4. Accuracy verification:")
    diff = torch.abs(pac_original - pac_optimized_warm).max()
    print(f"  Max difference: {diff:.6f}")
    
    # Test with multiple frequency bands
    print("\n5. Performance with more frequency bands:")
    
    for n_bands in [20, 50]:
        print(f"\n  {n_bands} phase x {n_bands} amplitude bands:")
        
        # Original
        start = time.time()
        _, _, _ = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=n_bands,
            amp_n_bands=n_bands,
            device=device,
            use_optimized_filter=False
        )
        time_orig = (time.time() - start) * 1000
        
        # Optimized (warm cache)
        start = time.time()
        _, _, _ = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=n_bands,
            amp_n_bands=n_bands,
            device=device,
            use_optimized_filter=True
        )
        time_opt = (time.time() - start) * 1000
        
        print(f"    Original: {time_orig:.2f} ms")
        print(f"    Optimized: {time_opt:.2f} ms")
        print(f"    Speedup: {time_orig/time_opt:.2f}x")
    
    # Test batch processing
    print("\n6. Batch processing efficiency:")
    batch_signal = torch.randn(8, 1, 1, seq_len).to(device)
    
    # Original
    start = time.time()
    _, _, _ = calculate_pac(
        batch_signal,
        fs=fs,
        pha_n_bands=10,
        amp_n_bands=10,
        device=device,
        use_optimized_filter=False
    )
    time_batch_orig = (time.time() - start) * 1000
    
    # Optimized
    start = time.time()
    _, _, _ = calculate_pac(
        batch_signal,
        fs=fs,
        pha_n_bands=10,
        amp_n_bands=10,
        device=device,
        use_optimized_filter=True
    )
    time_batch_opt = (time.time() - start) * 1000
    
    print(f"  Batch size 8:")
    print(f"    Original: {time_batch_orig:.2f} ms ({time_batch_orig/8:.2f} ms/sample)")
    print(f"    Optimized: {time_batch_opt:.2f} ms ({time_batch_opt/8:.2f} ms/sample)")
    print(f"    Speedup: {time_batch_orig/time_batch_opt:.2f}x")

if __name__ == "__main__":
    print("Optimized PAC Performance Test")
    print("=" * 50)
    benchmark_pac()
    print("\n✅ Testing complete!")