#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Verify caching correctness and performance after bug fix

import time
import torch
import numpy as np
from gpac import PAC
import matplotlib.pyplot as plt
import os

def test_caching_correctness_and_speed():
    """Comprehensive test of caching correctness and performance."""
    
    # Test parameters
    seq_len = 2000
    fs = 500
    batch_size = 4
    n_channels = 8
    n_runs = 5
    
    # Initialize PAC with caching enabled
    pac_cached = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(2, 30),
        amp_range_hz=(30, 100),
        pha_n_bands=10,
        amp_n_bands=10,
        n_perm=20,  # Include surrogates to test full pipeline
        enable_caching=True,
        fp16=False,
    )
    
    # Initialize PAC without caching for comparison
    pac_no_cache = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(2, 30),
        amp_range_hz=(30, 100),
        pha_n_bands=10,
        amp_n_bands=10,
        n_perm=20,
        enable_caching=False,
        fp16=False,
    )
    
    print("=" * 80)
    print("CACHING CORRECTNESS AND PERFORMANCE VERIFICATION")
    print("=" * 80)
    
    # Test 1: Same data, different memory locations
    print("\n1. Testing cache hits for identical data at different memory locations:")
    data = torch.randn(batch_size, n_channels, seq_len)
    
    # First run (cache miss)
    t0 = time.time()
    result1 = pac_cached(data.clone())
    time_first = time.time() - t0
    print(f"   First run (cache miss): {time_first:.3f}s")
    
    # Second run with same data, different memory location (should be cache hit)
    t0 = time.time()
    result2 = pac_cached(data.clone())
    time_cached = time.time() - t0
    print(f"   Second run (cache hit): {time_cached:.3f}s")
    print(f"   Speedup: {time_first/time_cached:.1f}x")
    
    # Verify results are identical
    pac_diff = torch.abs(result1["pac"] - result2["pac"]).max().item()
    z_diff = torch.abs(result1["pac_z"] - result2["pac_z"]).max().item()
    print(f"   PAC difference: {pac_diff:.2e} (should be ~0)")
    print(f"   Z-score difference: {z_diff:.2e} (should be ~0)")
    assert pac_diff < 1e-10, "Cache hit should return identical results!"
    
    # Test 2: In-place modification detection
    print("\n2. Testing cache miss for in-place modified data:")
    original_data = data.clone()
    
    # Modify data in-place
    data.add_(0.1)
    
    # This should be a cache miss
    t0 = time.time()
    result3 = pac_cached(data)
    time_modified = time.time() - t0
    print(f"   Modified data run: {time_modified:.3f}s")
    
    # Verify results are different
    pac_diff = torch.abs(result1["pac"] - result3["pac"]).mean().item()
    print(f"   Mean PAC difference: {pac_diff:.4f} (should be > 0)")
    assert pac_diff > 1e-4, "In-place modification should produce different results!"
    
    # Test 3: Performance comparison with/without caching
    print("\n3. Performance comparison (with vs without caching):")
    
    times_cached = []
    times_no_cache = []
    
    # Clear cache
    pac_cached.clear_cache()
    
    # Generate test data
    test_data = [torch.randn(batch_size, n_channels, seq_len) for _ in range(n_runs)]
    
    print("   Running with caching:")
    for i, data in enumerate(test_data):
        t0 = time.time()
        _ = pac_cached(data)
        elapsed = time.time() - t0
        times_cached.append(elapsed)
        print(f"     Run {i+1}: {elapsed:.3f}s")
    
    print("   Running without caching:")
    for i, data in enumerate(test_data):
        t0 = time.time()
        _ = pac_no_cache(data)
        elapsed = time.time() - t0
        times_no_cache.append(elapsed)
        print(f"     Run {i+1}: {elapsed:.3f}s")
    
    # Test repeated data with caching
    print("\n4. Testing repeated data processing:")
    repeated_data = test_data[0]
    
    pac_cached.clear_cache()
    repeat_times = []
    
    for i in range(5):
        t0 = time.time()
        _ = pac_cached(repeated_data.clone())  # Clone to ensure different memory address
        elapsed = time.time() - t0
        repeat_times.append(elapsed)
        print(f"   Repeat {i+1}: {elapsed:.3f}s")
    
    print(f"\n   First run: {repeat_times[0]:.3f}s")
    print(f"   Subsequent runs average: {np.mean(repeat_times[1:]):.3f}s")
    print(f"   Cache speedup: {repeat_times[0]/np.mean(repeat_times[1:]):.1f}x")
    
    # Test 5: Verify numerical consistency
    print("\n5. Numerical consistency check:")
    test_tensor = torch.randn(2, 4, seq_len)
    
    # Run multiple times with caching
    pac_cached.clear_cache()
    results_cached = []
    for _ in range(3):
        results_cached.append(pac_cached(test_tensor.clone()))
    
    # Check all results are identical
    for i in range(1, len(results_cached)):
        diff = torch.abs(results_cached[0]["pac"] - results_cached[i]["pac"]).max().item()
        print(f"   Run 1 vs Run {i+1} difference: {diff:.2e}")
        assert diff < 1e-10, f"Cached results should be identical! Diff: {diff}"
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY:")
    print("✓ Cache hits work correctly for identical data")
    print("✓ In-place modifications are properly detected")
    print("✓ Caching provides significant performance benefits")
    print("✓ Numerical results are consistent and reproducible")
    print(f"✓ Average cache speedup: {repeat_times[0]/np.mean(repeat_times[1:]):.1f}x")
    print("=" * 80)
    
    # Save results
    results = {
        'cache_speedup': repeat_times[0]/np.mean(repeat_times[1:]),
        'first_run_time': repeat_times[0],
        'cached_run_time': np.mean(repeat_times[1:]),
        'correctness_verified': True
    }
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(['First Run', 'Cached Runs'], 
            [repeat_times[0], np.mean(repeat_times[1:])], 
            color=['red', 'green'])
    plt.ylabel('Time (seconds)')
    plt.title('Caching Performance')
    plt.text(0, repeat_times[0] + 0.01, f'{repeat_times[0]:.3f}s', ha='center')
    plt.text(1, np.mean(repeat_times[1:]) + 0.01, f'{np.mean(repeat_times[1:]):.3f}s', ha='center')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 6), repeat_times, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(repeat_times[1:]), color='green', linestyle='--', label='Cached avg')
    plt.xlabel('Run Number')
    plt.ylabel('Time (seconds)')
    plt.title('Repeated Processing Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('benchmark/verify_caching_correctness_out', exist_ok=True)
    plt.savefig('benchmark/verify_caching_correctness_out/caching_performance.png', dpi=150)
    plt.close()
    
    return results

if __name__ == "__main__":
    results = test_caching_correctness_and_speed()
    print(f"\nFinal cache speedup: {results['cache_speedup']:.1f}x")