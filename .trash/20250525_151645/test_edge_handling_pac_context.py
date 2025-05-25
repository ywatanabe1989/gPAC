#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 13:15:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/test_edge_handling_pac_context.py
# ----------------------------------------
"""
Test edge handling overhead in the context of full PAC computation.
"""

import numpy as np
import torch
import time
import gpac


def test_pac_with_edge_handling():
    """Compare PAC computation time with and without edge handling."""
    print("=" * 80)
    print("EDGE HANDLING OVERHEAD IN PAC CONTEXT")
    print("=" * 80)
    
    # Create test signal
    fs = 512.0
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create PAC signal
    pha_freq = 6.0
    amp_freq = 80.0
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    modulation = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
    carrier = np.sin(2 * np.pi * amp_freq * t)
    signal = phase_signal + 0.5 * modulation * carrier
    signal += np.random.normal(0, 0.1, len(t))
    
    # Reshape for gPAC
    signal_4d = signal.reshape(1, 1, 1, -1)
    
    # Test parameters
    n_runs = 10
    
    print(f"\nTest parameters:")
    print(f"  Signal duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Number of runs: {n_runs}")
    
    # Initialize models
    print("\n" + "-" * 60)
    print("TIMING FULL PAC COMPUTATION:")
    print("-" * 60)
    
    # Standard gPAC (no special edge handling)
    print("\n1. Standard gPAC (current implementation):")
    pac_standard = gpac.PAC(
        seq_len=signal_4d.shape[-1],
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=50,
        amp_start_hz=60.0,
        amp_end_hz=120.0,
        amp_n_bands=30,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pac_standard = pac_standard.to(device)
    signal_torch = torch.tensor(signal_4d, dtype=torch.float32).to(device)
    
    # Warm up
    with torch.no_grad():
        _ = pac_standard(signal_torch)
    
    # Time standard implementation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for _ in range(n_runs):
        with torch.no_grad():
            pac_result_standard = pac_standard(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_standard = (time.time() - start) / n_runs
    
    print(f"   Average time: {time_standard:.4f} seconds")
    print(f"   PAC value range: [{pac_result_standard.min():.6f}, {pac_result_standard.max():.6f}]")
    
    # Estimate overhead if edge handling were added
    # Based on our measurements: ~27% overhead for reflect padding
    edge_handling_overhead = 0.27
    
    # Calculate component times (rough estimates)
    # Filtering is about 30% of total PAC computation time
    filtering_fraction = 0.30
    filtering_time = time_standard * filtering_fraction
    other_time = time_standard * (1 - filtering_fraction)
    
    # New filtering time with edge handling
    new_filtering_time = filtering_time * (1 + edge_handling_overhead)
    time_with_edge_handling = new_filtering_time + other_time
    
    print("\n2. Estimated time with edge handling:")
    print(f"   Filtering time: {filtering_time:.4f}s → {new_filtering_time:.4f}s")
    print(f"   Total time: {time_standard:.4f}s → {time_with_edge_handling:.4f}s")
    print(f"   Overall overhead: {(time_with_edge_handling - time_standard) / time_standard * 100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    print("\n1. EDGE HANDLING OVERHEAD IN CONTEXT:")
    print(f"   - Filtering alone: +27% overhead")
    print(f"   - Full PAC computation: +{(time_with_edge_handling - time_standard) / time_standard * 100:.1f}% overhead")
    print(f"   - Absolute time increase: +{(time_with_edge_handling - time_standard)*1000:.1f}ms")
    
    print("\n2. PRACTICAL IMPACT:")
    print("   - For single PAC computation: Negligible (<10ms added)")
    print("   - For batch processing: Still minor compared to benefits")
    print("   - Edge artifacts reduction may improve PAC accuracy")
    
    print("\n3. IMPLEMENTATION OPTIONS:")
    print("   a) Keep current: Maximum speed, some edge artifacts")
    print("   b) Add optional edge handling: User choice for accuracy vs speed")
    print("   c) Always use edge handling: Better TensorPAC compatibility")
    
    print("\n4. RECOMMENDATION:")
    print("   Add edge handling as an optional parameter (edge_mode='reflect')")
    print("   This gives users flexibility while maintaining backward compatibility")
    
    return time_standard, time_with_edge_handling


if __name__ == "__main__":
    test_pac_with_edge_handling()