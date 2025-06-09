#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:36:36 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/test__PAC.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/test__PAC.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import time

import pytest
import torch
from gpac import PAC


@torch.no_grad()
def test_pac_speed():
    """Test PAC speed with realistic neurophysiology data."""
    # Realistic parameters
    fs = 1000.0
    seq_len = 2000
    batch_size = 16
    n_channels = 64
    n_perm = 50
    n_pha_bands = 10
    n_amp_bands = 15

    # Create synthetic LFP-like data
    torch.manual_seed(42)
    x = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Add realistic phase-amplitude coupling
    time_vec = torch.linspace(0, seq_len / fs, seq_len)
    # Theta phase (8 Hz)
    theta_phase = 2 * torch.pi * 8 * time_vec
    # Gamma amplitude modulated by theta phase
    gamma_freq = 80
    gamma_amp = 1 + 0.5 * torch.cos(theta_phase)
    gamma_signal = gamma_amp * torch.sin(2 * torch.pi * gamma_freq * time_vec)

    # Add coupling to some channels
    for ch_idx in range(0, n_channels, 4):
        for batch_idx in range(batch_size):
            x[batch_idx, ch_idx] += (
                0.3 * torch.sin(theta_phase) + 0.2 * gamma_signal
            )

    # Initialize PAC with aggressive settings
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=4,
        pha_end_hz=12,
        pha_n_bands=n_pha_bands,
        amp_start_hz=60,
        amp_end_hz=100,
        amp_n_bands=n_amp_bands,
        n_perm=n_perm,
        surrogate_chunk_size=20,
        fp16=True,
        device_ids="all",
        compile_mode=True,
    )

    # Move to GPU
    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    # Warmup run
    with torch.no_grad():
        _ = pac(x[:2, :8])

    # Clear cache and sync
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Speed test
    start_time = time.time()
    with torch.no_grad():
        results = pac(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    compute_time = end_time - start_time

    # Results
    pac_values = results["pac"]

    # Assertions
    assert pac_values.shape == (
        batch_size,
        n_channels,
        n_pha_bands,
        n_amp_bands,
    )
    assert pac_values.min() >= 0.0
    assert pac_values.max() <= 1.0
    assert results["pac_z"] is not None


def run_speed_benchmark():
    """Non-test version for benchmarking."""
    # Realistic parameters
    fs = 1000.0
    seq_len = 2000
    batch_size = 16
    n_channels = 64
    n_perm = 50
    n_pha_bands = 10
    n_amp_bands = 15

    # Create synthetic LFP-like data
    torch.manual_seed(42)
    x = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Add realistic phase-amplitude coupling
    time_vec = torch.linspace(0, seq_len / fs, seq_len)
    # Theta phase (8 Hz)
    theta_phase = 2 * torch.pi * 8 * time_vec
    # Gamma amplitude modulated by theta phase
    gamma_freq = 80
    gamma_amp = 1 + 0.5 * torch.cos(theta_phase)
    gamma_signal = gamma_amp * torch.sin(2 * torch.pi * gamma_freq * time_vec)

    # Add coupling to some channels
    for ch_idx in range(0, n_channels, 4):
        for batch_idx in range(batch_size):
            x[batch_idx, ch_idx] += (
                0.3 * torch.sin(theta_phase) + 0.2 * gamma_signal
            )

    # Initialize PAC with aggressive settings
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=4,
        pha_end_hz=12,
        pha_n_bands=n_pha_bands,
        amp_start_hz=60,
        amp_end_hz=100,
        amp_n_bands=n_amp_bands,
        n_perm=n_perm,
        surrogate_chunk_size=20,
        fp16=True,
        device_ids="all",
        compile_mode=True,
    )

    # Move to GPU
    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Warmup run
    print("Warmup...")
    with torch.no_grad():
        _ = pac(x[:2, :8])

    # Clear cache and sync
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Speed test
    print(f"\nTesting PAC speed...")
    print(f"Data shape: {x.shape}")
    print(f"Phase bands: {pac.pha_mids.shape[0]}")
    print(f"Amplitude bands: {pac.amp_mids.shape[0]}")
    print(f"Total freq pairs: {pac.pha_mids.shape[0] * pac.amp_mids.shape[0]}")
    print(f"Permutations: {n_perm}")

    start_time = time.time()
    with torch.no_grad():
        results = pac(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    compute_time = end_time - start_time

    # Results
    pac_values = results["pac"]
    print(f"\nResults:")
    print(f"Computation time: {compute_time:.3f} seconds")
    print(f"PAC shape: {pac_values.shape}")
    print(
        f"PAC range: [{pac_values.min().item():.3f}, {pac_values.max().item():.3f}]"
    )
    print(f"PAC mean: {pac_values.mean().item():.3f}")

    if results["pac_z"] is not None:
        pac_z = results["pac_z"]
        print(
            f"PAC z-score range: [{pac_z.min().item():.3f}, {pac_z.max().item():.3f}]"
        )
        significant_pac = (pac_z > 2.0).sum().item()
        total_pairs = pac_z.numel()
        print(
            f"Significant PAC pairs (z>2): {significant_pac}/{total_pairs} ({100*significant_pac/total_pairs:.1f}%)"
        )

    # Performance metrics
    total_operations = (
        batch_size
        * n_channels
        * pac.pha_mids.shape[0]
        * pac.amp_mids.shape[0]
        * (1 + n_perm)
    )
    ops_per_second = total_operations / compute_time
    print(f"\nPerformance:")
    print(f"Total operations: {total_operations:,}")
    print(f"Operations/second: {ops_per_second:,.0f}")

    # Memory info
    memory_info = pac.get_memory_info()
    print(f"\nMemory:")
    for key, value in memory_info.items():
        print(f"{key}: {value}")

    # Validate coupling detection
    # Should find theta-gamma coupling around (8Hz, 80Hz)
    theta_idx = torch.argmin(torch.abs(pac.pha_mids - 8.0))
    gamma_idx = torch.argmin(torch.abs(pac.amp_mids - 80.0))
    coupling_strength = pac_values[:, :, theta_idx, gamma_idx].mean()
    print(f"\nCoupling validation:")
    print(
        f"Expected coupling at ({pac.pha_mids[theta_idx]:.1f}Hz, {pac.amp_mids[gamma_idx]:.1f}Hz)"
    )
    print(f"Coupling strength: {coupling_strength:.3f}")

    return results, compute_time


def test_memory_scaling():
    """Test memory scaling with different data sizes."""
    fs = 1000.0
    seq_len = 1000
    sizes = [
        (4, 16),  # Small
        (8, 32),  # Medium
        (16, 64),  # Large
        (32, 128),  # Very large
    ]

    for batch_size, n_channels in sizes:
        print(f"\nTesting size: {batch_size} x {n_channels} x {seq_len}")
        x = torch.randn(batch_size, n_channels, seq_len)

        if torch.cuda.is_available():
            x = x.cuda()

        pac = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=8,
            amp_n_bands=10,
            n_perm=50,
            surrogate_chunk_size=20,
            fp16=True,
            compile_mode=False,
        )

        if torch.cuda.is_available():
            pac = pac.cuda()

        try:
            start_time = time.time()
            with torch.no_grad():
                results = pac(x)
            compute_time = time.time() - start_time
            print(f"  ✅ Success: {compute_time:.3f}s")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ OOM: {str(e)}")
            else:
                print(f"  ❌ Error: {str(e)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("🚀 PAC Speed Test")
    print("=" * 50)

    results, compute_time = run_speed_benchmark()

    print("\n" + "=" * 50)
    print("📊 Memory Scaling Test")
    test_memory_scaling()

    print(f"\n✅ Tests completed!")
    print(f"Main test time: {compute_time:.3f}s")

    pytest.main([__file__, "-v"])

# EOF
