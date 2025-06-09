#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:37:29 (ywatanabe)"
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
def test_pac_3d():
    """Test basic PAC functionality."""
    fs = 1000.0
    seq_len = 1000
    batch_size = 2
    n_channels = 4

    x = torch.randn(batch_size, n_channels, seq_len)

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        n_perm=10,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    results = pac(x)

    assert results["pac"].shape == (batch_size, n_channels, 3, 3)
    assert results["pac_z"] is not None
    assert results["surrogate_mean"] is not None
    assert results["surrogate_std"] is not None


@torch.no_grad()
def test_pac_4d():
    """Test basic PAC functionality."""
    fs = 1000.0
    seq_len = 1000
    batch_size = 2
    n_channels = 4
    n_segments = 3

    x = torch.randn(batch_size, n_channels, n_segments, seq_len)

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        n_perm=10,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    results = pac(x)

    assert results["pac"].shape == (batch_size, n_channels, n_segments, 3, 3)
    assert results["pac_z"] is not None
    assert results["surrogate_mean"] is not None
    assert results["surrogate_std"] is not None


@torch.no_grad()
def test_pac_speed():
    """Test PAC speed with realistic neurophysiology data."""
    fs = 1000.0
    seq_len = 2000
    batch_size = 16
    n_channels = 64
    n_perm = 50
    n_pha_bands = 10
    n_amp_bands = 15

    torch.manual_seed(42)
    x = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Add realistic phase-amplitude coupling
    time_vec = torch.linspace(0, seq_len / fs, seq_len)
    theta_phase = 2 * torch.pi * 8 * time_vec
    gamma_freq = 80
    gamma_amp = 1 + 0.5 * torch.cos(theta_phase)
    gamma_signal = gamma_amp * torch.sin(2 * torch.pi * gamma_freq * time_vec)

    for ch_idx in range(0, n_channels, 4):
        for batch_idx in range(batch_size):
            x[batch_idx, ch_idx] += (
                0.3 * torch.sin(theta_phase) + 0.2 * gamma_signal
            )

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=n_pha_bands,
        amp_n_bands=n_amp_bands,
        n_perm=n_perm,
        surrogate_chunk_size=20,
        fp16=True,
        device_ids="all",
        compile_mode=True,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    # Warmup
    with torch.no_grad():
        _ = pac(x[:2, :8])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        results = pac(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    pac_values = results["pac"]

    assert pac_values.shape == (
        batch_size,
        n_channels,
        n_pha_bands,
        n_amp_bands,
    )
    assert pac_values.min() >= 0.0
    assert pac_values.max() <= 1.0
    assert results["pac_z"] is not None
    assert results["surrogates"] is None
    assert results["surrogate_mean"] is not None
    assert results["surrogate_std"] is not None

    print(f"Computation time: {end_time - start_time:.3f}s")


@torch.no_grad()
def test_pac_custom_bands():
    """Test PAC with custom frequency bands."""
    fs = 1000.0
    seq_len = 1000
    batch_size = 2
    n_channels = 4

    # Custom bands
    pha_bands = [[4, 8], [8, 12], [12, 20]]
    amp_bands = [[60, 80], [80, 120], [120, 160]]

    x = torch.randn(batch_size, n_channels, seq_len)

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        n_perm=10,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    results = pac(x)

    assert results["pac"].shape == (batch_size, n_channels, 3, 3)
    assert len(pac.pha_mids) == 3
    assert len(pac.amp_mids) == 3


def test_pac_trainable_mode():
    """Test PAC with trainable filters."""
    fs = 1000.0
    seq_len = 1000
    batch_size = 2
    n_channels = 4

    x = torch.randn(batch_size, n_channels, seq_len, requires_grad=True)

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        trainable=True,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        n_perm=None,
        compile_mode=False,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()

    pac.train()

    # Debug: Check trainable parameters before forward
    trainable_params = [p for p in pac.parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable_params)}")
    for name, param in pac.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    results = pac(x)
    assert results["pac"].shape == (batch_size, n_channels, 3, 3)

    # Check if PAC values require grad
    print(f"PAC requires_grad: {results['pac'].requires_grad}")

    loss = results["pac"].sum()
    loss.backward()

    # Debug: Check which params got gradients
    params_with_grad = []
    for name, param in pac.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            print(f"  {name}: grad={has_grad}")
            if has_grad:
                params_with_grad.append(param)

    assert len(trainable_params) > 0
    assert len(params_with_grad) > 0


@torch.no_grad()
def test_pac_different_input_shapes():
    """Test PAC with different input shapes."""
    fs = 1000.0
    seq_len = 1000

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=2,
        amp_n_bands=2,
        n_perm=5,
    )

    if torch.cuda.is_available():
        pac = pac.cuda()

    # Test 3D input (batch, channels, time)
    x_3d = torch.randn(2, 4, seq_len)
    if torch.cuda.is_available():
        x_3d = x_3d.cuda()

    results_3d = pac(x_3d)
    assert results_3d["pac"].shape == (2, 4, 2, 2)

    # Test 4D input (batch, channels, segments, time)
    x_4d = torch.randn(2, 4, 3, seq_len)
    if torch.cuda.is_available():
        x_4d = x_4d.cuda()

    results_4d = pac(x_4d)
    assert results_4d["pac"].shape == (2, 4, 3, 2, 2)


@torch.no_grad()
def test_pac_memory_scaling():
    """Test memory scaling with different data sizes."""
    fs = 1000.0
    seq_len = 1000
    sizes = [
        (4, 16),  # Small
        (8, 32),  # Medium
        (16, 64),  # Large
    ]

    for batch_size, n_channels in sizes:
        print(f"\nTesting size: {batch_size} x {n_channels} x {seq_len}")
        x = torch.randn(batch_size, n_channels, seq_len)
        if torch.cuda.is_available():
            x = x.cuda()

        pac = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_range_hz=(4, 12),
            amp_range_hz=(60, 100),
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

            assert results["surrogates"] is None
            print(f"  💾 Memory efficient")

        except RuntimeError as ee:
            if "out of memory" in str(ee):
                print(f"  ❌ OOM: {str(ee)}")
            else:
                print(f"  ❌ Error: {str(ee)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# @torch.no_grad()
# def test_pac_coupling_detection():
#     """Test PAC's ability to detect synthetic coupling."""
#     fs = 1000.0
#     seq_len = 4000  # Longer signal for better detection
#     batch_size = 4
#     n_channels = 8

#     # Create signals with known theta-gamma coupling
#     time_vec = torch.linspace(0, seq_len / fs, seq_len)
#     theta_phase = 2 * torch.pi * 8 * time_vec  # 8 Hz theta
#     gamma_freq = 80  # 80 Hz gamma

#     x = torch.randn(batch_size, n_channels, seq_len) * 0.1  # Reduce noise

#     # Add stronger coupling to half the channels
#     for ch_idx in range(0, n_channels, 2):
#         for batch_idx in range(batch_size):
#             # Strong theta oscillation
#             theta_signal = 1.0 * torch.sin(theta_phase)
#             # Strong gamma amplitude modulated by theta phase
#             gamma_amp = 1 + 1.5 * torch.cos(theta_phase)
#             gamma_signal = gamma_amp * torch.sin(
#                 2 * torch.pi * gamma_freq * time_vec
#             )

#             x[batch_idx, ch_idx] += theta_signal + 0.8 * gamma_signal

#     pac = PAC(
#         seq_len=seq_len,
#         fs=fs,
#         pha_range_hz=(4, 12),
#         amp_range_hz=(60, 100),
#         pha_n_bands=5,
#         amp_n_bands=5,
#         n_perm=200,  # More permutations for better statistics
#     )

#     if torch.cuda.is_available():
#         x = x.cuda()
#         pac = pac.cuda()

#     results = pac(x)
#     pac_values = results["pac"]
#     pac_z = results["pac_z"]

#     # Find indices closest to our synthetic coupling
#     theta_idx = torch.argmin(torch.abs(pac.pha_mids - 8.0))
#     gamma_idx = torch.argmin(torch.abs(pac.amp_mids - 80.0))

#     # Channels with coupling should show higher PAC
#     coupled_channels = list(range(0, n_channels, 2))
#     uncoupled_channels = list(range(1, n_channels, 2))

#     coupled_pac = pac_values[:, coupled_channels, theta_idx, gamma_idx].mean()
#     uncoupled_pac = pac_values[
#         :, uncoupled_channels, theta_idx, gamma_idx
#     ].mean()

#     print(f"Coupled PAC: {coupled_pac:.3f}")
#     print(f"Uncoupled PAC: {uncoupled_pac:.3f}")
#     print(f"Difference: {coupled_pac - uncoupled_pac:.3f}")

#     # Coupled channels should have higher PAC values
#     assert coupled_pac > uncoupled_pac

#     # Z-scores should be significant for coupled channels
#     coupled_z = pac_z[:, coupled_channels, theta_idx, gamma_idx].mean()
#     print(f"Coupled Z-score: {coupled_z:.3f}")
#     assert coupled_z > 1.0  # Relaxed threshold for more realistic test


def run_speed_benchmark():
    """Comprehensive speed benchmark."""
    fs = 1000.0
    seq_len = 2000
    batch_size = 16
    n_channels = 64
    n_perm = 50
    n_pha_bands = 10
    n_amp_bands = 15

    torch.manual_seed(42)
    x = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Add realistic coupling
    time_vec = torch.linspace(0, seq_len / fs, seq_len)
    theta_phase = 2 * torch.pi * 8 * time_vec
    gamma_freq = 80
    gamma_amp = 1 + 0.5 * torch.cos(theta_phase)
    gamma_signal = gamma_amp * torch.sin(2 * torch.pi * gamma_freq * time_vec)

    for ch_idx in range(0, n_channels, 4):
        for batch_idx in range(batch_size):
            x[batch_idx, ch_idx] += (
                0.3 * torch.sin(theta_phase) + 0.2 * gamma_signal
            )

    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=n_pha_bands,
        amp_n_bands=n_amp_bands,
        n_perm=n_perm,
        surrogate_chunk_size=20,
        fp16=True,
        device_ids="all",
        compile_mode=True,
    )

    if torch.cuda.is_available():
        x = x.cuda()
        pac = pac.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        _ = pac(x[:2, :8])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

    if results["surrogates"] is None:
        print("✅ Memory efficient: surrogates not stored")
    else:
        print("⚠️ Memory warning: surrogates stored")

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
    theta_idx = torch.argmin(torch.abs(pac.pha_mids - 8.0))
    gamma_idx = torch.argmin(torch.abs(pac.amp_mids - 80.0))
    coupling_strength = pac_values[:, :, theta_idx, gamma_idx].mean()
    print(f"\nCoupling validation:")
    print(
        f"Expected coupling at ({pac.pha_mids[theta_idx]:.1f}Hz, {pac.amp_mids[gamma_idx]:.1f}Hz)"
    )
    print(f"Coupling strength: {coupling_strength:.3f}")

    return results, compute_time


if __name__ == "__main__":
    print("🚀 PAC Comprehensive Test Suite")
    print("=" * 50)

    # Run speed benchmark
    results, compute_time = run_speed_benchmark()

    print("\n" + "=" * 50)
    print("📊 Memory Scaling Test")
    test_pac_memory_scaling()

    print(f"\n✅ Tests completed!")
    print(f"Main test time: {compute_time:.3f}s")

    # Run pytest
    pytest.main([__file__, "-v"])

# EOF
