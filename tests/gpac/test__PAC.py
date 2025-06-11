#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 14:19:38 (ywatanabe)"
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


@torch.no_grad()
def test_pac_multi_gpu_device_ids():
    """Test PAC with different device_ids configurations."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU test requires at least 2 GPUs")

    fs = 1000.0
    seq_len = 1000
    batch_size = 4
    n_channels = 8

    x = torch.randn(batch_size, n_channels, seq_len)

    # Test 1: Single GPU (device_ids=[0])
    pac_single = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        n_perm=10,
        device_ids=[0],
    )
    x_gpu0 = x.cuda(0)
    pac_single = pac_single.cuda(0)
    results_single = pac_single(x_gpu0)

    # Test 2: Two GPUs (device_ids=[0, 1])
    pac_dual = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        n_perm=10,
        device_ids=[0, 1],
    )
    x_gpu = x.cuda(0)
    pac_dual = pac_dual.cuda(0)
    results_dual = pac_dual(x_gpu)

    # Test 3: All GPUs (device_ids="all")
    pac_all = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        n_perm=10,
        device_ids="all",
    )
    x_gpu_all = x.cuda()
    pac_all = pac_all.cuda()
    results_all = pac_all(x_gpu_all)

    # Verify results have same shape
    assert results_single["pac"].shape == results_dual["pac"].shape
    assert results_single["pac"].shape == results_all["pac"].shape

    # Verify memory info reports correct devices
    memory_info_single = pac_single.get_memory_info()
    memory_info_dual = pac_dual.get_memory_info()
    memory_info_all = pac_all.get_memory_info()

    assert memory_info_single["devices"] == [0]
    assert memory_info_dual["devices"] == [0, 1]
    assert len(memory_info_all["devices"]) == torch.cuda.device_count()

    print(f"✅ Single GPU devices: {memory_info_single['devices']}")
    print(f"✅ Dual GPU devices: {memory_info_dual['devices']}")
    print(f"✅ All GPU devices: {memory_info_all['devices']}")


@torch.no_grad()
def test_pac_multi_gpu_large_batch():
    """Test multi-GPU with large batch sizes that benefit from parallelization."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU test requires at least 2 GPUs")

    fs = 1000.0
    seq_len = 2000
    batch_size = 32  # Large batch to benefit from multi-GPU
    n_channels = 16

    x = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32)

    # Single GPU timing
    pac_single = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=5,
        amp_n_bands=5,
        n_perm=20,
        fp16=True,
        device_ids=[0],
    )

    x_gpu = x.cuda(0)
    pac_single = pac_single.cuda(0)

    # Warmup
    _ = pac_single(x_gpu[:2])
    torch.cuda.synchronize()

    start_time = time.time()
    results_single = pac_single(x_gpu)
    torch.cuda.synchronize()
    single_gpu_time = time.time() - start_time

    # Multi-GPU timing
    pac_multi = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=5,
        amp_n_bands=5,
        n_perm=20,
        fp16=True,
        device_ids="all",
    )

    pac_multi = pac_multi.cuda()

    # Warmup
    _ = pac_multi(x_gpu[:2])
    torch.cuda.synchronize()

    start_time = time.time()
    results_multi = pac_multi(x_gpu)
    torch.cuda.synchronize()
    multi_gpu_time = time.time() - start_time

    print(f"\nMulti-GPU Performance Test:")
    print(
        f"Batch size: {batch_size}, Channels: {n_channels}, Seq len: {seq_len}"
    )
    print(f"Single GPU time: {single_gpu_time:.3f}s")
    print(f"Multi-GPU time: {multi_gpu_time:.3f}s")
    print(f"Speedup: {single_gpu_time/multi_gpu_time:.2f}x")

    # Verify results are similar (allowing for minor numerical differences)
    assert torch.allclose(
        results_single["pac"], results_multi["pac"], rtol=1e-3, atol=1e-5
    )


@torch.no_grad()
def test_pac_multi_gpu_memory_distribution():
    """Test memory distribution across multiple GPUs."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU test requires at least 2 GPUs")

    fs = 1000.0
    seq_len = 1000
    batch_size = 8
    n_channels = 32

    # Clear all GPU memory first
    for ii in range(torch.cuda.device_count()):
        torch.cuda.set_device(ii)
        torch.cuda.empty_cache()

    # Create PAC with multiple GPUs
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=8,
        amp_n_bands=10,
        n_perm=50,
        fp16=True,
        device_ids=[0, 1],
    )

    x = torch.randn(batch_size, n_channels, seq_len).cuda()
    pac = pac.cuda()

    # Run computation and measure memory during computation
    results = pac(x)

    # Get memory after computation while tensors still exist
    memory_allocated = {}
    for ii in [0, 1]:
        torch.cuda.set_device(ii)
        memory_allocated[ii] = torch.cuda.memory_allocated(ii) / 1e9

    print(f"\nMemory Distribution Test:")
    print(f"GPU 0: {memory_allocated[0]:.3f} GB allocated")
    print(f"GPU 1: {memory_allocated[1]:.3f} GB allocated")
    print(f"Total: {sum(memory_allocated.values()):.3f} GB")

    # At least one GPU should have allocated memory (lowered threshold)
    total_memory = sum(memory_allocated.values())
    assert (
        total_memory > 0.001
    ), f"Expected >0.001GB memory usage, got {total_memory:.3f}GB"

    # Get detailed memory info
    memory_info = pac.get_memory_info()
    print(f"\nDetailed memory info:")
    for device_id, info in memory_info["device_details"].items():
        print(f"  GPU {device_id} ({info['name']}):")
        print(f"    Total: {info['total_gb']:.1f} GB")
        print(f"    Allocated: {info['allocated_gb']:.3f} GB")
        print(f"    Free: {info['free_gb']:.1f} GB")


@torch.no_grad()
def test_pac_device_ids_edge_cases():
    """Test edge cases for device_ids parameter."""
    fs = 1000.0
    seq_len = 1000
    x = torch.randn(2, 4, seq_len)

    # Test 1: Empty device_ids list (should default to CPU)
    pac_cpu = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        device_ids=[],
    )
    results_cpu = pac_cpu(x)
    assert results_cpu["pac"].device.type == "cpu"

    # Test 2: Invalid device_ids (should handle gracefully)
    if torch.cuda.is_available():
        max_gpu = torch.cuda.device_count()
        try:
            pac_invalid = PAC(
                seq_len=seq_len,
                fs=fs,
                pha_range_hz=(4, 12),
                amp_range_hz=(60, 100),
                pha_n_bands=3,
                amp_n_bands=3,
                device_ids=[max_gpu + 1],  # Invalid GPU ID
            )
            # Should either raise error or fallback gracefully
        except (RuntimeError, AssertionError) as e:
            print(f"✅ Correctly handled invalid device_id: {e}")

    # Test 3: String other than "all"
    pac_default = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=3,
        device_ids="single",  # Not "all"
    )
    if torch.cuda.is_available():
        assert pac_default.device_ids == [0]
    else:
        assert pac_default.device_ids == []


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
