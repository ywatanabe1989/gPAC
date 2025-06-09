#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 18:45:00 (ywatanabe)"
# File: ./tests/gpac/test__PAC_vectorization.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/test__PAC_vectorization.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests vectorized PAC implementation for correctness
  - Verifies loop elimination performance improvement
  - Ensures numerical accuracy with loop-based reference
  - Tests edge cases and various input sizes

Dependencies:
  - scripts:
    - /src/gpac/_PAC.py
  - packages:
    - pytest
    - torch
    - numpy

IO:
  - input-files:
    - None
  - output-files:
    - None (test results to stdout)
"""

"""Imports"""
import sys
import time

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(__FILE__) + "/../../src")
from gpac._PAC import PAC
from gpac._ModulationIndex import ModulationIndexOptimized

"""Functions & Classes"""
class LoopBasedPAC:
    """Reference implementation with explicit loops for comparison."""
    
    def __init__(self):
        self.modulation_index = ModulationIndexOptimized(n_bins=18)
    
    def compute_mi_with_loops(self, phase, amplitude):
        """Original loop-based implementation for reference."""
        batch, channels, n_pha, _, seq_len = phase.shape
        _, _, _, n_amp, _ = amplitude.shape
        
        pac_values = torch.zeros(batch, channels, n_pha, n_amp, device=phase.device)
        
        # Original nested loops
        for p in range(n_pha):
            for a in range(n_amp):
                p_band = phase[:, :, p, 0, :]
                a_band = amplitude[:, :, 0, a, :]
                
                mi_result = self.modulation_index(
                    p_band.unsqueeze(2).unsqueeze(2),
                    a_band.unsqueeze(2).unsqueeze(2)
                )
                pac_values[:, :, p, a] = mi_result['mi'].squeeze()
                
        return pac_values


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def pac_model(device):
    """Create PAC model for testing."""
    return PAC(
        seq_len=512,
        fs=256,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=10,
        amp_start_hz=60,
        amp_end_hz=140,
        amp_n_bands=10,
        compile_mode=False  # Disable compile for testing
    ).to(device)


@pytest.fixture
def reference_pac():
    """Create reference loop-based PAC."""
    return LoopBasedPAC()


def test_vectorization_correctness(pac_model, reference_pac, device):
    """Test that vectorized implementation produces same results as loops."""
    torch.manual_seed(42)
    
    # Create test data
    batch, channels, seq_len = 2, 3, 512
    x = torch.randn(batch, channels, seq_len, device=device)
    
    # Get filtered data from PAC model
    with torch.no_grad():
        x_filtered = pac_model.bandpass(x)
        n_pha = pac_model.pha_mids.shape[0]
        n_amp = pac_model.amp_mids.shape[0]
        
        # Split and process
        pha_filtered = x_filtered[:, :, :n_pha, :]
        amp_filtered = x_filtered[:, :, n_pha:, :]
        
        pha_hilbert = pac_model.hilbert(pha_filtered)
        amp_hilbert = pac_model.hilbert(amp_filtered)
        
        phase = pha_hilbert[..., 0].unsqueeze(3)
        amplitude = amp_hilbert[..., 1].unsqueeze(2)
        
        # Compare vectorized vs loop-based
        pac_vectorized = pac_model._compute_mi_vectorized(phase, amplitude)
        pac_loops = reference_pac.compute_mi_with_loops(phase, amplitude)
        
    # Check numerical equivalence
    assert torch.allclose(pac_vectorized, pac_loops, rtol=1e-5, atol=1e-6), \
        f"Vectorized and loop-based results differ! Max diff: {(pac_vectorized - pac_loops).abs().max()}"


def test_vectorization_performance(pac_model, reference_pac, device):
    """Test that vectorized implementation is faster than loops."""
    torch.manual_seed(42)
    
    # Create larger test data for meaningful timing
    batch, channels, seq_len = 4, 8, 1024
    x = torch.randn(batch, channels, seq_len, device=device)
    
    # Prepare data
    with torch.no_grad():
        x_filtered = pac_model.bandpass(x)
        n_pha = pac_model.pha_mids.shape[0]
        n_amp = pac_model.amp_mids.shape[0]
        
        pha_filtered = x_filtered[:, :, :n_pha, :]
        amp_filtered = x_filtered[:, :, n_pha:, :]
        
        pha_hilbert = pac_model.hilbert(pha_filtered)
        amp_hilbert = pac_model.hilbert(amp_filtered)
        
        phase = pha_hilbert[..., 0].unsqueeze(3)
        amplitude = amp_hilbert[..., 1].unsqueeze(2)
        
        # Warmup
        _ = pac_model._compute_mi_vectorized(phase, amplitude)
        _ = reference_pac.compute_mi_with_loops(phase, amplitude)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Time vectorized implementation
        n_trials = 10
        start = time.time()
        for _ in range(n_trials):
            _ = pac_model._compute_mi_vectorized(phase, amplitude)
            if device == "cuda":
                torch.cuda.synchronize()
        time_vectorized = (time.time() - start) / n_trials
        
        # Time loop-based implementation
        start = time.time()
        for _ in range(n_trials):
            _ = reference_pac.compute_mi_with_loops(phase, amplitude)
            if device == "cuda":
                torch.cuda.synchronize()
        time_loops = (time.time() - start) / n_trials
        
    speedup = time_loops / time_vectorized
    print(f"\nPerformance comparison:")
    print(f"  Loop-based: {time_loops:.4f}s")
    print(f"  Vectorized: {time_vectorized:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    assert speedup > 1.2, f"Vectorized should be faster! Got {speedup:.2f}x speedup"


def test_different_band_sizes(pac_model, device):
    """Test vectorization with various phase/amplitude band counts."""
    test_configs = [
        (5, 5),    # Small
        (10, 10),  # Medium
        (20, 30),  # Asymmetric
        (30, 20),  # Asymmetric reversed
    ]
    
    for n_pha, n_amp in test_configs:
        # Create custom PAC model with valid frequency ranges
        # Use fs=512 to support amp frequencies up to 160 Hz
        model = PAC(
            seq_len=512,
            fs=512,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=n_pha,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=n_amp,
            compile_mode=False
        ).to(device)
        
        # Test forward pass
        x = torch.randn(2, 1, 512, device=device)
        with torch.no_grad():
            result = model(x)
            
        assert result['pac'].shape == (2, 1, n_pha, n_amp), \
            f"Wrong output shape for {n_pha}x{n_amp} bands"


def test_memory_efficiency(pac_model, device):
    """Test memory usage of vectorized implementation."""
    if device == "cpu":
        pytest.skip("Memory profiling only meaningful on GPU")
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Large batch test
    batch, channels, seq_len = 16, 4, 2048
    x = torch.randn(batch, channels, seq_len, device=device)
    
    # Measure peak memory
    start_mem = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        _ = pac_model(x)
        
    peak_mem = torch.cuda.max_memory_allocated() - start_mem
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    print(f"\nMemory usage for {batch}x{channels}x{seq_len}:")
    print(f"  Peak memory: {peak_mem_mb:.2f} MB")
    
    # Ensure reasonable memory usage
    # With 30x30 bands, memory usage includes:
    # - Filter kernels: 30*30 = 900 filters
    # - Intermediate filtered signals: batch * channels * 900 * seq_len
    # - ModulationIndex computations
    # For full PAC, expect ~1000x input size due to many intermediate tensors
    expected_max_mb = batch * channels * seq_len * 4 * 1000 / (1024 * 1024)
    assert peak_mem_mb < expected_max_mb, \
        f"Memory usage too high: {peak_mem_mb:.2f} MB > {expected_max_mb:.2f} MB"


def test_gradient_flow(pac_model, device):
    """Test that gradients flow correctly through vectorized implementation."""
    # Enable gradients
    x = torch.randn(2, 1, 512, device=device, requires_grad=True)
    
    # Forward pass
    result = pac_model(x)
    pac_matrix = result['pac']
    
    # Compute loss and backward
    loss = pac_matrix.mean()
    loss.backward()
    
    # Check gradient exists and is non-zero
    assert x.grad is not None, "No gradient computed"
    assert x.grad.abs().max() > 0, "Gradient is zero"
    
    # Check gradient shape
    assert x.grad.shape == x.shape, "Gradient shape mismatch"


def test_edge_cases(pac_model, device):
    """Test edge cases for vectorized implementation."""
    # Single batch
    x = torch.randn(1, 1, 512, device=device)
    result = pac_model(x)
    assert result['pac'].shape[0] == 1
    
    # Single channel
    x = torch.randn(2, 1, 512, device=device)
    result = pac_model(x)
    assert result['pac'].shape[1] == 1
    
    # Very short sequence - use valid frequency ranges
    model = PAC(
        seq_len=64, 
        fs=512,  # Higher fs to support amp frequencies up to 160 Hz
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=5,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=5,
        compile_mode=False
    ).to(device)
    x = torch.randn(2, 1, 64, device=device)
    result = model(x)
    assert result['pac'].shape[-1] > 0


def test_numerical_stability(pac_model, device):
    """Test numerical stability of vectorized implementation."""
    # Test with extreme values
    test_cases = [
        torch.randn(2, 1, 512, device=device) * 1e-6,  # Very small
        torch.randn(2, 1, 512, device=device) * 1e3,   # Large
        torch.zeros(2, 1, 512, device=device),         # All zeros
    ]
    
    for x in test_cases:
        with torch.no_grad():
            result = pac_model(x)
            pac = result['pac']
            
        # Check for NaN or Inf
        assert not torch.isnan(pac).any(), "NaN values in output"
        assert not torch.isinf(pac).any(), "Inf values in output"
        
        # PAC values should be bounded
        assert (pac >= 0).all(), "Negative PAC values"
        assert (pac <= 1).all(), "PAC values > 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF