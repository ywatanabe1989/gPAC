#!/usr/bin/env python3
"""
Profile gPAC performance to identify bottlenecks.
"""

import torch
import time
import numpy as np
from contextlib import contextmanager
import cProfile
import pstats
import io

# Add gPAC to path
import sys
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')

from gpac import calculate_pac, PAC
from gpac._SyntheticDataGenerator import SyntheticDataGenerator

@contextmanager
def timer(name):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {(end - start)*1000:.2f} ms")

def profile_components():
    """Profile individual components of gPAC."""
    
    # Test parameters
    fs = 512.0
    duration = 2.0
    seq_len = int(fs * duration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Signal length: {seq_len} samples ({duration}s @ {fs}Hz)")
    print("-" * 50)
    
    # Generate test signal
    torch.manual_seed(42)
    signal = torch.randn(1, 1, 1, seq_len).to(device)
    
    # Profile calculate_pac
    print("\n1. Profiling calculate_pac (full pipeline):")
    with timer("Total PAC calculation"):
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=10,
            amp_n_bands=10,
            device=device
        )
    
    # Profile PAC module initialization
    print("\n2. Profiling PAC module initialization:")
    with timer("PAC module init"):
        pac_module = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=10,
            amp_n_bands=10
        ).to(device)
    
    # Profile forward pass only
    print("\n3. Profiling forward pass only:")
    with timer("Forward pass"):
        result = pac_module(signal)
    
    # Profile individual components
    print("\n4. Profiling individual components:")
    
    # BandPass filter
    from gpac._BandPassFilter import BandPassFilter
    with timer("BandPass filter init"):
        filter_module = BandPassFilter(
            pha_bands=torch.tensor([[4.0, 8.0], [8.0, 12.0]]),
            amp_bands=torch.tensor([[30.0, 50.0], [50.0, 80.0]]),
            fs=fs,
            seq_len=seq_len
        ).to(device)
    
    with timer("BandPass forward"):
        filtered = filter_module(signal[:, :, 0, :])  # Remove segment dimension
    
    # Hilbert transform
    from gpac._Hilbert import Hilbert
    hilbert = Hilbert(seq_len=seq_len).to(device)
    with timer("Hilbert forward"):
        analytic = hilbert(filtered)
        # Extract phase and amplitude from tensor
        phase = analytic[..., 0]
        amplitude = analytic[..., 1]
    
    # Modulation Index
    from gpac._ModulationIndex import ModulationIndex
    mi = ModulationIndex().to(device)
    with timer("MI forward"):
        # MI expects 5D input: (batch, channel, segment, band, time)
        # We need to add segment dimension and select appropriate bands
        phase_5d = phase.unsqueeze(2)  # Add segment dimension
        amplitude_5d = amplitude.unsqueeze(2)  # Add segment dimension
        mi_values = mi(phase_5d[:, :, :, 0:1, :], amplitude_5d[:, :, :, 1:2, :])
    
    # Test batch processing efficiency
    print("\n5. Batch processing efficiency:")
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        batch_signal = torch.randn(batch_size, 1, 1, seq_len).to(device)
        
        # Warm up
        _ = pac_module(batch_signal)
        
        # Time it
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        _ = pac_module(batch_signal)
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.time()
        
        time_per_sample = (end - start) / batch_size * 1000
        efficiency = (batch_sizes[0] * (end - start) / batch_size) / (end - start)
        print(f"  Batch size {batch_size}: {time_per_sample:.2f} ms/sample (efficiency: {efficiency:.2f}x)")

def detailed_profile():
    """Run detailed profiling with cProfile."""
    print("\n" + "="*50)
    print("DETAILED PROFILING WITH cProfile")
    print("="*50)
    
    # Setup
    fs = 512.0
    seq_len = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal = torch.randn(1, 1, 1, seq_len).to(device)
    
    # Profile the main function
    pr = cProfile.Profile()
    pr.enable()
    
    # Run PAC calculation multiple times
    for _ in range(10):
        pac_values, _, _ = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=10,
            amp_n_bands=10,
            device=device
        )
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())

def memory_profile():
    """Profile memory usage."""
    print("\n" + "="*50)
    print("MEMORY PROFILING")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory profiling")
        return
    
    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Setup
    fs = 512.0
    seq_len = 2048
    device = 'cuda'
    
    print(f"Initial memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # Create signal
    signal = torch.randn(4, 1, 1, seq_len).to(device)
    print(f"After signal creation: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # Run PAC
    pac_values, _, _ = calculate_pac(
        signal,
        fs=fs,
        pha_n_bands=20,
        amp_n_bands=20,
        device=device
    )
    
    print(f"After PAC calculation: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

if __name__ == "__main__":
    print("gPAC Performance Profiling")
    print("="*50)
    
    # Basic component profiling
    profile_components()
    
    # Detailed profiling
    detailed_profile()
    
    # Memory profiling
    memory_profile()
    
    print("\nProfiling complete!")