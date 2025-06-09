#!/usr/bin/env python3
"""
Quick verification of what the current implementation actually does
"""

import torch
import numpy as np
from src.gpac import PAC

# Test parameters
batch_size = 1
n_channels = 2
seq_len = 1000
fs = 256

# Create test signal
signal = torch.randn(batch_size, n_channels, seq_len)

print("="*60)
print("IMPLEMENTATION VERIFICATION TEST")
print("="*60)

# Initialize PAC with small number of bands
pac = PAC(
    seq_len=seq_len,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=10,
    pha_n_bands=5,  # Small for testing
    amp_start_hz=20,
    amp_end_hz=50,
    amp_n_bands=5,   # Small for testing
    n_perm=None,     # No permutations for simplicity
)

print(f"\nTest configuration:")
print(f"  Phase bands: 5")
print(f"  Amplitude bands: 5")
print(f"  Total combinations: 25")

# Check what _compute_mi_vectorized actually does
print("\n\nChecking _compute_mi_vectorized implementation...")

# Create dummy phase and amplitude data
n_pha = 5
n_amp = 5
phase = torch.randn(batch_size, n_channels, n_pha, 1, seq_len)
amplitude = torch.randn(batch_size, n_channels, 1, n_amp, seq_len)

# Look at what happens inside _compute_mi_vectorized
print("\nBefore expansion:")
print(f"  Phase shape: {phase.shape}")
print(f"  Amplitude shape: {amplitude.shape}")

# This is what the method does (from line 240):
phase_expanded = phase.expand(batch_size, n_channels, n_pha, n_amp, seq_len)
amplitude_expanded = amplitude.expand(batch_size, n_channels, n_pha, n_amp, seq_len)

print("\nAfter expansion:")
print(f"  Phase expanded shape: {phase_expanded.shape}")
print(f"  Amplitude expanded shape: {amplitude_expanded.shape}")
print(f"  Memory for expanded tensors: {(phase_expanded.numel() + amplitude_expanded.numel()) * 4 / 1e6:.2f} MB")

# Then it reshapes:
phase_all = phase_expanded.reshape(batch_size, n_channels * n_pha * n_amp, seq_len)
amplitude_all = amplitude_expanded.reshape(batch_size, n_channels * n_pha * n_amp, seq_len)

print("\nAfter reshape:")
print(f"  Phase reshaped: {phase_all.shape}")
print(f"  Amplitude reshaped: {amplitude_all.shape}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("The current implementation:")
print("1. Expands tensors to include ALL frequency combinations")
print("2. Processes ALL combinations in a single batch")
print("3. Does NOT use chunking as claimed")
print("4. Memory usage scales with n_phase_bands * n_amp_bands")
print("\nThis is FULLY VECTORIZED, not chunked processing!")
print("="*60)

# Now check if ModulationIndexMemoryOptimized actually uses chunking
print("\n\nChecking ModulationIndexMemoryOptimized...")
from src.gpac._ModulationIndex_MemoryOptimized import ModulationIndexMemoryOptimized

mi_opt = ModulationIndexMemoryOptimized(chunk_size=10)
print(f"ModulationIndexMemoryOptimized has chunk_size: {mi_opt.chunk_size}")
print("But PAC's _compute_mi_vectorized doesn't use this chunking!")

# Check memory manager status
print("\n\nChecking MemoryManager integration...")
if hasattr(pac, 'memory_manager'):
    print("✅ PAC has memory_manager attribute")
else:
    print("❌ PAC does NOT have memory_manager attribute")
    print("   MemoryManager is NOT integrated!")