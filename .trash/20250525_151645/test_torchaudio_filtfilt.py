#!/usr/bin/env python3
"""Test torchaudio's filtfilt implementation for gPAC"""

import torch
import torch.nn.functional as F
import time
import numpy as np

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
    print("✅ torchaudio is available")
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("❌ torchaudio not available")

# Our implementations for comparison
def conv1d_filtfilt(signal, kernel, padding='same'):
    """Manual filtfilt-style implementation"""
    filtered = F.conv1d(signal, kernel, padding=padding)
    filtered_back = F.conv1d(filtered.flip(-1), kernel, padding=padding).flip(-1)
    return filtered_back

def torchaudio_filtfilt_wrapper(signal, kernel):
    """
    Wrapper to use torchaudio.functional.filtfilt with our conv kernels.
    Note: filtfilt expects IIR filter coefficients (b, a), but we have FIR kernels.
    For FIR filters, a = [1.0] and b = kernel
    """
    # signal shape: (batch, channels, time)
    # kernel shape: (out_channels, in_channels, kernel_length)
    
    batch, n_ch, seq_len = signal.shape
    n_filters = kernel.shape[0]
    
    # Process each filter separately (torchaudio expects 1D or 2D input)
    results = []
    for i in range(n_filters):
        # Get FIR coefficients for this filter
        b = kernel[i, 0, :].flip(0)  # Flip for convolution->filter convention
        a = torch.tensor([1.0], device=kernel.device)
        
        # Apply filtfilt to all batch items
        # torchaudio expects (batch, time) or (time,)
        filtered = torchaudio.functional.filtfilt(
            signal[:, 0, :],  # (batch, time)
            a, b,
            clamp=False
        )
        results.append(filtered)
    
    # Stack results: (n_filters, batch, time) -> (batch, n_filters, time)
    return torch.stack(results, dim=1).transpose(0, 1)

# Test setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on: {device}")

# Create test data
batch_size = 10
seq_len = 10000
n_filters = 50  # Number of frequency bands
kernel_len = 101

signal = torch.randn(batch_size, 1, seq_len).to(device)
kernels = torch.randn(n_filters, 1, kernel_len).to(device)

# Normalize kernels (important for stability)
kernels = kernels / kernels.sum(dim=-1, keepdim=True)

print(f"\nTest configuration:")
print(f"- Batch size: {batch_size}")
print(f"- Sequence length: {seq_len}")
print(f"- Number of filters: {n_filters}")
print(f"- Filter length: {kernel_len}")

# Warm up
for _ in range(10):
    _ = F.conv1d(signal, kernels, padding='same')
    if device == 'cuda':
        torch.cuda.synchronize()

# Timing tests
n_runs = 50
print(f"\n⏱️  Timing {n_runs} runs each:")
print("-" * 60)

# 1. Standard conv1d (baseline)
times = []
for _ in range(n_runs):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    result_standard = F.conv1d(signal, kernels, padding='same')
    
    if device == 'cuda':
        torch.cuda.synchronize()
    times.append(time.time() - start)

time_standard = np.mean(times[10:])
print(f"Standard conv1d:              {time_standard*1000:.3f} ms (baseline)")

# 2. Manual filtfilt implementation
times = []
for _ in range(n_runs):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    result_manual = conv1d_filtfilt(signal, kernels)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    times.append(time.time() - start)

time_manual = np.mean(times[10:])
print(f"Manual filtfilt (2x conv1d):  {time_manual*1000:.3f} ms ({time_manual/time_standard:.1f}x)")

# 3. Torchaudio filtfilt (if available)
if TORCHAUDIO_AVAILABLE:
    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        result_torchaudio = torchaudio_filtfilt_wrapper(signal, kernels)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    time_torchaudio = np.mean(times[10:])
    print(f"Torchaudio filtfilt:          {time_torchaudio*1000:.3f} ms ({time_torchaudio/time_standard:.1f}x)")

print("\n📊 PERFORMANCE SUMMARY:")
if TORCHAUDIO_AVAILABLE:
    print(f"- Manual filtfilt is {time_manual/time_standard:.1f}x slower than conv1d")
    print(f"- Torchaudio filtfilt is {time_torchaudio/time_standard:.1f}x slower than conv1d")
    print(f"- Torchaudio vs Manual: {time_torchaudio/time_manual:.2f}x")
    
    if time_torchaudio > time_manual:
        print("\n⚠️  Note: torchaudio's filtfilt might be slower here because:")
        print("   1. It's designed for IIR filters (more general)")
        print("   2. We're processing many filters sequentially")
        print("   3. Our manual version is optimized for parallel FIR filtering")
else:
    print("❌ Cannot test torchaudio - not installed")

print("\n💡 RECOMMENDATION:")
print("For gPAC with many parallel FIR filters:")
print("- Stick with optimized parallel conv1d approach")
print("- Add optional filtfilt mode for exact TensorPAC matching")
print("- The 2x overhead is acceptable given overall speedup")