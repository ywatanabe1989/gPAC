#!/usr/bin/env python3
"""Simple test of filtfilt performance impact"""

import torch
import torch.nn.functional as F
import time
import numpy as np

# Test setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing on: {device}")
print()

# Realistic gPAC parameters
batch_size = 1
seq_len = 5000  # 5 seconds at 1000 Hz
n_phase_filters = 50
n_amp_filters = 30
total_filters = n_phase_filters + n_amp_filters

# Create test signal and filters
signal = torch.randn(batch_size, 1, seq_len).to(device)
# Phase filters (shorter)
phase_kernels = torch.randn(n_phase_filters, 1, 61).to(device)
# Amplitude filters (longer)
amp_kernels = torch.randn(n_amp_filters, 1, 121).to(device)

# Warm up GPU
for _ in range(10):
    _ = F.conv1d(signal, phase_kernels[:1], padding='same')
    if device == 'cuda':
        torch.cuda.synchronize()

print("🔬 PERFORMANCE TEST: Single vs Double Pass Filtering")
print("="*50)

# Time single-pass filtering (current gPAC approach)
n_runs = 100
times_single = []

for _ in range(n_runs):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    # Filter with phase bands
    phase_filtered = F.conv1d(signal, phase_kernels, padding='same')
    # Filter with amplitude bands  
    amp_filtered = F.conv1d(signal, amp_kernels, padding='same')
    
    if device == 'cuda':
        torch.cuda.synchronize()
    times_single.append(time.time() - start)

time_single = np.mean(times_single[10:]) * 1000  # Convert to ms

# Time double-pass filtering (filtfilt-style)
times_double = []

for _ in range(n_runs):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    # Phase bands - forward and backward
    phase_fwd = F.conv1d(signal, phase_kernels, padding='same')
    phase_bwd = F.conv1d(signal.flip(-1), phase_kernels, padding='same').flip(-1)
    
    # Amplitude bands - forward and backward
    amp_fwd = F.conv1d(signal, amp_kernels, padding='same')
    amp_bwd = F.conv1d(signal.flip(-1), amp_kernels, padding='same').flip(-1)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    times_double.append(time.time() - start)

time_double = np.mean(times_double[10:]) * 1000  # Convert to ms

print(f"Single-pass filtering: {time_single:.3f} ms")
print(f"Double-pass filtering: {time_double:.3f} ms")
print(f"Overhead: {time_double/time_single:.2f}x slower")
print()

# Estimate impact on full PAC computation
print("📊 ESTIMATED IMPACT ON FULL PAC:")
print("-"*40)

# Rough breakdown of PAC computation time
# Based on our earlier measurements
if device == 'cuda':
    total_pac_time = 2.0  # ~2ms for full PAC on GPU
    filtering_fraction = 0.3  # ~30% of time is filtering
else:
    total_pac_time = 50.0  # ~50ms on CPU
    filtering_fraction = 0.2  # ~20% of time is filtering

current_filter_time = total_pac_time * filtering_fraction
new_filter_time = current_filter_time * (time_double/time_single)
new_total_time = total_pac_time - current_filter_time + new_filter_time

print(f"Current total PAC time: {total_pac_time:.1f} ms")
print(f"With filtfilt: {new_total_time:.1f} ms")
print(f"Overall slowdown: {new_total_time/total_pac_time:.2f}x")
print()

print("🎯 CONCLUSION:")
if new_total_time/total_pac_time < 1.5:
    print("✅ Filtfilt overhead is acceptable (<50% total slowdown)")
    print("   Still much faster than TensorPAC!")
else:
    print("⚠️  Filtfilt adds significant overhead (>50% slowdown)")
    print("   Consider making it optional for accuracy vs speed")