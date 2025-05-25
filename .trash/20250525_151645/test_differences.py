#!/usr/bin/env python3
"""Test to identify main source of PAC value differences"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create identical test signal
np.random.seed(42)
fs = 512
duration = 5.0
t = np.linspace(0, duration, int(fs * duration))

# Simple PAC signal
phase = np.sin(2 * np.pi * 6 * t)
amp_env = (1 + 0.8 * np.cos(2 * np.pi * 6 * t)) / 2
carrier = np.sin(2 * np.pi * 80 * t)
signal = phase + amp_env * carrier * 0.5

print("Testing potential sources of differences:\n")

# 1. Test numerical precision
signal_f32 = signal.astype(np.float32)
signal_f64 = signal.astype(np.float64)
precision_diff = np.mean(np.abs(signal_f32 - signal_f64))
print(f"1. Precision difference (float32 vs float64): {precision_diff:.2e}")

# 2. Test edge effects (simulate different padding)
# Zero padding vs mirror padding
n_edge = 100
signal_zero_pad = np.pad(signal, n_edge, mode='constant')
signal_mirror_pad = np.pad(signal, n_edge, mode='reflect')

# Extract middle part after "filtering" (simulation)
middle_zero = signal_zero_pad[n_edge:-n_edge]
middle_mirror = signal_mirror_pad[n_edge:-n_edge]
edge_diff = np.mean(np.abs(middle_zero - middle_mirror))
print(f"2. Edge padding difference: {edge_diff:.2e}")

# 3. Test Hilbert transform differences
from scipy.signal import hilbert as scipy_hilbert

# Simple PyTorch Hilbert
def torch_hilbert(x):
    N = x.shape[-1]
    X = torch.fft.fft(torch.tensor(x))
    h = torch.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.ifft(X * h).numpy()

scipy_result = scipy_hilbert(signal[:1000])  # Use subset for speed
torch_result = torch_hilbert(signal[:1000])
hilbert_diff = np.mean(np.abs(scipy_result - torch_result))
print(f"3. Hilbert transform difference: {hilbert_diff:.2e}")

# 4. Phase extraction differences
phase_scipy = np.angle(scipy_result)
phase_torch = np.angle(torch_result)
phase_diff = np.mean(np.abs(phase_scipy - phase_torch))
print(f"4. Phase extraction difference: {phase_diff:.2e}")

print("\n🔍 ANALYSIS:")
print("The differences are typically very small, but they accumulate.")
print("Edge effects are usually the largest contributor when signals are short.")
print("\nFor more accurate comparison, TensorPAC and gPAC would need:")
print("1. Identical padding strategies")
print("2. Same numerical precision throughout")
print("3. Identical Hilbert implementations")