#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 18:37:42 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/example__Hilbert.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/example__Hilbert.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionality:
    * Demonstrates Hilbert transform functionality with differentiability testing
Input:
    * No external input required - generates synthetic test signals
Output:
    * Visualization plots showing phase, amplitude, and frequency analysis
    * Performance metrics and gradient verification results
Prerequisites:
    * torch, numpy, matplotlib, gpac._Hilbert module
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpac._Hilbert import Hilbert


def run_main():
    sample_rate = 1_000
    duration = 2.0
    time_steps = int(sample_rate * duration)
    t_vals = torch.linspace(0, duration, time_steps)

    signal = (
        torch.sin(2 * np.pi * 10 * t_vals)
        + 0.5 * torch.sin(2 * np.pi * 30 * t_vals)
        + 0.3 * torch.cos(2 * np.pi * 50 * t_vals)
    )
    signal += 0.1 * torch.randn_like(signal)

    hilbert = Hilbert(seq_len=time_steps, dim=-1, fp16=False)

    signal_grad = signal.clone().requires_grad_(True)
    result = hilbert(signal_grad)
    phase, amplitude = result[..., 0], result[..., 1]

    loss = amplitude.mean()
    loss.backward()

    print(f"Signal shape: {signal.shape}")
    print(f"Phase shape: {phase.shape}")
    print(f"Amplitude shape: {amplitude.shape}")
    print(f"Gradient exists: {signal_grad.grad is not None}")
    print(f"Gradient norm: {signal_grad.grad.norm().item():.6f}")

    analytic = hilbert.get_analytic_signal(signal)
    instantaneous_freq = (
        torch.diff(unwrap_phase(phase)) * sample_rate / (2 * np.pi)
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(t_vals.numpy(), signal.detach().numpy())
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_vals.numpy(), amplitude.detach().numpy())
    axes[1].set_title("Instantaneous Amplitude")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t_vals.numpy(), phase.detach().numpy())
    axes[2].set_title("Instantaneous Phase")
    axes[2].set_ylabel("Phase (rad)")

    axes[3].plot(t_vals[1:].numpy(), instantaneous_freq.detach().numpy())
    axes[3].set_title("Instantaneous Frequency")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(__DIR__, "hilbert_example.png"), dpi=150)
    plt.show()

    batch_signal = torch.stack([signal, signal * 0.8, signal * 1.2])
    batch_result = hilbert(batch_signal)
    print(f"Batch input shape: {batch_signal.shape}")
    print(f"Batch output shape: {batch_result.shape}")

    print(
        f"Real part matches original: {torch.allclose(analytic.real, signal, atol=1e-5)}"
    )
    print(f"Amplitude is positive: {(amplitude >= 0).all()}")

    large_signal = torch.randn(10_000)
    hilbert_large = Hilbert(seq_len=10_000)

    start_time = time.time()
    for _ in range(100):
        _ = hilbert_large(large_signal)
    end_time = time.time()

    print(
        f"Average processing time (10k samples): {(end_time - start_time) / 100 * 1_000:.2f} ms"
    )


def unwrap_phase(phase):
    diff = torch.diff(phase)
    diff = ((diff + torch.pi) % (2 * torch.pi)) - torch.pi
    return torch.cat([phase[:1], phase[:1] + torch.cumsum(diff, dim=0)])


if __name__ == "__main__":
    run_main()

# alias cb="tee >(xsel --clipboard --input)"

# EOF
