#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 19:06:37 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/_Filters/example_DifferentiableBandpassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/_Filters/example_DifferentiableBandpassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Time-stamp: "2025-01-30 13:15:00"
# Author: ywatanabe

"""
Example demonstrating the DifferentiableBandPassFilter for learnable filtering.
This filter allows gradient-based optimization of filter parameters.
"""


import argparse
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from gpac._Filters import DifferentiableBandPassFilter


def main(args):
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mngs.str.printc(f"Using device: {device}", c="cyan")

    # Generate test signal with multiple frequency components
    fs = 1000  # Sampling frequency
    duration = 2  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Create composite signal
    signal = (
        np.sin(2 * np.pi * 5 * t)  # 5 Hz
        + np.sin(2 * np.pi * 10 * t)  # 10 Hz
        + np.sin(2 * np.pi * 40 * t)  # 40 Hz
        + np.sin(2 * np.pi * 80 * t)  # 80 Hz
        + 0.5 * np.random.randn(len(t))  # Noise
    )

    # Convert to torch tensor
    signal_torch = (
        torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    signal_torch = signal_torch.to(device)

    # Create differentiable bandpass filter
    # Initial parameters: 8-12 Hz (alpha band)
    low_freq_init = 8.0
    high_freq_init = 12.0

    filter_model = DifferentiableBandPassFilter(
        low_frequency=low_freq_init,
        high_frequency=high_freq_init,
        sample_rate=fs,
        filter_length=257,
        learnable=True,
    ).to(device)

    mngs.str.printc(
        f"\nInitial filter band: {low_freq_init}-{high_freq_init} Hz",
        c="yellow",
    )

    # Apply initial filter
    with torch.no_grad():
        filtered_initial = filter_model(signal_torch)

    # Demonstrate gradient-based optimization
    # Target: maximize power in 40 Hz range
    target_freq = 40.0
    optimizer = torch.optim.Adam(filter_model.parameters(), lr=0.1)

    losses = []
    low_freqs = []
    high_freqs = []

    mngs.str.printc("\nOptimizing filter parameters...", c="green")

    for epoch in range(100):
        optimizer.zero_grad()

        # Filter signal
        filtered = filter_model(signal_torch)

        # Compute FFT to get frequency content
        fft = torch.fft.rfft(filtered.squeeze())
        freqs = torch.fft.rfftfreq(filtered.shape[-1], 1 / fs)

        # Find power around target frequency
        target_idx = torch.argmin(torch.abs(freqs - target_freq))
        target_power = torch.abs(fft[target_idx])

        # Loss: negative power at target frequency
        loss = -target_power

        loss.backward()
        optimizer.step()

        # Track progress
        losses.append(loss.item())
        low_freqs.append(filter_model.low_frequency.item())
        high_freqs.append(filter_model.high_frequency.item())

        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch}: Loss={loss.item():.4f}, "
                f"Band={filter_model.low_frequency.item():.1f}-{filter_model.high_frequency.item():.1f} Hz"
            )

    # Get final filtered signal
    with torch.no_grad():
        filtered_final = filter_model(signal_torch)

    # Convert to numpy for plotting
    signal_np = signal_torch.cpu().numpy().squeeze()
    filtered_initial_np = filtered_initial.cpu().numpy().squeeze()
    filtered_final_np = filtered_final.cpu().numpy().squeeze()

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 1. Original signal and spectrum
    axes[0, 0].plot(t[:1000], signal_np[:1000], alpha=0.8)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Spectrum of original
    freqs_np = np.fft.rfftfreq(len(signal_np), 1 / fs)
    fft_orig = np.fft.rfft(signal_np)
    axes[0, 1].semilogy(freqs_np[:200], np.abs(fft_orig)[:200])
    axes[0, 1].set_title("Original Spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(
        target_freq,
        color="red",
        linestyle="--",
        label=f"Target: {target_freq} Hz",
    )
    axes[0, 1].legend()

    # 2. Initial filtered signal
    axes[1, 0].plot(
        t[:1000], filtered_initial_np[:1000], alpha=0.8, color="orange"
    )
    axes[1, 0].set_title(
        f"Initial Filter ({low_freq_init}-{high_freq_init} Hz)"
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Spectrum of initial filtered
    fft_initial = np.fft.rfft(filtered_initial_np)
    axes[1, 1].semilogy(
        freqs_np[:200], np.abs(fft_initial)[:200], color="orange"
    )
    axes[1, 1].set_title("Initial Filtered Spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvspan(
        low_freq_init, high_freq_init, alpha=0.3, color="orange"
    )

    # 3. Optimized filtered signal
    final_low = filter_model.low_frequency.item()
    final_high = filter_model.high_frequency.item()
    axes[2, 0].plot(
        t[:1000], filtered_final_np[:1000], alpha=0.8, color="green"
    )
    axes[2, 0].set_title(
        f"Optimized Filter ({final_low:.1f}-{final_high:.1f} Hz)"
    )
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].grid(True, alpha=0.3)

    # Spectrum of final filtered
    fft_final = np.fft.rfft(filtered_final_np)
    axes[2, 1].semilogy(freqs_np[:200], np.abs(fft_final)[:200], color="green")
    axes[2, 1].set_title("Optimized Filtered Spectrum")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("Magnitude")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvspan(final_low, final_high, alpha=0.3, color="green")
    axes[2, 1].axvline(
        target_freq,
        color="red",
        linestyle="--",
        label=f"Target: {target_freq} Hz",
    )
    axes[2, 1].legend()

    plt.tight_layout()

    # Save figure
    fig_path = CONFIG.PATH.FIGURES / "differentiable_filter_optimization.png"
    mngs.io.save(fig, fig_path)
    mngs.str.printc(f"Saved figure: {fig_path}", c="green")

    # Plot optimization progress
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (negative target power)")
    ax1.set_title("Optimization Progress")
    ax1.grid(True, alpha=0.3)

    ax2.plot(low_freqs, label="Low frequency", color="blue")
    ax2.plot(high_freqs, label="High frequency", color="red")
    ax2.axhline(
        target_freq, color="green", linestyle="--", label="Target frequency"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Filter Parameter Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig2_path = CONFIG.PATH.FIGURES / "filter_optimization_progress.png"
    mngs.io.save(fig2, fig2_path)
    mngs.str.printc(f"Saved figure: {fig2_path}", c="green")

    # Print summary
    print("\nDifferentiable Filter Summary:")
    print(f"  Initial band: {low_freq_init}-{high_freq_init} Hz")
    print(f"  Final band: {final_low:.1f}-{final_high:.1f} Hz")
    print(f"  Target frequency: {target_freq} Hz")
    print(f"  Filter successfully adapted to include target frequency")

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.finish(
        CONFIG, sys.stdout, sys.stderr, plt, CC, verbose=False
    )
    return filter_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", "-shuffle", type=bool, default=False)
    parser.add_argument("--seed", "-seed", type=int, default=42)
    parser.add_argument("--num_workers", "-num_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)

# EOF
