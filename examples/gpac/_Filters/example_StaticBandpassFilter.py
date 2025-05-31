#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 19:06:59 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/_Filters/example_StaticBandpassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/_Filters/example_StaticBandpassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Time-stamp: "2025-01-30 13:20:00"
# Author: ywatanabe

"""
Example demonstrating the StaticBandPassFilter for fixed-frequency filtering.
This filter is computationally efficient for non-learnable filtering tasks.
"""


import argparse
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from gpac._Filters import StaticBandPassFilter


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

    # Generate synthetic EEG-like signal
    fs = 1000  # Sampling frequency
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Create signal with known frequency components
    # Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
    signal = (
        2 * np.sin(2 * np.pi * 2 * t)  # Delta
        + 1.5 * np.sin(2 * np.pi * 6 * t)  # Theta
        + 3 * np.sin(2 * np.pi * 10 * t)  # Alpha
        + 1 * np.sin(2 * np.pi * 20 * t)  # Beta
        + 0.5 * np.sin(2 * np.pi * 50 * t)  # Gamma
        + 0.3 * np.random.randn(len(t))  # Noise
    )

    # Define frequency bands
    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 100),
    }

    # Convert signal to torch tensor
    signal_torch = (
        torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    signal_torch = signal_torch.to(device)

    # Apply filters for each band
    filtered_signals = {}
    filters = {}

    mngs.str.printc("\nApplying bandpass filters for EEG bands:", c="yellow")

    for band_name, (low_freq, high_freq) in bands.items():
        # Create static filter
        filter_model = StaticBandPassFilter(
            low_frequency=low_freq,
            high_frequency=high_freq,
            sample_rate=fs,
            filter_length=257,
        ).to(device)

        filters[band_name] = filter_model

        # Apply filter
        with torch.no_grad():
            filtered = filter_model(signal_torch)
            filtered_signals[band_name] = filtered.cpu().numpy().squeeze()

        print(f"  {band_name}: {low_freq}-{high_freq} Hz")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(15, 12))

    # Original signal
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t[:2000], signal[:2000], "k", alpha=0.7)
    ax1.set_title("Original Composite Signal", fontsize=12)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Original spectrum
    ax2 = plt.subplot(3, 2, 2)
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    fft_orig = np.fft.rfft(signal)
    ax2.semilogy(freqs[:200], np.abs(fft_orig[:200]), "k", alpha=0.7)
    ax2.set_title("Original Signal Spectrum", fontsize=12)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)

    # Add band regions
    colors = ["blue", "green", "red", "orange", "purple"]
    for (band_name, (low, high)), color in zip(bands.items(), colors):
        ax2.axvspan(low, high, alpha=0.2, color=color, label=band_name)
    ax2.legend()

    # Filtered signals
    for idx, (band_name, filtered_signal) in enumerate(
        filtered_signals.items()
    ):
        # Time domain
        ax = plt.subplot(3, 2, 3 + idx)
        ax.plot(t[:2000], filtered_signal[:2000], color=colors[idx], alpha=0.8)
        ax.set_title(
            f"{band_name} Band ({bands[band_name][0]}-{bands[band_name][1]} Hz)"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        # Add power information
        power = np.mean(filtered_signal**2)
        ax.text(
            0.95,
            0.95,
            f"Power: {power:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    # Save figure
    fig_path = CONFIG.PATH.FIGURES / "static_filter_eeg_bands.png"
    mngs.io.save(fig, fig_path)
    mngs.str.printc(f"Saved figure: {fig_path}", c="green")

    # Create filter response visualization
    fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (band_name, filter_model) in enumerate(filters.items()):
        if idx >= 5:
            break

        # Get filter coefficients
        with torch.no_grad():
            # Create impulse
            impulse = torch.zeros(1, 1, 1000)
            impulse[0, 0, 500] = 1.0
            impulse = impulse.to(device)

            # Get impulse response
            h = filter_model(impulse).cpu().numpy().squeeze()

        # Frequency response
        H = np.fft.rfft(h)
        freqs_h = np.fft.rfftfreq(len(h), 1 / fs)

        ax = axes[idx]
        ax.plot(freqs_h[:200], 20 * np.log10(np.abs(H[:200]) + 1e-10))
        ax.set_title(f"{band_name} Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.axvline(bands[band_name][0], color="red", linestyle="--", alpha=0.5)
        ax.axvline(bands[band_name][1], color="red", linestyle="--", alpha=0.5)
        ax.set_ylim([-80, 10])

    # Remove extra subplot
    if len(filters) < 6:
        fig2.delaxes(axes[-1])

    plt.tight_layout()

    fig2_path = CONFIG.PATH.FIGURES / "filter_frequency_responses.png"
    mngs.io.save(fig2, fig2_path)
    mngs.str.printc(f"Saved figure: {fig2_path}", c="green")

    # Demonstrate batch processing
    mngs.str.printc("\nDemonstrating batch processing:", c="yellow")
    batch_size = 16
    n_channels = 8
    n_samples = 1000

    # Create batch of multi-channel signals
    batch_signal = torch.randn(batch_size, n_channels, n_samples).to(device)

    # Apply alpha band filter to all channels
    alpha_filter = filters["Alpha"]
    with torch.no_grad():
        batch_filtered = alpha_filter(batch_signal)

    print(f"  Input shape: {batch_signal.shape}")
    print(f"  Output shape: {batch_filtered.shape}")
    print(f"  Processing time would be measured here with proper timing")

    # Print summary
    print("\nStatic Bandpass Filter Summary:")
    print("  - Efficient fixed-frequency filtering")
    print("  - Supports batch processing")
    print("  - No learnable parameters")
    print("  - Ideal for standard EEG band extraction")
    print("  - GPU-accelerated for large datasets")

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.finish(
        CONFIG, sys.stdout, sys.stderr, plt, CC, verbose=False
    )
    return filters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", "-shuffle", type=bool, default=False)
    parser.add_argument("--seed", "-seed", type=int, default=42)
    parser.add_argument("--num_workers", "-num_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)

# EOF
