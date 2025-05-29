#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 02:58:10 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/gpac/_Filters/example_StaticBandpassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/_Filters/example_StaticBandpassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates StaticBandpassFilter usage and visualization
  - Shows how to apply static bandpass filtering to signals
  - Visualizes filter frequency responses
  - Compares time and frequency domain results
  - Saves filtered signal outputs and frequency responses

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch
    - numpy
    - matplotlib

IO:
  - input-files:
    - None (generates synthetic signals)

  - output-files:
    - ./scripts/example_StaticBandpassFilter/static_bandpass_filter_demo.png
    - ./scripts/example_StaticBandpassFilter/static_filter_frequency_response.png
"""

"""Imports"""
import argparse

import numpy as np
import torch

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""
def create_multi_freq_signal(fs=256, duration=2.0):
    """Create a test signal with multiple frequency components."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)

    # Multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t)  # 5 Hz
        + np.sin(2 * np.pi * 10 * t)  # 10 Hz
        + np.sin(2 * np.pi * 40 * t)  # 40 Hz
        + np.sin(2 * np.pi * 80 * t)  # 80 Hz
    )

    return signal, t


def main(args):
    """Run StaticBandpassFilter demonstration."""
    import mngs
    from gpac._Filters._StaticBandpassFilter import StaticBandPassFilter
    from gpac._Profiler import create_profiler

    mngs.str.printc("🚀 StaticBandpassFilter Demonstration", c="green")
    mngs.str.printc("=" * 60, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Parameters
    fs = 256  # Sampling frequency
    duration = 2.0
    n_samples = int(fs * duration)

    # Create test signal
    mngs.str.printc("\n📡 Creating multi-frequency test signal...", c="cyan")
    signal, t = create_multi_freq_signal(fs, duration)
    mngs.str.printc("Signal components: 5, 10, 40, 80 Hz", c="cyan")

    # Define frequency bands
    bands = torch.tensor(
        [
            [3, 7],  # Around 5 Hz
            [8, 12],  # Around 10 Hz
            [35, 45],  # Around 40 Hz
            [75, 85],  # Around 80 Hz
        ]
    )

    mngs.str.printc("\n🎯 Filter bands:", c="yellow")
    for i, (low, high) in enumerate(bands):
        mngs.str.printc(f"  Band {i+1}: {low:.0f}-{high:.0f} Hz", c="yellow")

    # Create filter
    mngs.str.printc("\n🔧 Creating StaticBandpassFilter...", c="blue")

    with profiler.profile("Filter Initialization"):
        filter_model = StaticBandPassFilter(
            bands=bands, fs=fs, seq_len=n_samples, window="hamming"
        )

        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        filter_model = filter_model.to(device)

    mngs.str.printc(f"✅ Filter created on {device}", c="green")
    mngs.str.printc(f"  Number of filters: {filter_model.n_filters}", c="cyan")
    mngs.str.printc(f"  Filter length: {filter_model.filter_length}", c="cyan")
    mngs.str.printc(f"  Window type: {filter_model.window_type}", c="cyan")

    # Apply filtering
    mngs.str.printc("\n🔄 Applying bandpass filtering...", c="blue")

    with profiler.profile("Filtering"):
        signal_torch = (
            torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        signal_torch = signal_torch.to(device)
        filtered = filter_model(signal_torch)

    mngs.str.printc(f"✅ Filtered output shape: {filtered.shape}", c="green")

    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")

    n_bands = len(bands)
    fig, axes = mngs.plt.subplots(
        n_bands + 1, 2, figsize=(12, 3 * (n_bands + 1))
    )

    # Original signal
    ax = axes[0, 0]
    ax.plot(t[:256], signal[:256])
    ax.set_xyt("Time (s)", "Amplitude", "Original Signal (first 1s)")
    ax.grid(True)

    # Original signal spectrum
    ax = axes[0, 1]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)[: n_samples // 2]
    fft = np.abs(np.fft.fft(signal))[: n_samples // 2]
    ax.plot(freqs, fft)
    ax.set_xyt("Frequency (Hz)", "Magnitude", "Original Signal Spectrum")
    ax.set_xlim(0, 100)
    ax.grid(True)

    # Filtered signals
    filtered_cpu = filtered.cpu()
    for i, (low, high) in enumerate(bands):
        filtered_signal = filtered_cpu[0, 0, i].numpy()

        # Time domain
        ax = axes[i + 1, 0]
        ax.plot(t[:256], filtered_signal[:256])
        ax.set_xyt(
            "Time (s)", "Amplitude", f"Band {i+1}: {low:.0f}-{high:.0f} Hz"
        )
        ax.grid(True)

        # Frequency domain
        ax = axes[i + 1, 1]
        fft_filtered = np.abs(np.fft.fft(filtered_signal))[: n_samples // 2]
        ax.plot(freqs, fft_filtered)
        ax.set_xyt("Frequency (Hz)", "Magnitude", f"Band {i+1} Spectrum")
        ax.set_xlim(0, 100)
        ax.axvspan(low, high, alpha=0.3, color="green", label="Passband")
        ax.grid(True)
        ax.legend()

    # Save figure
    spath = "static_bandpass_filter_demo.png"
    mngs.io.save(fig, spath, symlink_from_cwd=True)

    # Create filter frequency response figure
    mngs.str.printc(
        "\n📊 Creating filter frequency response plot...", c="cyan"
    )

    fig, ax = mngs.plt.subplots(1, 1, figsize=(10, 6))

    for i, (low, high) in enumerate(bands):
        # Get filter kernel
        kernel = filter_model.kernels[i].cpu().numpy()
        # Compute frequency response
        freq_response = np.abs(np.fft.fft(kernel, n=1024))[:512]
        freqs_kernel = np.linspace(0, fs / 2, 512)

        ax.plot(
            freqs_kernel,
            freq_response,
            label=f"Band {i+1}: {low:.0f}-{high:.0f} Hz",
        )

    ax.set_xyt("Frequency (Hz)", "Magnitude", "Filter Frequency Responses")
    ax.set_xlim(0, 100)
    ax.grid(True)
    ax.legend()

    # Save frequency response figure
    spath_freq = "static_filter_frequency_response.png"
    mngs.io.save(fig, spath_freq, symlink_from_cwd=True)

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 60, c="green")
    profiler.print_summary()

    mngs.str.printc("\n✅ StaticBandpassFilter demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to:", c="green")
    mngs.str.printc(f"  - {spath}", c="cyan")
    mngs.str.printc(f"  - {spath_freq}", c="cyan")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="StaticBandpassFilter demonstration"
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
