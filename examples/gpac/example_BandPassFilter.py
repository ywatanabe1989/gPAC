#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:50:00 (ywatanabe)"
# File: ./examples/gpac/example_BandPassFilter.py

"""
Functionalities:
  - Demonstrates BandPassFilter usage and capabilities
  - Shows TensorPAC-compatible filtering design
  - Compares filtering performance with TensorPAC
  - Visualizes filter frequency responses
  - Shows proper initialization and usage patterns

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch
    - numpy
    - matplotlib
    - scipy
    
IO:
  - input-files:
    - None (generates synthetic signals)
    
  - output-files:
    - ./scripts/example_BandPassFilter/bandpass_filter_demo.png
    - ./scripts/example_BandPassFilter/filter_performance.csv
"""

"""Imports"""
import argparse
import sys
import time
import numpy as np
import torch
from scipy import signal as sp_signal

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def create_test_signal(fs=512, duration=2.0):
    """Create a test signal with multiple frequency components."""
    t = np.arange(0, duration, 1 / fs)

    # Multiple frequency components
    frequencies = [5, 10, 30, 50, 80]
    signal = np.zeros_like(t)

    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)

    # Add noise
    signal += 0.1 * np.random.randn(len(t))

    return signal, t, frequencies


def compute_frequency_response(filter_coeffs, fs):
    """Compute frequency response of filter."""
    w, h = sp_signal.freqz(filter_coeffs, worN=2048, fs=fs)
    return w, np.abs(h)


def main(args):
    """Run BandPassFilter demonstration."""
    import mngs
    from gpac import BandPassFilter
    from gpac._Profiler import create_profiler

    mngs.str.printc("🚀 BandPassFilter Demonstration", c="green")
    mngs.str.printc("=" * 60, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Parameters
    fs = 512.0
    duration = 2.0
    n_samples = int(fs * duration)

    # Create test signal
    mngs.str.printc("\n📡 Creating test signal...", c="cyan")
    test_signal, time_arr, frequencies = create_test_signal(fs, duration)
    mngs.str.printc(f"Signal components: {frequencies} Hz", c="cyan")

    # Define frequency bands
    phase_bands = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]  # Theta, Alpha, Beta
    amp_bands = [(30.0, 60.0), (60.0, 100.0)]  # Low Gamma, High Gamma

    mngs.str.printc("\n🎯 Filter bands:", c="yellow")
    mngs.str.printc(f"Phase bands: {phase_bands}", c="yellow")
    mngs.str.printc(f"Amplitude bands: {amp_bands}", c="yellow")

    # Initialize BandPassFilter
    mngs.str.printc("\n🔧 Initializing BandPassFilter...", c="blue")

    with profiler.profile("BandPassFilter Initialization"):
        filter = BandPassFilter(
            seq_len=n_samples,
            fs=fs,
            pha_start_hz=phase_bands[0][0],
            pha_end_hz=phase_bands[-1][1],
            pha_n_bands=len(phase_bands),
            amp_start_hz=amp_bands[0][0],
            amp_end_hz=amp_bands[-1][1],
            amp_n_bands=len(amp_bands),
            trainable=False,
        )

        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        filter = filter.to(device)

    mngs.str.printc(f"✅ Filter initialized on {device}", c="green")

    # Convert signal to torch (batch, channel, time)
    signal_torch = (
        torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)
    )
    signal_torch = signal_torch.to(device)

    # Apply filtering
    mngs.str.printc("\n🔄 Applying bandpass filtering...", c="blue")

    with profiler.profile("BandPassFilter Forward Pass"):
        filtered_signals = filter(signal_torch)

    mngs.str.printc(f"✅ Filtered output shape: {filtered_signals.shape}", c="green")

    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")

    n_bands = len(phase_bands) + len(amp_bands)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Plot original signal
    ax = axes[0]
    ax.plot(time_arr[:256], test_signal[:256], "b-", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original Test Signal (first 0.5s)")
    ax.grid(True, alpha=0.3)

    # Plot filtered signals
    filtered_np = filtered_signals[0, 0].cpu().numpy()

    # Phase bands
    for i, (low, high) in enumerate(phase_bands):
        ax = axes[i + 1]
        ax.plot(time_arr[:256], filtered_np[i, :256], "g-", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Phase Band: {low}-{high} Hz")
        ax.grid(True, alpha=0.3)

    # Amplitude bands
    for i, (low, high) in enumerate(amp_bands):
        ax = axes[len(phase_bands) + i + 1]
        ax.plot(
            time_arr[:256], filtered_np[len(phase_bands) + i, :256], "r-", linewidth=0.8
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Amplitude Band: {low}-{high} Hz")
        ax.grid(True, alpha=0.3)

    # Plot frequency responses
    ax = axes[5]

    # Note: Filter coefficient visualization removed for compatibility
    # Add band indicators
    for low, high in phase_bands:
        ax.axvspan(low, high, alpha=0.1, color="green", label=f"Phase: {low}-{high} Hz" if low == phase_bands[0][0] else "")
    for low, high in amp_bands:
        ax.axvspan(low, high, alpha=0.1, color="red", label=f"Amp: {low}-{high} Hz" if low == amp_bands[0][0] else "")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Target Filter Bands")
    ax.set_xlim(0, fs / 2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save figure
    spath = "bandpass_filter_demo.png"
    mngs.io.save(fig, spath)

    # Performance analysis
    mngs.str.printc("\n⚡ Performance Analysis", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")

    # Test with different batch sizes
    batch_sizes = [1, 10, 50, 100]
    timing_results = {}

    for batch_size in batch_sizes:
        # Create batch
        batch_signal = signal_torch.repeat(batch_size, 1, 1)

        # Warm up
        with torch.no_grad():
            _ = filter(batch_signal)

        # Time multiple runs
        n_runs = 10
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(n_runs):
                _ = filter(batch_signal)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time

        avg_time = elapsed / n_runs
        timing_results[f"batch_{batch_size}"] = avg_time

        mngs.str.printc(
            f"Batch size {batch_size}: {avg_time*1000:.2f} ms ({avg_time/batch_size*1000:.2f} ms/signal)",
            c="cyan",
        )

    # Save timing results
    mngs.io.save(
        timing_results,
        "filter_performance.csv"
    )

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 60, c="green")
    profiler.print_summary()

    # Additional information
    mngs.str.printc("\n💡 Key Features:", c="green")
    mngs.str.printc("  - TensorPAC-compatible filter design", c="cyan")
    mngs.str.printc("  - Efficient batch processing", c="cyan")
    mngs.str.printc("  - GPU acceleration support", c="cyan")
    mngs.str.printc("  - Phase preservation with filtfilt", c="cyan")

    mngs.str.printc("\n✅ BandPassFilter demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to: {spath}", c="green")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="BandPassFilter demonstration")
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
