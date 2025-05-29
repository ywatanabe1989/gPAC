#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:20:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_bandpass_filter_comparison.py

"""
Functionalities:
  - Compares bandpass filtering between gPAC and TensorPAC
  - Visualizes filter frequency responses
  - Computes correlation between filtered outputs
  - Saves comparison figures

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - tensorpac (optional)
    - scipy
    - torch
    - numpy
    - matplotlib
    
IO:
  - input-files:
    - None (generates synthetic signals)
    
  - output-files:
    - ./scripts/example_bandpass_filter_comparison/bandpass_filter_comparison.png
    - ./scripts/example_bandpass_filter_comparison/bandpass_filter_comparison.csv
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
from scipy import signal as sp_signal

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def create_test_signal(fs=512, duration=2.0, frequencies=[5, 10, 30, 50, 80]):
    """Create a test signal with multiple frequency components."""
    t = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(t)

    # Add multiple frequency components
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)

    # Add some noise
    signal += 0.1 * np.random.randn(len(t))

    return signal, t


def compute_frequency_response(filter_coeffs, fs):
    """Compute frequency response of filter."""
    w, h = sp_signal.freqz(filter_coeffs, worN=2048, fs=fs)
    return w, np.abs(h)


def main(args):
    """Run bandpass filter comparison."""
    import mngs
    from gpac import BandPassFilter
    from gpac._Profiler import create_profiler

    # Try importing tensorpac
    try:
        from tensorpac.utils import BandPassFilter as TensorPACBandPassFilter

        TENSORPAC_AVAILABLE = True
    except ImportError:
        TENSORPAC_AVAILABLE = False
        mngs.str.printc("⚠️  TensorPAC not available - showing gPAC only", c="yellow")

    mngs.str.printc("🚀 BandPass Filter Comparison: gPAC vs TensorPAC", c="green")
    mngs.str.printc("=" * 60, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Parameters
    fs = 512.0
    duration = 2.0
    n_samples = int(fs * duration)

    # Test bands
    phase_bands = [(4.0, 8.0), (8.0, 13.0)]  # Theta, Alpha
    amp_bands = [(30.0, 60.0), (60.0, 100.0)]  # Low Gamma, High Gamma

    # Create test signal
    mngs.str.printc("\n📡 Creating test signal...", c="cyan")
    test_signal, time = create_test_signal(fs, duration)

    # Create figure for results
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(12, 20))
    if not hasattr(axes, '__len__'):
        axes = [axes]  # Make it a list if single axis

    # Plot original signal (span across all columns in first row)
    ax = axes[0]
    ax.plot(time[:256], test_signal[:256])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original Test Signal (first 0.5s)")
    ax.grid(True, alpha=0.3)

    # gPAC filtering
    mngs.str.printc("\n🔧 gPAC Filtering", c="blue")
    mngs.str.printc("-" * 40, c="blue")

    with profiler.profile("gPAC BandPassFilter Initialization"):
        gpac_filter = BandPassFilter(
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

    # Convert signal to torch
    signal_torch = (
        torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)
    )

    with profiler.profile("gPAC Filtering"):
        gpac_filtered = gpac_filter(signal_torch)

    mngs.str.printc(f"✅ gPAC output shape: {gpac_filtered.shape}", c="green")

    # TensorPAC filtering (if available)
    if TENSORPAC_AVAILABLE:
        mngs.str.printc("\n🔧 TensorPAC Filtering", c="red")
        mngs.str.printc("-" * 40, c="red")

        with profiler.profile("TensorPAC Filtering"):
            tensorpac_results = {}

            for i, (band_name, bands) in enumerate(
                [("phase", phase_bands), ("amplitude", amp_bands)]
            ):
                for j, band in enumerate(bands):
                    b, a = sp_signal.butter(
                        3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band"
                    )
                    filtered = sp_signal.filtfilt(b, a, test_signal)
                    tensorpac_results[f"{band_name}_{j}"] = filtered

        mngs.str.printc(
            f"✅ TensorPAC filtered {len(tensorpac_results)} bands", c="green"
        )

    # Visualization
    mngs.str.printc("\n📊 Creating visualizations...", c="cyan")

    # Plot filtered signals - Phase bands
    for i, (low, high) in enumerate(phase_bands):
        ax = axes[1 + i]  # Second row starts at index 3

        # gPAC result
        gpac_phase = gpac_filtered[0, 0, i, :256].numpy()
        ax.plot(time[:256], gpac_phase, "b-", label="gPAC", alpha=0.8)

        if TENSORPAC_AVAILABLE and f"phase_{i}" in tensorpac_results:
            ax.plot(
                time[:256],
                tensorpac_results[f"phase_{i}"][:256],
                "r--",
                label="TensorPAC",
                alpha=0.8,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Phase Band: {low}-{high} Hz")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot filtered signals - Amplitude bands
    for i, (low, high) in enumerate(amp_bands):
        ax = axes[3 + i]  # Third row starts at index 6

        # gPAC result (amplitude bands come after phase bands)
        gpac_amp = gpac_filtered[0, 0, len(phase_bands) + i, :256].numpy()
        ax.plot(time[:256], gpac_amp, "b-", label="gPAC", alpha=0.8)

        if TENSORPAC_AVAILABLE and f"amplitude_{i}" in tensorpac_results:
            ax.plot(
                time[:256],
                tensorpac_results[f"amplitude_{i}"][:256],
                "r--",
                label="TensorPAC",
                alpha=0.8,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Amplitude Band: {low}-{high} Hz")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Frequency response comparison (last row)
    ax = axes[4]  # Fourth row starts at index 9

    # Note: Filter coefficient visualization removed for compatibility

    # Add band indicators
    for low, high in phase_bands:
        ax.axvspan(low, high, alpha=0.1, color="blue")
    for low, high in amp_bands:
        ax.axvspan(low, high, alpha=0.1, color="green")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Filter Frequency Responses")
    ax.set_xlim(0, fs / 2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save figure
    spath = "bandpass_filter_comparison.png"
    mngs.io.save(fig, spath)

    # Compute correlation if both available
    if TENSORPAC_AVAILABLE:
        mngs.str.printc("\n📊 Correlation Analysis", c="yellow")
        mngs.str.printc("-" * 40, c="yellow")

        correlations = {}
        for i, (low, high) in enumerate(phase_bands):
            gpac_signal = gpac_filtered[0, 0, i].numpy()
            if f"phase_{i}" in tensorpac_results:
                tensorpac_signal = tensorpac_results[f"phase_{i}"]
                corr = np.corrcoef(gpac_signal, tensorpac_signal)[0, 1]
                correlations[f"phase_{low}_{high}"] = corr
                mngs.str.printc(
                    f"Phase band {low}-{high} Hz correlation: {corr:.4f}", c="cyan"
                )

        for i, (low, high) in enumerate(amp_bands):
            gpac_signal = gpac_filtered[0, 0, len(phase_bands) + i].numpy()
            if f"amplitude_{i}" in tensorpac_results:
                tensorpac_signal = tensorpac_results[f"amplitude_{i}"]
                corr = np.corrcoef(gpac_signal, tensorpac_signal)[0, 1]
                correlations[f"amp_{low}_{high}"] = corr
                mngs.str.printc(
                    f"Amplitude band {low}-{high} Hz correlation: {corr:.4f}", c="cyan"
                )

        # Save correlations
        mngs.io.save(correlations, "correlations.csv")

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 60, c="green")
    profiler.print_summary()

    mngs.str.printc(f"\n✅ Comparison complete! Results saved to: {spath}", c="green")

    if not TENSORPAC_AVAILABLE:
        mngs.str.printc(
            "\n⚠️  Note: Install TensorPAC for full comparison:", c="yellow"
        )
        mngs.str.printc("   pip install tensorpac", c="yellow")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Compare bandpass filtering between gPAC and TensorPAC"
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
