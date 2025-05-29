#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:25:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_hilbert_comparison.py

"""
Functionalities:
  - Compares Hilbert transform between gPAC and TensorPAC/SciPy
  - Computes instantaneous amplitude and phase
  - Analyzes computational efficiency
  - Saves comparison figures and performance metrics

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
    - ./scripts/example_hilbert_comparison/hilbert_comparison.png
    - ./scripts/example_hilbert_comparison/hilbert_performance_comparison.png
    - ./scripts/example_hilbert_comparison/hilbert_comparison.csv
    - ./scripts/example_hilbert_comparison/hilbert_performance_comparison.csv
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


def create_test_signals():
    """Create various test signals for Hilbert transform comparison."""
    fs = 512.0
    duration = 2.0
    t = np.arange(0, duration, 1 / fs)

    signals = {}

    # 1. Pure sinusoid
    signals["sinusoid"] = {
        "signal": np.sin(2 * np.pi * 10 * t),
        "description": "10 Hz sinusoid",
    }

    # 2. Amplitude modulated signal
    carrier_freq = 50.0
    mod_freq = 5.0
    signals["am_signal"] = {
        "signal": (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
        * np.sin(2 * np.pi * carrier_freq * t),
        "description": f"AM signal: {carrier_freq} Hz carrier, {mod_freq} Hz modulation",
    }

    # 3. Chirp signal
    signals["chirp"] = {
        "signal": sp_signal.chirp(t, 10.0, duration, 100.0),
        "description": "Linear chirp: 10-100 Hz",
    }

    # 4. Multi-component signal
    signals["multi_component"] = {
        "signal": (
            np.sin(2 * np.pi * 10 * t)
            + 0.5 * np.sin(2 * np.pi * 25 * t)
            + 0.3 * np.sin(2 * np.pi * 60 * t)
        ),
        "description": "Multi-component: 10, 25, 60 Hz",
    }

    return signals, t, fs


def compute_analytic_properties(analytic_signal):
    """Compute instantaneous amplitude and phase from analytic signal."""
    amplitude = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)
    return amplitude, phase


def main(args):
    """Run Hilbert transform comparison."""
    import mngs
    from gpac import Hilbert
    from gpac._Profiler import create_profiler

    # Try importing tensorpac
    try:
        from tensorpac.methods import _hilbert

        TENSORPAC_AVAILABLE = True
    except ImportError:
        TENSORPAC_AVAILABLE = False
        mngs.str.printc(
            "⚠️  TensorPAC not available - comparing with SciPy instead", c="yellow"
        )

    mngs.str.printc(
        "🚀 Hilbert Transform Comparison: gPAC vs Standard Implementations", c="green"
    )
    mngs.str.printc("=" * 70, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mngs.str.printc(f"📍 Using device: {device}", c="cyan")

    # Create test signals
    mngs.str.printc("\n📡 Creating test signals...", c="cyan")
    test_signals, time, fs = create_test_signals()

    # Calculate sequence length
    duration = 2.0  # from create_test_signals
    seq_len = int(fs * duration)
    
    # Initialize gPAC Hilbert
    with profiler.profile("gPAC Hilbert Initialization"):
        gpac_hilbert = Hilbert(seq_len=seq_len)
        gpac_hilbert = gpac_hilbert.to(device)

    # Create figure for results
    n_signals = len(test_signals)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_signals, 4, figsize=(16, 14))
    axes = axes.flatten()  # Flatten for 1D indexing

    # Store results for analysis
    correlations = {}

    # Process each test signal
    for idx, (signal_name, signal_info) in enumerate(test_signals.items()):
        mngs.str.printc(f"\n🔄 Processing: {signal_info['description']}", c="blue")

        signal = signal_info["signal"]

        # gPAC Hilbert transform
        with profiler.profile(f"gPAC Hilbert - {signal_name}"):
            # Convert to torch and add batch dimensions
            signal_torch = torch.from_numpy(signal).float().to(device)
            signal_torch = (
                signal_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [1, 1, 1, n_times]

            # Apply Hilbert transform
            gpac_analytic = gpac_hilbert(signal_torch)
            gpac_analytic_np = gpac_analytic[0, 0, 0].cpu().numpy()

        # SciPy Hilbert transform (reference)
        with profiler.profile(f"SciPy Hilbert - {signal_name}"):
            scipy_analytic = sp_signal.hilbert(signal)

        # TensorPAC Hilbert transform (if available)
        if TENSORPAC_AVAILABLE:
            with profiler.profile(f"TensorPAC Hilbert - {signal_name}"):
                # TensorPAC expects 2D input
                tensorpac_analytic = _hilbert(signal.reshape(1, -1))[0]
        else:
            tensorpac_analytic = None

        # Compute instantaneous properties
        gpac_amp, gpac_phase = compute_analytic_properties(gpac_analytic_np)
        scipy_amp, scipy_phase = compute_analytic_properties(scipy_analytic)

        if tensorpac_analytic is not None:
            tensorpac_amp, tensorpac_phase = compute_analytic_properties(
                tensorpac_analytic
            )

        # Plotting
        # Original signal
        ax = axes[idx * 4]
        ax.plot(time[:256], signal[:256], "k-", alpha=0.8)
        ax.set_xyt(
            "Time (s)", "Amplitude", f'{signal_info["description"]}\nOriginal Signal'
        )
        ax.grid(True, alpha=0.3)

        # Instantaneous amplitude comparison
        ax = axes[idx * 4 + 1]
        ax.plot(
            time[:256], scipy_amp[:256], "g-", label="SciPy", alpha=0.8, linewidth=2
        )
        ax.plot(time[:256], gpac_amp[:256], "b--", label="gPAC", alpha=0.8)
        if tensorpac_analytic is not None:
            ax.plot(time[:256], tensorpac_amp[:256], "r:", label="TensorPAC", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Instantaneous Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Instantaneous phase comparison
        ax = axes[idx * 4 + 2]
        ax.plot(
            time[:256], scipy_phase[:256], "g-", label="SciPy", alpha=0.8, linewidth=2
        )
        ax.plot(time[:256], gpac_phase[:256], "b--", label="gPAC", alpha=0.8)
        if tensorpac_analytic is not None:
            ax.plot(
                time[:256], tensorpac_phase[:256], "r:", label="TensorPAC", alpha=0.8
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Phase (rad)")
        ax.set_title("Instantaneous Phase")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error analysis
        ax = axes[idx * 4 + 3]

        # Amplitude error
        amp_error_gpac = np.abs(gpac_amp - scipy_amp)
        ax.semilogy(
            time[:256], amp_error_gpac[:256], "b-", label="gPAC amp error", alpha=0.8
        )

        # Phase error (accounting for phase wrapping)
        phase_error_gpac = np.abs(np.angle(np.exp(1j * (gpac_phase - scipy_phase))))
        ax.semilogy(
            time[:256],
            phase_error_gpac[:256],
            "b--",
            label="gPAC phase error",
            alpha=0.8,
        )

        if tensorpac_analytic is not None:
            amp_error_tensorpac = np.abs(tensorpac_amp - scipy_amp)
            phase_error_tensorpac = np.abs(
                np.angle(np.exp(1j * (tensorpac_phase - scipy_phase)))
            )
            ax.semilogy(
                time[:256],
                amp_error_tensorpac[:256],
                "r-",
                label="TensorPAC amp error",
                alpha=0.8,
            )
            ax.semilogy(
                time[:256],
                phase_error_tensorpac[:256],
                "r--",
                label="TensorPAC phase error",
                alpha=0.8,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Error vs SciPy Reference")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compute and store correlations
        amp_corr_gpac = np.corrcoef(gpac_amp, scipy_amp)[0, 1]
        phase_corr_gpac = np.corrcoef(gpac_phase, scipy_phase)[0, 1]

        correlations[f"{signal_name}_amp_gpac"] = amp_corr_gpac
        correlations[f"{signal_name}_phase_gpac"] = phase_corr_gpac

        mngs.str.printc(
            f"  Amplitude correlation (gPAC vs SciPy): {amp_corr_gpac:.6f}", c="cyan"
        )
        mngs.str.printc(
            f"  Phase correlation (gPAC vs SciPy): {phase_corr_gpac:.6f}", c="cyan"
        )

        if tensorpac_analytic is not None:
            amp_corr_tensorpac = np.corrcoef(tensorpac_amp, scipy_amp)[0, 1]
            phase_corr_tensorpac = np.corrcoef(tensorpac_phase, scipy_phase)[0, 1]
            correlations[f"{signal_name}_amp_tensorpac"] = amp_corr_tensorpac
            correlations[f"{signal_name}_phase_tensorpac"] = phase_corr_tensorpac
            mngs.str.printc(
                f"  Amplitude correlation (TensorPAC vs SciPy): {amp_corr_tensorpac:.6f}",
                c="cyan",
            )
            mngs.str.printc(
                f"  Phase correlation (TensorPAC vs SciPy): {phase_corr_tensorpac:.6f}",
                c="cyan",
            )

    # Save main comparison figure
    spath = "hilbert_comparison.png"
    mngs.io.save(fig, spath)

    # Performance comparison figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if axes.ndim == 1:
        pass  # Already 1D
    else:
        axes = axes.flatten()

    # Extract timing data
    timings = {"gPAC": [], "SciPy": [], "TensorPAC": []}
    for signal_name in test_signals.keys():
        for result in profiler.results:
            if f"gPAC Hilbert - {signal_name}" in result.name:
                timings["gPAC"].append(result.duration)
            elif f"SciPy Hilbert - {signal_name}" in result.name:
                timings["SciPy"].append(result.duration)
            elif (
                TENSORPAC_AVAILABLE
                and f"TensorPAC Hilbert - {signal_name}" in result.name
            ):
                timings["TensorPAC"].append(result.duration)

    # Bar plot of timings
    methods = list(timings.keys())
    if not TENSORPAC_AVAILABLE:
        methods.remove("TensorPAC")

    avg_times = [np.mean(timings[m]) if timings[m] else 0 for m in methods]
    colors = ["blue", "green", "red"][: len(methods)]

    ax = axes[0]
    bars = ax.bar(methods, avg_times, color=colors, alpha=0.7)
    ax.set_xlabel(None)
        ax.set_ylabel("Average Time (s)")
        ax.set_title("Hilbert Transform Performance Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time*1000:.2f}ms",
            ha="center",
            va="bottom",
        )

    # Speedup comparison
    ax = axes[1]
    if avg_times[0] > 0:  # gPAC time
        speedups = [avg_times[i] / avg_times[0] for i in range(len(methods))]
        bars = ax.bar(methods, speedups, color=colors, alpha=0.7)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(None)
        ax.set_ylabel("Speedup Factor (vs gPAC)")
        ax.set_title("Relative Performance")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
            )

    # Save performance figure
    perf_spath = (
        "hilbert_performance_comparison.png"
    )
    mngs.io.save(fig, perf_spath)

    # Save correlations
    mngs.io.save(correlations, "correlations.csv")

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 70, c="green")
    profiler.print_summary()

    mngs.str.printc(f"\n✅ Comparison complete! Results saved to:", c="green")
    mngs.str.printc(f"   - {spath}", c="green")
    mngs.str.printc(f"   - {perf_spath}", c="green")

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
        description="Compare Hilbert transform between gPAC and TensorPAC/SciPy"
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
