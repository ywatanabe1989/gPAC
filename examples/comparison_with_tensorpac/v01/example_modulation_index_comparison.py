#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:30:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_modulation_index_comparison.py

"""
Functionalities:
  - Compares modulation index (MI) calculation between gPAC and TensorPAC
  - Tests MI with different coupling strengths
  - Analyzes computational efficiency
  - Saves comparison figures and correlation metrics

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
    - None (generates synthetic PAC signals)
    
  - output-files:
    - ./scripts/example_modulation_index_comparison/modulation_index_comparison.png
    - ./scripts/example_modulation_index_comparison/modulation_index_summary.png
    - ./scripts/example_modulation_index_comparison/modulation_index_comparison.csv
    - ./scripts/example_modulation_index_comparison/modulation_index_summary.csv
    - ./scripts/example_modulation_index_comparison/correlations.pkl
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


def create_pac_test_signals(fs=512.0, duration=2.0):
    """Create test signals with varying PAC strength."""
    from gpac._SyntheticDataGenerator import generate_pac_signal

    n_times = int(fs * duration)

    # Different coupling strengths
    coupling_strengths = [0.0, 0.3, 0.6, 0.9]
    signals = {}

    for strength in coupling_strengths:
        # Generate PAC signal
        signal = generate_pac_signal(
            n_epochs=5,
            n_channels=1,
            n_times=n_times,
            pha_freq=6.0,  # Theta
            amp_freq=80.0,  # Gamma
            coupling_strength=strength,
            fs=fs,
        )

        signals[f"coupling_{strength}"] = {
            "data": signal,
            "strength": strength,
            "description": f"Coupling strength: {strength}",
        }

    # Add noise-only signal
    signals["noise"] = {
        "data": np.random.randn(5, 1, n_times),
        "strength": 0.0,
        "description": "Pure noise",
    }

    return signals, fs


def compute_phase_amplitude(signal, pha_band, amp_band, fs):
    """Extract phase and amplitude using filtering and Hilbert transform."""
    # Phase signal
    b_pha, a_pha = sp_signal.butter(
        3, [pha_band[0] / (fs / 2), pha_band[1] / (fs / 2)], btype="band"
    )
    pha_filtered = sp_signal.filtfilt(b_pha, a_pha, signal)
    pha_complex = sp_signal.hilbert(pha_filtered)
    phase = np.angle(pha_complex)

    # Amplitude signal
    b_amp, a_amp = sp_signal.butter(
        3, [amp_band[0] / (fs / 2), amp_band[1] / (fs / 2)], btype="band"
    )
    amp_filtered = sp_signal.filtfilt(b_amp, a_amp, signal)
    amp_complex = sp_signal.hilbert(amp_filtered)
    amplitude = np.abs(amp_complex)

    return phase, amplitude


def main(args):
    """Run modulation index comparison."""
    import mngs
    from gpac import ModulationIndex
    from gpac._SyntheticDataGenerator import generate_pac_signal
    from gpac._Profiler import create_profiler

    # Try importing tensorpac
    try:
        from tensorpac import Pac

        TENSORPAC_AVAILABLE = True
    except ImportError:
        TENSORPAC_AVAILABLE = False
        mngs.str.printc("⚠️  TensorPAC not available - showing gPAC only", c="yellow")

    mngs.str.printc("🚀 Modulation Index Comparison: gPAC vs TensorPAC", c="green")
    mngs.str.printc("=" * 60, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mngs.str.printc(f"📍 Using device: {device}", c="cyan")

    # Parameters
    pha_band = (4.0, 8.0)  # Theta
    amp_band = (60.0, 100.0)  # High gamma
    n_bins = 100

    # Create test signals
    mngs.str.printc("\n📡 Creating test signals with varying PAC strength...", c="cyan")
    test_signals, fs = create_pac_test_signals()

    # Initialize gPAC ModulationIndex
    with profiler.profile("gPAC MI Initialization"):
        gpac_mi = ModulationIndex(
            n_bins=n_bins, method="MVL", device=device  # Mean Vector Length
        )

    # Results storage
    results = {"gpac": {}, "tensorpac": {}, "manual": {}}

    # Create figure
    n_signals = len(test_signals)
    fig, axes = mngs.plt.subplots(n_signals, 3, figsize=(16, 10))
    axes = axes.flatten()  # Flatten for 1D indexing

    # Process each test signal
    for idx, (signal_name, signal_info) in enumerate(test_signals.items()):
        mngs.str.printc(f"\n🔄 Processing: {signal_info['description']}", c="blue")

        signal_data = signal_info["data"]
        n_epochs = signal_data.shape[0]

        # Manual phase/amplitude extraction for comparison
        all_phases = []
        all_amplitudes = []

        for epoch in range(n_epochs):
            phase, amplitude = compute_phase_amplitude(
                signal_data[epoch, 0], pha_band, amp_band, fs
            )
            all_phases.append(phase)
            all_amplitudes.append(amplitude)

        phases_np = np.array(all_phases)
        amplitudes_np = np.array(all_amplitudes)

        # gPAC computation
        with profiler.profile(f"gPAC MI - {signal_name}"):
            # Convert to torch
            phases_torch = torch.from_numpy(phases_np).float().to(device)
            amplitudes_torch = torch.from_numpy(amplitudes_np).float().to(device)

            # Add channel dimension
            phases_torch = phases_torch.unsqueeze(1)  # [n_epochs, 1, n_times]
            amplitudes_torch = amplitudes_torch.unsqueeze(1)

            # Compute MI
            gpac_mi_value = gpac_mi(phases_torch, amplitudes_torch)
            gpac_mi_np = gpac_mi_value.cpu().numpy()

        results["gpac"][signal_name] = gpac_mi_np.mean()

        # TensorPAC computation (if available)
        if TENSORPAC_AVAILABLE:
            with profiler.profile(f"TensorPAC MI - {signal_name}"):
                # TensorPAC PAC object
                pac = Pac(
                    idpac=(1, 1, 0), f_pha=pha_band, f_amp=amp_band, n_bins=n_bins
                )

                # Compute PAC
                tensorpac_mi = pac.filterfit(
                    sf=fs,
                    x_pha=signal_data[:, 0],  # [n_epochs, n_times]
                    x_amp=signal_data[:, 0],
                    n_jobs=1,
                )

                results["tensorpac"][signal_name] = tensorpac_mi.mean()

        # Plotting
        # Signal example
        ax = axes[idx * 3]
        time = np.arange(signal_data.shape[2]) / fs
        ax.plot(time[:512], signal_data[0, 0, :512], "k-", alpha=0.8)
        ax.set_xyt(
            "Time (s)", "Amplitude", f"{signal_info['description']}\nSignal Example"
        )
        ax.grid(True, alpha=0.3)

        # Phase-amplitude plot
        ax = axes[idx * 3 + 1]

        # Bin phases and compute mean amplitude per bin
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

        # Use first epoch for visualization
        phase_example = all_phases[0]
        amp_example = all_amplitudes[0]

        # Compute binned amplitudes
        binned_amps = []
        for i in range(n_bins):
            mask = (phase_example >= phase_bins[i]) & (
                phase_example < phase_bins[i + 1]
            )
            if np.any(mask):
                binned_amps.append(np.mean(amp_example[mask]))
            else:
                binned_amps.append(0)

        binned_amps = np.array(binned_amps)

        # Plot
        ax.plot(phase_centers, binned_amps, "b-", linewidth=2, label="Binned amplitude")
        ax.axhline(
            np.mean(amp_example),
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Mean amplitude",
        )
        ax.set_xyt("Phase (rad)", "Amplitude", "Phase-Amplitude Coupling")
        ax.set_xlim(-np.pi, np.pi)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MI values comparison
        ax = axes[idx * 3 + 2]

        methods = ["gPAC"]
        mi_values = [results["gpac"][signal_name]]
        colors = ["blue"]

        if TENSORPAC_AVAILABLE and signal_name in results["tensorpac"]:
            methods.append("TensorPAC")
            mi_values.append(results["tensorpac"][signal_name])
            colors.append("red")

        bars = ax.bar(methods, mi_values, color=colors, alpha=0.7)
        ax.set_xyt(
            None,
            "Modulation Index",
            f'MI Values (Expected strength: {signal_info["strength"]})',
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, mi_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

    # Save main figure
    spath = (
        "./scripts/example_modulation_index_comparison/modulation_index_comparison.png"
    )
    mngs.io.save(fig, spath, symlink_from_cwd=True)

    # Summary comparison figure
    fig, axes = mngs.plt.subplots(1, 2, figsize=(12, 5))
    if axes.ndim == 1:
        pass  # Already 1D
    else:
        axes = axes.flatten()

    # MI vs coupling strength
    coupling_strengths = []
    gpac_mis = []
    tensorpac_mis = []

    for signal_name, signal_info in test_signals.items():
        if "coupling" in signal_name:
            coupling_strengths.append(signal_info["strength"])
            gpac_mis.append(results["gpac"][signal_name])
            if TENSORPAC_AVAILABLE and signal_name in results["tensorpac"]:
                tensorpac_mis.append(results["tensorpac"][signal_name])

    # Sort by coupling strength
    sorted_idx = np.argsort(coupling_strengths)
    coupling_strengths = np.array(coupling_strengths)[sorted_idx]
    gpac_mis = np.array(gpac_mis)[sorted_idx]

    ax = axes[0]
    ax.plot(coupling_strengths, gpac_mis, "bo-", label="gPAC", markersize=8)
    if TENSORPAC_AVAILABLE and len(tensorpac_mis) > 0:
        tensorpac_mis = np.array(tensorpac_mis)[sorted_idx]
        ax.plot(
            coupling_strengths, tensorpac_mis, "rs--", label="TensorPAC", markersize=8
        )

    ax.set_xyt("Coupling Strength", "Modulation Index", "MI vs Coupling Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation plot (if both available)
    ax = axes[1]
    if TENSORPAC_AVAILABLE and len(tensorpac_mis) > 0:
        ax.scatter(gpac_mis, tensorpac_mis, s=100, alpha=0.7)

        # Add diagonal line
        min_val = min(np.min(gpac_mis), np.min(tensorpac_mis))
        max_val = max(np.max(gpac_mis), np.max(tensorpac_mis))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        # Compute correlation
        corr = np.corrcoef(gpac_mis, tensorpac_mis)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xyt("gPAC MI", "TensorPAC MI", "MI Correlation")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "TensorPAC not available\nfor correlation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Save summary figure
    summary_spath = (
        "./scripts/example_modulation_index_comparison/modulation_index_summary.png"
    )
    mngs.io.save(fig, summary_spath, symlink_from_cwd=True)

    # Save all results
    mngs.io.save(
        results,
        "./scripts/example_modulation_index_comparison/mi_results.pkl",
        symlink_from_cwd=True,
    )

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 60, c="green")
    profiler.print_summary()

    # Print MI comparison
    mngs.str.printc("\n📊 Modulation Index Results:", c="yellow")
    mngs.str.printc("-" * 40, c="yellow")
    for signal_name in results["gpac"].keys():
        mngs.str.printc(f"\n{test_signals[signal_name]['description']}:", c="cyan")
        mngs.str.printc(f"  gPAC MI: {results['gpac'][signal_name]:.6f}", c="cyan")
        if TENSORPAC_AVAILABLE and signal_name in results["tensorpac"]:
            mngs.str.printc(
                f"  TensorPAC MI: {results['tensorpac'][signal_name]:.6f}", c="cyan"
            )
            diff = abs(results["gpac"][signal_name] - results["tensorpac"][signal_name])
            mngs.str.printc(f"  Difference: {diff:.6f}", c="cyan")

    mngs.str.printc(f"\n✅ Comparison complete! Results saved to:", c="green")
    mngs.str.printc(f"   - {spath}", c="green")
    mngs.str.printc(f"   - {summary_spath}", c="green")

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
        description="Compare modulation index between gPAC and TensorPAC"
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
