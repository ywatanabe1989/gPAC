#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:45:00 (ywatanabe)"
# File: ./examples/gpac/example_PAC.py

"""
Functionalities:
  - Demonstrates PAC calculation with gPAC
  - Generates synthetic signals with known PAC
  - Calculates PAC with and without statistical testing
  - Visualizes PAC matrices and z-scores
  - Saves results to file

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
    - ./scripts/example_PAC/pac_analysis_results.gif
    - ./scripts/example_PAC/pac_results.npz
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def generate_pac_signal(fs=1000, duration=5):
    """Generate synthetic signal with known PAC."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Phase component (theta, 6 Hz)
    phase_freq = 6  # Hz
    phase_signal = np.sin(2 * np.pi * phase_freq * t)

    # Amplitude component (gamma, 80 Hz) modulated by phase
    amp_freq = 80  # Hz
    modulation_depth = 0.7  # How strongly phase modulates amplitude (0-1)

    # Create amplitude modulation envelope
    amp_envelope = 1 + modulation_depth * np.sin(2 * np.pi * phase_freq * t)

    # Generate modulated high-frequency signal
    amp_signal = amp_envelope * np.sin(2 * np.pi * amp_freq * t)

    # Combine signals with some noise
    noise_level = 0.2
    noise = noise_level * np.random.randn(n_samples)
    signal = phase_signal + 0.3 * amp_signal + noise

    return signal, t, phase_freq, amp_freq, phase_signal, amp_signal, amp_envelope


def main(args):
    """Run PAC analysis demonstration."""
    import mngs
    from gpac import calculate_pac
    from gpac._Profiler import create_profiler

    mngs.str.printc("🚀 PAC Analysis Example", c="green")
    mngs.str.printc("=" * 50, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    duration = 5  # Signal duration (seconds)

    # 1. Generate synthetic signal with PAC
    mngs.str.printc("\n📡 Generating synthetic signal with PAC...", c="cyan")
    with profiler.profile("Signal Generation"):
        (
            signal,
            t,
            phase_freq,
            amp_freq,
            phase_signal,
            amp_signal,
            amp_envelope,
        ) = generate_pac_signal(fs, duration)

        # Prepare signal for gPAC (4D tensor: batch, channels, segments, time)
        signal_tensor = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

    mngs.str.printc(f"Signal shape: {signal_tensor.shape}", c="cyan")
    mngs.str.printc(f"Signal duration: {duration} seconds", c="cyan")
    mngs.str.printc(f"Sampling rate: {fs} Hz", c="cyan")
    mngs.str.printc(f"True coupling: θ={phase_freq} Hz → γ={amp_freq} Hz", c="yellow")

    # 2. Calculate PAC without statistical testing
    mngs.str.printc("\n🔄 Calculating PAC...", c="blue")
    with profiler.profile("PAC Calculation (No Permutation)"):
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal_tensor,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=20,
            amp_start_hz=60,
            amp_end_hz=120,
            amp_n_bands=20,
            n_perm=None,  # No permutation testing
        )

    # Convert to numpy for plotting
    pac_matrix = pac_values[0, 0].cpu().numpy()

    # 3. Calculate PAC with statistical testing
    mngs.str.printc("\n🔄 Calculating PAC with permutation testing...", c="blue")
    with profiler.profile("PAC Calculation (With Permutation)"):
        pac_zscore, _, _ = calculate_pac(
            signal_tensor,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=20,
            amp_start_hz=60,
            amp_end_hz=120,
            amp_n_bands=20,
            n_perm=200,  # 200 permutations for z-score calculation
        )

    pac_z_matrix = pac_zscore[0, 0].cpu().numpy()

    # 4. Visualize results
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot original signal (first second)
    ax = axes[0]
    plot_duration = 1  # seconds
    plot_samples = int(fs * plot_duration)
    ax.plot(t[:plot_samples], signal[:plot_samples], "b-", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original Signal (first second)")
    ax.set_xlim(0, plot_duration)

    # Plot phase and amplitude components
    ax = axes[1]
    ax.plot(
        t[:plot_samples],
        phase_signal[:plot_samples],
        "g-",
        label=f"Phase ({phase_freq} Hz)",
    )
    ax.plot(
        t[:plot_samples],
        0.3 * amp_signal[:plot_samples],
        "r-",
        alpha=0.7,
        label=f"Amplitude ({amp_freq} Hz)",
    )
    ax.plot(
        t[:plot_samples],
        0.3 * amp_envelope[:plot_samples],
        "k--",
        label="Modulation envelope",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal Components")
    ax.set_xlim(0, plot_duration)
    ax.legend()

    # Plot PAC matrix
    ax = axes[2]
    im = ax.imshow(
        pac_matrix,
        aspect="auto",
        origin="lower",
        extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
        cmap="hot",
    )
    ax.set_xlabel("Amplitude Frequency (Hz)")
    ax.set_ylabel("Phase Frequency (Hz)")
    ax.set_title("Phase-Amplitude Coupling")
    plt.colorbar(im, ax=ax, label="PAC Strength")

    # Add markers for true coupling
    ax.plot(
        amp_freq,
        phase_freq,
        "wo",
        markersize=10,
        markeredgecolor="w",
        markeredgewidth=2,
    )
    ax.text(amp_freq + 2, phase_freq + 0.5, "True\ncoupling", color="white", fontsize=9)

    # Plot z-scored PAC
    ax = axes[3]
    im = ax.imshow(
        pac_z_matrix,
        aspect="auto",
        origin="lower",
        extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
        cmap="hot",
        vmin=0,
        vmax=5,
    )
    ax.set_xlabel("Amplitude Frequency (Hz)")
    ax.set_ylabel("Phase Frequency (Hz)")
    ax.set_title("PAC Z-scores (n_perm=200)")
    plt.colorbar(im, ax=ax, label="Z-score")

    # Mark significant regions (z > 2 ~ p < 0.05)
    significant = pac_z_matrix > 2
    ax.contour(
        amp_freqs,
        pha_freqs,
        significant,
        levels=[0.5],
        colors="cyan",
        linewidths=2,
    )
    ax.plot(
        amp_freq,
        phase_freq,
        "wo",
        markersize=10,
        markeredgecolor="w",
        markeredgewidth=2,
    )

    # Save figure
    spath = "pac_analysis_results.gif"
    mngs.io.save(fig, spath)

    # 5. Report findings
    mngs.str.printc("\n📊 PAC Analysis Results", c="yellow")
    mngs.str.printc("=" * 50, c="yellow")
    mngs.str.printc(f"Maximum PAC value: {pac_matrix.max():.4f}", c="cyan")
    max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
    mngs.str.printc(
        f"Peak coupling at: Phase {pha_freqs[max_idx[0]]:.1f} Hz, Amplitude {amp_freqs[max_idx[1]]:.1f} Hz",
        c="cyan",
    )
    mngs.str.printc(
        f"Expected coupling: Phase {phase_freq} Hz, Amplitude {amp_freq} Hz", c="cyan"
    )

    mngs.str.printc(f"\nMaximum z-score: {pac_z_matrix.max():.2f}", c="cyan")
    significant_pairs = np.sum(pac_z_matrix > 2)
    total_pairs = pac_z_matrix.size
    mngs.str.printc(
        f"Significant frequency pairs (z > 2): {significant_pairs}/{total_pairs} ({100*significant_pairs/total_pairs:.1f}%)",
        c="cyan",
    )

    # 6. Save results
    mngs.str.printc("\n💾 Saving results...", c="blue")
    results_data = {
        "pac_values": pac_matrix,
        "pac_zscores": pac_z_matrix,
        "pha_freqs": pha_freqs,
        "amp_freqs": amp_freqs,
        "signal": signal,
        "fs": fs,
    }
    mngs.io.save(
        results_data, "pac_results.npz"
    )

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 50, c="green")
    profiler.print_summary()

    mngs.str.printc("\n✅ PAC analysis completed successfully!", c="green")
    mngs.str.printc(f"💾 Results saved to: {spath}", c="green")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="PAC analysis example")
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
