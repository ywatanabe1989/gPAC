#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 18:05:00 (ywatanabe)"
# File: example_Hilbert.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_hilbert.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates Hilbert transform for extracting instantaneous phase and amplitude
  - Shows phase-amplitude coupling (PAC) analysis workflow
  - Visualizes phase-amplitude relationships
  - Demonstrates batch processing and GPU acceleration

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, mngs

IO:
  - input-files: None (generates synthetic signal with theta-gamma PAC)
  - output-files: ./examples/gpac/example_Hilbert_out/figures/hilbert_transform_example.png
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def generate_pac_signal(fs, duration, theta_freq, gamma_freq):
    """Generate synthetic signal with phase-amplitude coupling."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)

    # Theta phase signal
    theta_phase = 2 * np.pi * theta_freq * t

    # Gamma amplitude modulated by theta phase
    gamma_amplitude = 1 + 0.5 * np.cos(theta_phase)  # PAC modulation
    gamma_signal = gamma_amplitude * np.sin(2 * np.pi * gamma_freq * t)

    # Composite signal
    signal = np.sin(theta_phase) + gamma_signal + 0.1 * np.random.randn(n_samples)

    return signal, t, n_samples


def compute_pac_coupling(phase_values, amplitude_values, n_bins=18):
    """Compute mean amplitude per phase bin."""
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp_per_bin = []

    for i in range(n_bins):
        mask = (phase_values >= phase_bins[i]) & (phase_values < phase_bins[i + 1])
        if mask.sum() > 0:
            mean_amp_per_bin.append(amplitude_values[mask].mean())
        else:
            mean_amp_per_bin.append(0)

    return np.array(mean_amp_per_bin), phase_bins


def main(args):
    """Main function to demonstrate Hilbert transform usage."""
    import mngs
    from gpac import Hilbert, BandPassFilter
    
    # Generate synthetic signal with phase-amplitude coupling
    signal, t, n_samples = generate_pac_signal(
        args.fs, args.duration, args.theta_freq, args.gamma_freq
    )

    mngs.str.printc("Hilbert Transform Example", c='yellow')
    print("=" * 50)
    print(f"Signal duration: {args.duration} seconds")
    print(f"Sampling frequency: {args.fs} Hz")

    # Convert to torch tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Step 1: Extract theta band for phase
    print("\n1. Extracting theta phase (4-8 Hz)...")
    theta_filter = BandPassFilter(
        fs=args.fs,
        seq_len=n_samples,
        f_lower=4.0,
        f_upper=8.0,
        n_taps=args.n_taps,
    )
    theta_filtered = theta_filter(signal_tensor)

    # Apply Hilbert transform to get phase
    hilbert = Hilbert()
    theta_complex = hilbert(theta_filtered)
    theta_phase_extracted = torch.angle(theta_complex)
    theta_amplitude = torch.abs(theta_complex)

    print(f"Theta phase shape: {theta_phase_extracted.shape}")
    print(
        f"Phase range: [{theta_phase_extracted.min():.2f}, "
        f"{theta_phase_extracted.max():.2f}] radians"
    )

    # Step 2: Extract gamma band for amplitude
    print("\n2. Extracting gamma amplitude (30-50 Hz)...")
    gamma_filter = BandPassFilter(
        fs=args.fs,
        seq_len=n_samples,
        f_lower=30.0,
        f_upper=50.0,
        n_taps=args.n_taps,
    )
    gamma_filtered = gamma_filter(signal_tensor)

    # Apply Hilbert transform to get amplitude
    gamma_complex = hilbert(gamma_filtered)
    gamma_amplitude_extracted = torch.abs(gamma_complex)

    print(f"Gamma amplitude shape: {gamma_amplitude_extracted.shape}")
    print(
        f"Amplitude range: [{gamma_amplitude_extracted.min():.2f}, "
        f"{gamma_amplitude_extracted.max():.2f}]"
    )

    # Step 3: Analyze phase-amplitude relationship
    print("\n3. Computing phase-amplitude coupling...")

    phase_values = theta_phase_extracted.squeeze().numpy()
    amplitude_values = gamma_amplitude_extracted.squeeze().numpy()

    # Compute mean amplitude per phase bin
    mean_amp_per_bin, phase_bins = compute_pac_coupling(
        phase_values, amplitude_values, n_bins=18
    )

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Original signal
    time_window = (t >= 1) & (t <= 2)  # 1-second window
    axes[0, 0].plot(t[time_window], signal[time_window], "k-", alpha=0.7)
    axes[0, 0].set_title("Original Signal (1s window)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlabel("Time (s)")

    # Theta band signal and phase
    axes[0, 1].plot(
        t[time_window], theta_filtered.squeeze()[time_window], "b-", label="Filtered"
    )
    ax_phase = axes[0, 1].twinx()
    ax_phase.plot(
        t[time_window],
        theta_phase_extracted.squeeze()[time_window],
        "r--",
        alpha=0.7,
        label="Phase",
    )
    axes[0, 1].set_title("Theta Band")
    axes[0, 1].set_ylabel("Amplitude", color="b")
    ax_phase.set_ylabel("Phase (rad)", color="r")
    axes[0, 1].set_xlabel("Time (s)")

    # Gamma band signal and amplitude envelope
    axes[1, 0].plot(
        t[time_window],
        gamma_filtered.squeeze()[time_window],
        "g-",
        alpha=0.5,
        label="Filtered",
    )
    axes[1, 0].plot(
        t[time_window],
        gamma_amplitude_extracted.squeeze()[time_window],
        "r-",
        linewidth=2,
        label="Amplitude",
    )
    axes[1, 0].set_title("Gamma Band with Amplitude Envelope")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()

    # Phase-amplitude coupling plot
    n_bins = len(mean_amp_per_bin)
    axes[1, 1].bar(range(n_bins), mean_amp_per_bin, color="purple", alpha=0.7)
    axes[1, 1].set_title("Phase-Amplitude Coupling")
    axes[1, 1].set_xlabel("Phase Bin")
    axes[1, 1].set_ylabel("Mean Gamma Amplitude")
    axes[1, 1].set_xticks([0, n_bins // 2, n_bins])
    axes[1, 1].set_xticklabels(["-π", "0", "π"])

    # Scatter plot of phase vs amplitude
    axes[2, 0].scatter(phase_values[::10], amplitude_values[::10], alpha=0.3, s=1)
    axes[2, 0].set_title("Phase vs Amplitude Scatter")
    axes[2, 0].set_xlabel("Theta Phase (rad)")
    axes[2, 0].set_ylabel("Gamma Amplitude")
    axes[2, 0].set_xlim(-np.pi, np.pi)

    # Polar plot
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    axes[2, 1] = plt.subplot(3, 2, 6, projection="polar")
    axes[2, 1].plot(
        phase_centers, mean_amp_per_bin, "o-", color="purple", linewidth=2, markersize=8
    )
    axes[2, 1].set_title("PAC Polar Plot")
    axes[2, 1].set_theta_zero_location("N")
    axes[2, 1].set_theta_direction(-1)

    plt.tight_layout()
    
    # Save using mngs
    save_path = CONFIG.PATH.FIGURES / "hilbert_transform_example.png"
    mngs.io.save(fig, save_path)
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    # Example with batch processing
    print("\n4. Batch processing example...")
    batch_size = 16
    n_channels = 4
    batch_signal = torch.randn(batch_size, n_channels, n_samples)

    # Filter and apply Hilbert transform
    batch_filtered = theta_filter(batch_signal)
    batch_complex = hilbert(batch_filtered)
    batch_phase = torch.angle(batch_complex)
    batch_amplitude = torch.abs(batch_complex)

    print(f"Batch input shape: {batch_signal.shape}")
    print(f"Batch phase shape: {batch_phase.shape}")
    print(f"Batch amplitude shape: {batch_amplitude.shape}")

    # GPU acceleration example
    if args.use_gpu and torch.cuda.is_available():
        print("\n5. GPU-accelerated Hilbert transform...")
        device = torch.device("cuda")

        # Move to GPU
        hilbert_gpu = hilbert.to(device)
        signal_gpu = batch_filtered.to(device)

        # Compute on GPU
        complex_gpu = hilbert_gpu(signal_gpu)
        phase_gpu = torch.angle(complex_gpu)
        amplitude_gpu = torch.abs(complex_gpu)

        print(f"GPU computation successful!")
        print(f"Output device: {phase_gpu.device}")

    print("\n6. Summary:")
    print("- Hilbert transform extracts instantaneous phase and amplitude")
    print(
        "- Essential for PAC analysis: phase from low frequencies, amplitude from high"
    )
    print("- Supports batch processing for multiple channels/trials")
    print("- GPU acceleration available for large-scale analysis")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='Hilbert transform example for gPAC')
    parser.add_argument(
        "--fs", type=float, default=250.0, help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Signal duration (seconds)"
    )
    parser.add_argument(
        "--theta_freq", type=float, default=6.0, help="Theta frequency (Hz)"
    )
    parser.add_argument(
        "--gamma_freq", type=float, default=40.0, help="Gamma frequency (Hz)"
    )
    parser.add_argument("--n_taps", type=int, default=101, help="Number of filter taps")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU acceleration if available"
    )
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == '__main__':
    run_main()

# EOF