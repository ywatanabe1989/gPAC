#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 01:00:00 (ywatanabe)"
# File: ./examples/trainability/example_pac_trainability_simple.py

"""
Functionalities:
  - Demonstrates how SincNet-style filters can learn to detect specific frequencies
  - Shows basic gradient-based optimization of filter parameters
  - Visualizes filter adaptation during training
  - Saves learned filter responses and training progress

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
    - ./scripts/example_pac_trainability_simple/pac_trainability_simple_demo.png
    - ./scripts/example_pac_trainability_simple/pac_trainability_simple_demo.csv
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def main(args):
    """Run simple filter trainability demonstration."""
    import mngs
    from gpac._Filters._DifferentiableBandpassFilter import DifferentiableBandPassFilter
    from gpac._Profiler import create_profiler

    mngs.str.printc("🚀 Simple SincNet Filter Trainability Demo", c="green")
    mngs.str.printc("=" * 60, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Parameters
    fs = 256.0
    sig_len = 512
    n_epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Target signal with known frequency
    target_freq = 10.0  # Hz (we want the filter to learn this)
    t = torch.arange(sig_len, device=device) / fs
    target_signal = torch.sin(2 * np.pi * target_freq * t)
    noise = 0.1 * torch.randn_like(target_signal)
    noisy_signal = target_signal + noise

    mngs.str.printc(f"\n📍 Device: {device}", c="cyan")
    mngs.str.printc(f"🎯 Target frequency to learn: {target_freq} Hz", c="yellow")

    # Initialize filter with wrong bands
    with profiler.profile("Filter Initialization"):
        filter_model = DifferentiableBandPassFilter(
            sig_len=sig_len,
            fs=fs,
            pha_low_hz=15,  # Start at wrong frequency
            pha_high_hz=25,
            pha_n_bands=1,
            amp_low_hz=50,
            amp_high_hz=100,
            amp_n_bands=1,
            filter_length=101,
            normalization="std",
        ).to(device)

    # Get initial band
    initial_bands = filter_model.get_filter_banks()
    mngs.str.printc(
        f"\n📊 Initial band: {initial_bands['pha_bands'][0].cpu().numpy()} Hz", c="cyan"
    )

    # Optimizer
    optimizer = optim.Adam(filter_model.parameters(), lr=0.1)

    # Storage for visualization
    losses = []
    band_history = []

    # Training loop
    mngs.str.printc("\n🔄 Training filter to detect target frequency...", c="blue")

    with profiler.profile("Training Loop"):
        for epoch in range(n_epochs):
            # Forward pass - filter the noisy signal
            filtered = filter_model(noisy_signal.unsqueeze(0).unsqueeze(0))
            # Shape: [1, 1, 2, sig_len] (2 bands: phase and amplitude)

            # Extract phase band result
            filtered_signal = filtered[0, 0, 0]  # Get the phase band filtered signal

            # Loss: maximize correlation with clean target signal
            correlation = torch.sum(filtered_signal * target_signal)
            energy = torch.sqrt(
                torch.sum(filtered_signal**2) * torch.sum(target_signal**2)
            )
            normalized_correlation = correlation / (energy + 1e-8)

            # We want to maximize correlation, so minimize negative correlation
            loss = -normalized_correlation

            # Add regularization
            reg_losses = filter_model.get_regularization_loss(0.01, 0.01)
            total_loss = loss + reg_losses["total"]

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Constrain parameters
            filter_model.constrain_parameters()

            # Record for visualization
            losses.append(loss.item())

            # Record band every 5 epochs
            if epoch % 5 == 0:
                bands = filter_model.get_filter_banks()
                band_history.append(bands["pha_bands"][0].cpu().numpy())

                if epoch % 10 == 0:
                    current_band = bands["pha_bands"][0].cpu().numpy()
                    mngs.str.printc(
                        f"  Epoch {epoch:3d}: Band = [{current_band[0]:.1f}, {current_band[1]:.1f}] Hz, "
                        f"Loss = {loss.item():.4f}",
                        c="cyan",
                    )

    # Get final band
    final_bands = filter_model.get_filter_banks()
    final_band = final_bands["pha_bands"][0].cpu().numpy()
    mngs.str.printc(
        f"\n✅ Final learned band: [{final_band[0]:.1f}, {final_band[1]:.1f}] Hz",
        c="green",
    )
    mngs.str.printc(
        f"   Target {target_freq} Hz is {'✓ inside' if final_band[0] <= target_freq <= final_band[1] else '✗ outside'} the learned band",
        c="green",
    )

    # Visualization
    with profiler.profile("Visualization"):
        fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))
        if axes.ndim == 1:
            axes = axes.reshape(2, 2)

        # Plot loss curve
        ax = axes[0, 0]
        ax.plot(losses)
        ax.set_xyt("Epoch", "Loss (negative correlation)", "Training Loss")
        ax.grid(True, alpha=0.3)

        # Plot band evolution
        ax = axes[0, 1]
        band_history = np.array(band_history)
        epochs_recorded = np.arange(0, n_epochs, 5)
        ax.fill_between(
            epochs_recorded,
            band_history[:, 0],
            band_history[:, 1],
            alpha=0.3,
            label="Filter band",
        )
        ax.plot(epochs_recorded, band_history[:, 0], "b--", label="Low cutoff")
        ax.plot(epochs_recorded, band_history[:, 1], "b-", label="High cutoff")
        ax.axhline(
            target_freq,
            color="r",
            linestyle=":",
            linewidth=2,
            label=f"Target freq ({target_freq} Hz)",
        )
        ax.set_xyt("Epoch", "Frequency (Hz)", "Filter Band Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot signals
        ax = axes[1, 0]
        time_plot = t[:256].cpu().numpy()
        ax.plot(
            time_plot,
            noisy_signal[:256].cpu().numpy(),
            "gray",
            alpha=0.5,
            label="Noisy input",
        )
        ax.plot(
            time_plot,
            target_signal[:256].cpu().numpy(),
            "r",
            alpha=0.8,
            label="Clean target",
        )

        # Filter final signal
        with torch.no_grad():
            final_filtered = filter_model(noisy_signal.unsqueeze(0).unsqueeze(0))
            final_filtered_signal = final_filtered[0, 0, 0]

        ax.plot(
            time_plot,
            final_filtered_signal[:256].cpu().numpy(),
            "b",
            label="Filtered output",
        )
        ax.set_xyt("Time (s)", "Amplitude", "Signal Filtering Result")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot frequency response
        ax = axes[1, 1]
        with torch.no_grad():
            # Get the actual filter coefficients
            filters = filter_model._compute_filters()
            phase_filter = filters[0].cpu().numpy()  # First filter is the phase filter

            # Compute frequency response
            from scipy import signal as sp_signal

            freqs = np.fft.rfftfreq(len(phase_filter), 1 / fs)
            freq_response = np.abs(np.fft.rfft(phase_filter))

            ax.plot(freqs, freq_response)
            ax.axvline(
                target_freq, color="r", linestyle=":", linewidth=2, label=f"Target freq"
            )
            ax.axvspan(
                final_band[0], final_band[1], alpha=0.2, color="b", label="Learned band"
            )
            ax.set_xyt("Frequency (Hz)", "Magnitude", "Final Filter Frequency Response")
            ax.set_xlim(0, 50)
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Save figure
    spath = "./scripts/example_pac_trainability_simple/pac_trainability_simple_demo.png"
    mngs.io.save(fig, spath, symlink_from_cwd=True)

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 60, c="green")
    profiler.print_summary()

    mngs.str.printc(
        f"\n✅ Simple trainability demo complete! Results saved to: {spath}", c="green"
    )

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Simple demonstration of trainable SincNet filters"
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
