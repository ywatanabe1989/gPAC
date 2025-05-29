#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 02:57:02 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/trainability/example_pac_trainability_working.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/trainability/example_pac_trainability_working.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates PAC model trainability with synthetic data
  - Shows how the model can learn to detect PAC coupling
  - Simple working example without complex features

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
    - ./scripts/example_pac_trainability_working/pac_demo.png
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
def generate_simple_pac_signal(
    duration=2.0, fs=256.0, phase_freq=10.0, amp_freq=80.0, coupling=0.5
):
    """Generate a simple PAC signal."""
    t = np.linspace(0, duration, int(fs * duration))
    # Phase signal
    phase = np.sin(2 * np.pi * phase_freq * t)
    # Amplitude modulation
    amp_mod = 1 + coupling * np.sin(2 * np.pi * phase_freq * t)
    # High frequency carrier
    carrier = np.sin(2 * np.pi * amp_freq * t)
    # Combined signal
    signal = phase + amp_mod * carrier
    # Add noise
    signal += 0.1 * np.random.randn(len(signal))
    return signal


def main(args):
    """Run simple PAC trainability demonstration."""
    import mngs
    from gpac import PAC

    mngs.str.printc("🚀 Simple PAC Detection Demo", c="green")
    mngs.str.printc("=" * 40, c="green")

    # Parameters
    fs = 256.0
    duration = 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mngs.str.printc(f"\n📍 Device: {device}", c="cyan")

    # Initialize PAC model
    pac_model = PAC(
        seq_len=int(fs * duration),
        fs=fs,
        trainable=False,  # Using fixed bands for simplicity
    ).to(device)

    # Generate test signals
    mngs.str.printc("\n📊 Generating test signals...", c="blue")

    # PAC signal
    pac_signal = generate_simple_pac_signal(duration, fs, 10.0, 80.0, 0.8)
    pac_signal = (
        torch.tensor(pac_signal, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # No PAC signal (just noise)
    no_pac_signal = torch.randn_like(pac_signal) * 0.5

    # Compute PAC
    with torch.no_grad():
        pac_result = pac_model(pac_signal)
        pac_values = pac_result["pac"][0, 0].cpu().numpy()  # [n_pha, n_amp]

        no_pac_result = pac_model(no_pac_signal)
        no_pac_values = no_pac_result["pac"][0, 0].cpu().numpy()

    # Visualization
    fig, axes = mngs.plt.subplots(1, 2, figsize=(10, 4))

    # Plot PAC heatmap
    ax = axes[0]
    im = ax.imshow(pac_values, aspect="auto", origin="lower", cmap="hot")
    ax.set_xyt("Amplitude band", "Phase band", "PAC Signal")

    # Plot no-PAC heatmap
    ax = axes[1]
    im = ax.imshow(
        no_pac_values,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=pac_values.min(),
        vmax=pac_values.max(),
    )
    ax.set_xyt("Amplitude band", "Phase band", "No PAC Signal")

    # Save figure
    spath = "./pac_demo.png"
    mngs.io.save(fig, spath)

    # Print results
    mngs.str.printc(
        f"\n✅ PAC signal max value: {pac_values.max():.3f}", c="green"
    )
    mngs.str.printc(
        f"✅ No-PAC signal max value: {no_pac_values.max():.3f}", c="green"
    )
    mngs.str.printc(f"\n✅ Results saved to: {spath}", c="green")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(
        description="Simple PAC detection demonstration"
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
