#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 00:45:36 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/gpac/core/example__BandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/core/example__BandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates BandPassFilter functionality for PAC analysis
  - Shows static and trainable bandpass filtering
  - Visualizes filtered phase and amplitude bands
  - Compares filter characteristics and frequency responses

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, mngs

IO:
  - input-files: None (generates synthetic PAC signals)
  - output-files:
    - 01_static_bandpass_filter.gif
    - 02_trainable_bandpass_filter.gif
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import gpac
import matplotlib.pyplot as plt
import torch

"""Functions & Classes"""
def demo_bandpass_filter(trainable):
    """Demonstrate static BandPassFilter with visualization."""
    import mngs

    trainable_str = {True: "trainable", False: "static"}[trainable]

    mngs.str.printc(
        f"=== Demo {trainable_str.capitalize()} BandPassFilter ===", c="yellow"
    )

    pac_config = gpac.dataset.multi_class_multi_pac_config
    # Generate synthetic PAC data
    batch = gpac.dataset.generate_pac_batch(
        batch_size=4,
        n_channels=19,
        n_segments=8,
        duration_sec=2,
        fs=512,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    # Create filter
    bpf = gpac.core.BandPassFilter(
        fs=metadata["fs"][0],
        pha_range_hz=(4.0, 12.0),  # Theta-alpha range
        amp_range_hz=(30.0, 100.0),  # Gamma range
        pha_n_bands=4,
        amp_n_bands=8,
        trainable=trainable,
    )

    if torch.cuda.is_available():
        bpf = bpf.cuda()
        signal = signal.cuda()

    # Apply filter
    filtered = bpf(signal)
    # (4, 19, 8, 12, 1024)
    print(filtered.shape)

    # Visualize results
    fig, axes = mngs.plt.subplots(3, 1, figsize=(12, 10))

    i_sample, i_channel, i_segment = 0, 0, 0

    # Original signal
    ax_raw = axes[0]
    ax_raw.plot(
        signal[i_sample, i_channel, i_segment].cpu().numpy(), "b-", alpha=0.7
    )
    ax_raw.set_xyt("Time [samples]", "Amplitude", "Original Signal")
    ax_raw.grid(True, alpha=0.3)

    # Phase bands (first channels)
    ax_pha = axes[1]
    for i_pha_band in range(bpf.info["pha_n_bands"]):
        ax_pha.plot(
            filtered[i_sample, i_channel, i_segment, i_pha_band]
            .detach()
            .cpu()
            .numpy(),
            label=f"Band {i_pha_band+1}",
            alpha=0.7,
        )
    ax_pha.set_xyt("Time [samples]", "Amplitude", "Phase Bands (4-12 Hz)")
    ax_pha.legend(loc="upper right")
    ax_pha.grid(True, alpha=0.3)

    # Amplitude bands (show subset)
    ax_amp = axes[2]
    for i_amp_band in range(
        bpf.info["pha_n_bands"],
        bpf.info["pha_n_bands"] + bpf.info["amp_n_bands"] - 1,
    ):
        ax_amp.plot(
            filtered[i_sample, i_channel, i_segment, i_amp_band]
            .detach()
            .cpu()
            .numpy(),
            label=f"Band {i_amp_band+1}",
            alpha=0.7,
        )
    ax_amp.set_xyt(
        "Time [samples]",
        "Amplitude",
        "Amplitude Bands (30-100 Hz, first 4 shown)",
    )
    ax_amp.legend(loc="upper right")
    ax_amp.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    i_trainable = {True: "01", False: "02"}[trainable]
    mngs.io.save(fig, f"{i_trainable}_{trainable_str}_bandpass_filter.gif")
    plt.close()

    print(f"Filter info: {bpf.info}")
    print(f"Input shape: {signal.shape}")
    print(f"Output shape: {filtered.shape}")


def main(args):
    """Main function to demonstrate BandPassFilter functionality."""
    demo_bandpass_filter(trainable=False)
    demo_bandpass_filter(trainable=True)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(
        description="Demonstrate BandPassFilter functionality"
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
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
