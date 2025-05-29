#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:30:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_mngs_comparison.py

"""
Functionalities:
  - Compares PAC computation between gPAC and MNGS package
  - Validates output shapes and correlation
  - Visualizes PAC matrices from both implementations
  - Saves comparison results and figures

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - mngs
    - numpy
    - matplotlib
    
IO:
  - input-files:
    - None (uses demo signals from mngs)
    
  - output-files:
    - ./scripts/example_mngs_comparison/mngs_comparison.png
    - ./scripts/example_mngs_comparison/mngs_comparison.csv
"""

"""Imports"""
import argparse

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def main(args):
    import gpac
    import numpy as np
    import mngs
    from gpac._Profiler import create_profiler

    mngs.str.printc("🚀 PAC Comparison: gPAC vs MNGS", c="green")
    mngs.str.printc("=" * 50, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Generate demo PAC signal
    mngs.str.printc("\n📡 Generating demo PAC signal...", c="cyan")
    xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")

    # Parameters
    pha_n_bands = 20
    amp_n_bands = 15
    n_epochs = 2
    n_channels = 2

    # Use subset of data
    test_signal = xx[:n_epochs, :n_channels]

    mngs.str.printc(f"Signal shape: {test_signal.shape}", c="cyan")
    mngs.str.printc(f"Sampling rate: {fs} Hz", c="cyan")
    mngs.str.printc(
        f"Phase bands: {pha_n_bands}, Amplitude bands: {amp_n_bands}", c="cyan"
    )

    # MNGS PAC computation
    mngs.str.printc("\n🔧 Computing PAC with MNGS...", c="blue")
    with profiler.profile("MNGS PAC"):
        pac_mngs, pha_mids_hz, amp_mids_hz = mngs.dsp.pac(
            test_signal, fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
        )

    # gPAC computation
    mngs.str.printc("\n🔧 Computing PAC with gPAC...", c="red")
    with profiler.profile("gPAC PAC"):
        pac_gpac, pha_mids_hz_g, amp_mids_hz_g = gpac.calculate_pac(
            test_signal, fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
        )

    # Verify shapes
    mngs.str.printc(f"\n✅ MNGS PAC shape: {pac_mngs.shape}", c="green")
    mngs.str.printc(f"✅ gPAC PAC shape: {pac_gpac.shape}", c="green")
    assert pac_mngs.shape == pac_gpac.shape, "Shape mismatch!"

    # Compute correlation
    mngs.str.printc("\n📊 Computing correlation between results...", c="yellow")
    correlations = []
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            corr = np.corrcoef(
                pac_mngs[epoch, channel].flatten(), pac_gpac[epoch, channel].flatten()
            )[0, 1]
            correlations.append(corr)
            mngs.str.printc(
                f"  Epoch {epoch}, Channel {channel}: r = {corr:.4f}", c="cyan"
            )

    mean_corr = np.mean(correlations)
    mngs.str.printc(f"\n📈 Mean correlation: {mean_corr:.4f}", c="yellow")

    # Visualization
    fig, axes = mngs.plt.subplots(n_epochs, 3, figsize=(15, 5 * n_epochs))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for epoch in range(n_epochs):
        # MNGS result
        ax = axes[epoch, 0]
        im = ax.imshow(pac_mngs[epoch, 0], aspect="auto", origin="lower", cmap="hot")
        ax.set_xyt("Amplitude bands", "Phase bands", f"MNGS PAC - Epoch {epoch}")
        plt.colorbar(im, ax=ax)

        # gPAC result
        ax = axes[epoch, 1]
        im = ax.imshow(pac_gpac[epoch, 0], aspect="auto", origin="lower", cmap="hot")
        ax.set_xyt("Amplitude bands", "Phase bands", f"gPAC PAC - Epoch {epoch}")
        plt.colorbar(im, ax=ax)

        # Difference
        ax = axes[epoch, 2]
        diff = pac_mngs[epoch, 0] - pac_gpac[epoch, 0]
        im = ax.imshow(
            diff,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-np.abs(diff).max(),
            vmax=np.abs(diff).max(),
        )
        ax.set_xyt("Amplitude bands", "Phase bands", f"Difference - Epoch {epoch}")
        plt.colorbar(im, ax=ax)

    # Save figure
    spath = "./scripts/example_mngs_comparison/mngs_comparison.png"
    mngs.io.save(fig, spath, symlink_from_cwd=True)

    # Save comparison data
    comparison_data = {
        "mean_correlation": mean_corr,
        "correlations": correlations,
        "pha_mids_hz_mngs": pha_mids_hz,
        "pha_mids_hz_gpac": pha_mids_hz_g,
        "amp_mids_hz_mngs": amp_mids_hz,
        "amp_mids_hz_gpac": amp_mids_hz_g,
    }
    mngs.io.save(
        comparison_data,
        "./scripts/example_mngs_comparison/mngs_comparison.csv",
        symlink_from_cwd=True,
    )

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 50, c="green")
    profiler.print_summary()

    mngs.str.printc(f"\n✅ Comparison complete! Results saved to: {spath}", c="green")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Compare PAC computation between gPAC and MNGS"
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
