#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-05 14:27:56 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/performance/multiple_gpus/comodulogram.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/multiple_gpus/comodulogram.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Plots PAC comodulogram for the same sample using single and multi-GPU
  - Demonstrates computational consistency between configurations
  - Shows that multi-GPU produces identical results to single-GPU
  - Saves visualization comparing the two approaches

Dependencies:
  - scripts:
    - ./utils.py
  - packages:
    - torch, numpy, matplotlib, scitex, gpac
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - comodulogram_comparison.gif
    - comodulogram_results.yaml
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scitex
import numpy as np
import torch
from utils import (create_pac_model, create_test_config, generate_test_data,
                   print_gpu_info, print_test_header)

"""Functions & Classes"""
def compute_pac_comodulogram(pac_model, data):
    """
    Compute PAC values and return as comodulogram matrix.

    Args:
        pac_model: PAC model instance
        data: Input data tensor

    Returns:
        pac_values: Comodulogram matrix (n_pha x n_amp)
    """
    # Ensure model and data are on the correct device
    if torch.cuda.is_available():
        if hasattr(pac_model, "cuda"):
            pac_model = pac_model.cuda()
        data = data.cuda()

    # Compute PAC
    result = pac_model(data)

    # Extract PAC values from result
    if isinstance(result, dict):
        pac_values = result.get("pac", result.get("pac_zscore", None))
    elif isinstance(result, tuple):
        pac_values = result[0]
    else:
        pac_values = result

    # Average across batch and channels if needed
    if pac_values.dim() > 2:
        pac_values = pac_values.mean(dim=(0, 1))

    return pac_values.cpu().numpy()


def plot_comodulogram_comparison(pac_single, pac_multi, pha_freqs, amp_freqs):
    """
    Create side-by-side comodulogram plots for single and multi-GPU results.

    Args:
        pac_single: PAC values from single GPU
        pac_multi: PAC values from multi-GPU
        pha_freqs: Phase frequency centers
        amp_freqs: Amplitude frequency centers

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = scitex.plt.subplots(1, 3, figsize=(18, 5))

    # Common colormap parameters
    vmin = min(pac_single.min(), pac_multi.min())
    vmax = max(pac_single.max(), pac_multi.max())
    cmap = "hot"

    # Plot single GPU comodulogram
    im1 = axes[0].imshow(
        pac_single.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
    )
    axes[0].set_xlabel("Phase Frequency (Hz)")
    axes[0].set_ylabel("Amplitude Frequency (Hz)")
    axes[0].set_title("Single GPU")
    plt.colorbar(im1, ax=axes[0], label="PAC Value")

    # Plot multi-GPU comodulogram
    im2 = axes[1].imshow(
        pac_multi.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
    )
    axes[1].set_xlabel("Phase Frequency (Hz)")
    axes[1].set_ylabel("Amplitude Frequency (Hz)")
    axes[1].set_title(f"Multi-GPU ({torch.cuda.device_count()} GPUs)")
    plt.colorbar(im2, ax=axes[1], label="PAC Value")

    # Plot difference
    diff = pac_multi - pac_single
    im3 = axes[2].imshow(
        diff.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-np.abs(diff).max(),
        vmax=np.abs(diff).max(),
        extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
    )
    axes[2].set_xlabel("Phase Frequency (Hz)")
    axes[2].set_ylabel("Amplitude Frequency (Hz)")
    axes[2].set_title("Difference (Multi - Single)")
    plt.colorbar(im3, ax=axes[2], label="Difference")

    # Add correlation info
    correlation = np.corrcoef(pac_single.flatten(), pac_multi.flatten())[0, 1]
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()

    fig.suptitle(
        f"PAC Comodulogram Comparison\n"
        f"Correlation: {correlation:.6f}, Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}",
        fontsize=14,
    )

    plt.tight_layout()
    return fig, correlation, max_diff, mean_diff


def run_comodulogram_comparison(config):
    """
    Run PAC computation on identical data using single and multi-GPU configurations.

    Args:
        config: Test configuration dictionary

    Returns:
        results: Dictionary with results and metrics
    """
    print_test_header(
        "Comodulogram Consistency Test",
        "Comparing single vs multi-GPU PAC results",
    )

    # Generate test data
    print("\n==> Generating test data...")
    data = generate_test_data(
        config["batch_size"], config["n_channels"], config["seq_len"]
    )
    print(f"Data shape: {data.shape}")

    # Create PAC models
    print("\n==> Creating PAC models...")
    pac_single = create_pac_model(config, multi_gpu=False)
    pac_multi = create_pac_model(config, multi_gpu=True)

    # Get frequency information
    pha_freqs = pac_single.phase_frequencies.cpu().numpy()
    amp_freqs = pac_single.amplitude_frequencies.cpu().numpy()

    # Compute PAC with single GPU
    print("\n==>  Computing PAC with single GPU...")
    pac_values_single = compute_pac_comodulogram(pac_single, data)
    print(f"Single GPU PAC shape: {pac_values_single.shape}")
    print(
        f"Single GPU PAC range: [{pac_values_single.min():.6f}, {pac_values_single.max():.6f}]"
    )

    # Compute PAC with multi-GPU
    print(f"\n==>  Computing PAC with {torch.cuda.device_count()} GPUs...")
    pac_values_multi = compute_pac_comodulogram(pac_multi, data)
    print(f"Multi-GPU PAC shape: {pac_values_multi.shape}")
    print(
        f"Multi-GPU PAC range: [{pac_values_multi.min():.6f}, {pac_values_multi.max():.6f}]"
    )

    # Create comparison plot
    print("\n==> Creating comparison visualization...")
    fig, correlation, max_diff, mean_diff = plot_comodulogram_comparison(
        pac_values_single, pac_values_multi, pha_freqs, amp_freqs
    )

    # Print consistency metrics
    print("\n==> Consistency Metrics:")
    print(f"   Correlation: {correlation:.6f}")
    print(f"   Maximum difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")

    if correlation > 0.999:
        print("    Excellent consistency between single and multi-GPU")
    elif correlation > 0.99:
        print("    Very good consistency between single and multi-GPU")
    elif correlation > 0.95:
        print("   �  Good consistency, but some differences detected")
    else:
        print("   L Poor consistency - investigate implementation")

    results = {
        "correlation": float(correlation),
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "single_gpu_min": float(pac_values_single.min()),
        "single_gpu_max": float(pac_values_single.max()),
        "multi_gpu_min": float(pac_values_multi.min()),
        "multi_gpu_max": float(pac_values_multi.max()),
        "n_gpus": torch.cuda.device_count(),
        "data_shape": list(data.shape),
        "pac_shape": list(pac_values_single.shape),
    }

    return fig, results


def main(args):
    """Main comodulogram comparison function."""
    print_gpu_info()

    if not torch.cuda.is_available():
        print("L CUDA not available")
        return 1

    # Create test configuration
    config = create_test_config()
    config["n_perm"] = args.n_perm

    # Adjust parameters for comodulogram visualization
    config["batch_size"] = args.batch_size
    config["pha_n_bands"] = args.pha_bands
    config["amp_n_bands"] = args.amp_bands

    print("\n==> Test Configuration:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Channels: {config['n_channels']}")
    print(f"   Sequence length: {config['seq_len']} ({config['seq_sec']}s)")
    print(f"   Phase bands: {config['pha_n_bands']}")
    print(f"   Amplitude bands: {config['amp_n_bands']}")
    print(f"   Permutations: {config['n_perm']}")

    # Run comparison
    fig, results = run_comodulogram_comparison(config)

    # Save results
    scitex.io.save(fig, "comodulogram_comparison.gif")
    scitex.io.save(results, "comodulogram_results.yaml")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    script_mode = scitex.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Compare PAC comodulogram between single and multi-GPU"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=4,
        help="Batch size for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--pha_bands",
        type=int,
        default=20,
        help="Number of phase frequency bands (default: %(default)s)",
    )
    parser.add_argument(
        "--amp_bands",
        type=int,
        default=15,
        help="Number of amplitude frequency bands (default: %(default)s)",
    )
    parser.add_argument(
        "--n_perm",
        "-p",
        type=int,
        default=0,
        help="Number of permutations for statistical testing (default: %(default)s)",
    )
    args = parser.parse_args()
    scitex.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    scitex.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
