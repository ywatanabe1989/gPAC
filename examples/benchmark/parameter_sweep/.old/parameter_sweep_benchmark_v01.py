#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 11:09:25 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/performance/parameter_sweep/parameter_sweep_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/parameter_sweep/parameter_sweep_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Systematic parameter sweep benchmark for gPAC vs TensorPAC
  - Varies one parameter at a time to clarify dependencies
  - Creates comprehensive visualizations of performance scaling

Dependencies:
  - scripts: None
  - packages: gpac, tensorpac, torch, numpy, matplotlib, mngs, yaml

IO:
  - input-files: None
  - output-files: ./parameter_sweep_out/parameter_sweep_results.yaml, ./parameter_sweep_out/parameter_dependencies.gif
"""

"""Imports"""
import argparse
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mngs
from tensorpac import Pac as TensorPAC

sys.path.append("../../../src")
from collections import defaultdict

import pandas as pd
from gpac import PAC  # Updated import to match latest implementation

"""Parameters"""
# Baseline parameters - reduced for TensorPAC compatibility
BASELINE = {
    # Signal Size Parameters
    "batch_size": 4,  # Reduced for TensorPAC compatibility
    "n_channels": 8,  # Reduced parameter
    "seq_sec": 4.0,  # Reduced duration in seconds
    "fs": 250,  # Standard sampling frequency
    # PAC Resolution Parameters
    "pha_n_bands": 10,  # Reduced for better TensorPAC compatibility
    "amp_n_bands": 15,  # Reduced for better TensorPAC compatibility
    # Computation Parameters
    "n_perm": 0,  # Start with no permutations for speed
    "fp16": False,
    "multi_gpu": False,  # Add multi-GPU option
    "device": "cuda",
}

# Parameters to vary (one at a time) - reduced ranges for TensorPAC compatibility
VARIATIONS = {
    # Signal Size Parameters
    "batch_size": [1, 2, 4, 8, 16],  # Reduced max batch size
    "n_channels": [1, 2, 4, 8, 16],  # Reduced max channels
    "seq_sec": [1.0, 2.0, 4.0, 8.0],  # Reduced duration range
    "fs": [250, 500, 1000],  # Reduced frequency range
    # PAC Resolution Parameters - reduced ranges
    "pha_n_bands": [5, 10, 15, 20],  # Reduced range
    "amp_n_bands": [5, 10, 15, 20],  # Reduced range
    # Computation Parameters
    "n_perm": [0, 16, 50, 100, 200],  # Updated to match latest examples
    "fp16": [False, True],
    "multi_gpu": [False, True],  # Add multi-GPU comparison
}

"""Functions & Classes"""
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Parameter sweep benchmark for gPAC"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=3,
        help="Number of repeats for each measurement",
    )
    parser.add_argument(
        "--no-quick",
        action="store_true",
        help="Quick test with fewer parameter values",
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    args.quick = not args.no_quick
    return args


def benchmark_gpac(
    batch_size: int,
    n_channels: int,
    seq_sec: float,
    fs: int,
    pha_n_bands: int,
    amp_n_bands: int,
    n_perm: int,
    fp16: bool = False,
    multi_gpu: bool = False,
    device: str = "cuda",
) -> tuple:
    """Benchmark gPAC with given parameters using correct API."""
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create synthetic data
    seq_len = int(seq_sec * fs)
    data = torch.randn(batch_size, n_channels, seq_len)

    # Configure device
    device_ids = "all" if multi_gpu else [0]

    # Initialize PAC with correct API
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=pha_n_bands,
        amp_start_hz=30.0,
        amp_end_hz=100.0,
        amp_n_bands=amp_n_bands,
        device_ids=device_ids,
        n_perm=n_perm,
        fp16=fp16,  # Add fp16 parameter
    )

    # Move to appropriate device if CUDA
    if device == "cuda" and torch.cuda.is_available():
        pac = pac.cuda()
        data = data.cuda()
    elif device == "cpu":
        pac = pac.cpu()
        data = data.cpu()

    # Multiple warm-up runs for GPU to stabilize performance
    if device == "cuda":
        for _ in range(3):
            with torch.no_grad():
                _ = pac(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    else:
        # Single warm-up for CPU
        with torch.no_grad():
            _ = pac(data)

    # Time the computation
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        result = pac(data)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    comp_time = time.time() - start_time

    return comp_time, result


def benchmark_tensorpac(data, fs, pha_n_bands, amp_n_bands, n_perm):
    """Benchmark TensorPAC with given parameters."""
    batch_size, n_chs, n_samples = data.shape

    # TensorPAC expects (n_epochs, n_channels, n_times)
    # For multi-channel, we need to process each channel separately
    # and average the results

    # IMPORTANT: We use explicit frequency bands instead of TensorPAC's string
    # configurations (e.g., 'mres', 'hres') to ensure fair comparison with gPAC.
    # String configurations in TensorPAC create overlapping bands which differ
    # from gPAC's sequential non-overlapping approach.

    # Define explicit frequency bands to match gPAC (sequential non-overlapping)
    pha_edges = np.linspace(2, 20, pha_n_bands + 1)
    amp_edges = np.linspace(30, 100, amp_n_bands + 1)

    # Convert to band pairs for TensorPAC
    f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]
    f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]

    # Initialize PAC with explicit bands
    pac = TensorPAC(
        idpac=(2, 0, 0),
        f_pha=f_pha,
        f_amp=f_amp,
        dcomplex="hilbert",
        verbose=False,
    )

    # Reshape data for TensorPAC (n_epochs * n_channels, n_times)
    data_reshaped = data.reshape(-1, n_samples)

    # Time the computation
    start_time = time.time()
    result = pac.filterfit(
        fs, data_reshaped, n_perm=n_perm if n_perm else 0, verbose=False
    )
    comp_time = time.time() - start_time

    return comp_time, result


def run_parameter_sweep(args):
    """Run systematic parameter sweep."""
    results = defaultdict(list)

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Running parameter sweep on device: {device}")
    print("=" * 60)

    # Quick mode adjustments
    if args.quick:
        for param in VARIATIONS:
            if param not in ["fp16", "device"]:
                VARIATIONS[param] = VARIATIONS[param][
                    ::2
                ]  # Take every other value

    # Iterate through each parameter
    for param_name, param_values in VARIATIONS.items():
        print(f"\nVarying parameter: {param_name}")
        print("-" * 40)

        for param_value in param_values:
            # Create test configuration
            config = BASELINE.copy()
            config[param_name] = param_value
            config["device"] = device

            # Skip invalid combinations
            if param_name == "fp16" and param_value and device == "cpu":
                continue

            print(f"  {param_name} = {param_value}...", end=" ", flush=True)

            # Skip invalid combinations
            if config["multi_gpu"] and not torch.cuda.is_available():
                continue
            if config["multi_gpu"] and torch.cuda.device_count() < 2:
                continue

            # Run benchmarks
            times_gpac = []
            times_tp = []

            for _ in range(args.n_repeats):
                # gPAC
                try:
                    time_gpac, _ = benchmark_gpac(
                        batch_size=config["batch_size"],
                        n_channels=config["n_channels"],
                        seq_sec=config["seq_sec"],
                        fs=config["fs"],
                        pha_n_bands=config["pha_n_bands"],
                        amp_n_bands=config["amp_n_bands"],
                        n_perm=config["n_perm"],
                        fp16=config["fp16"],
                        multi_gpu=config["multi_gpu"],
                        device=config["device"],
                    )
                    times_gpac.append(time_gpac)
                except Exception as e:
                    print(f"gPAC error: {e}")
                    times_gpac.append(np.nan)

                # TensorPAC (skip if using GPU-specific features, but allow batch_size > 1)
                if not config["fp16"] and not config["multi_gpu"]:
                    try:
                        # Generate data for TensorPAC
                        seq_len = int(config["fs"] * config["seq_sec"])
                        data_tp = np.random.randn(
                            config["batch_size"], config["n_channels"], seq_len
                        ).astype(np.float32)

                        time_tp, _ = benchmark_tensorpac(
                            data_tp,
                            config["fs"],
                            config["pha_n_bands"],
                            config["amp_n_bands"],
                            config["n_perm"],
                        )
                        times_tp.append(time_tp)
                    except Exception as e:
                        print(f"TensorPAC error: {e}")
                        times_tp.append(np.nan)
                else:
                    times_tp.append(np.nan)

            # Store results
            results["param_name"].append(param_name)
            results["param_value"].append(param_value)
            results["gpac_time_mean"].append(np.nanmean(times_gpac))
            results["gpac_time_std"].append(np.nanstd(times_gpac))
            results["tensorpac_time_mean"].append(np.nanmean(times_tp))
            results["tensorpac_time_std"].append(np.nanstd(times_tp))
            results["speedup"].append(
                np.nanmean(times_tp) / np.nanmean(times_gpac)
            )
            results["config"].append(config.copy())

            # Calculate throughput (samples per second)
            seq_len = int(config["fs"] * config["seq_sec"])
            total_samples = (
                config["batch_size"] * config["n_channels"] * seq_len
            )
            results["gpac_throughput"].append(
                total_samples / np.nanmean(times_gpac) / 1e6
            )

            print(
                f"gPAC: {np.nanmean(times_gpac):.4f}s, "
                + f"TensorPAC: {np.nanmean(times_tp):.4f}s, "
                + f"Speedup: {results['speedup'][-1]:.1f}x"
            )

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    return df


def plot_parameter_dependency_plots(df):
    """Create comprehensive visualization of parameter dependencies."""
    import mngs

    # Get unique parameters
    params = df["param_name"].unique()
    n_params = len(params)

    # Create figure with subplots
    fig = mngs.plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Color map for parameters
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))

    for idx, param in enumerate(params):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        # Filter data for this parameter
        param_data = df[df["param_name"] == param].copy()
        param_data = param_data.sort_values("param_value")

        # Plot gPAC performance
        ax.errorbar(
            param_data["param_value"],
            param_data["gpac_time_mean"],
            yerr=param_data["gpac_time_std"],
            label="gPAC",
            marker="o",
            color=colors[idx],
            linewidth=2,
            markersize=8,
            capsize=5,
        )

        # Plot TensorPAC performance where available
        valid_tp = ~param_data["tensorpac_time_mean"].isna()
        if valid_tp.any():
            ax.errorbar(
                param_data.loc[valid_tp, "param_value"],
                param_data.loc[valid_tp, "tensorpac_time_mean"],
                yerr=param_data.loc[valid_tp, "tensorpac_time_std"],
                label="TensorPAC",
                marker="s",
                color=colors[idx],
                alpha=0.5,
                linewidth=2,
                markersize=8,
                capsize=5,
                linestyle="--",
            )

        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f'Performance vs {param.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Log scale for some parameters
        if param in ["batch_size", "n_channels", "fs"]:
            ax.set_xscale("log", base=2)

        # Add speedup on secondary y-axis where applicable
        if valid_tp.any():
            ax2 = ax.twinx()
            speedups = param_data.loc[valid_tp, "speedup"]
            ax2.plot(
                param_data.loc[valid_tp, "param_value"],
                speedups,
                "g-",
                alpha=0.7,
                linewidth=2,
                label="Speedup",
            )
            ax2.set_ylabel("Speedup (x)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")

    plt.suptitle("gPAC Parameter Dependencies", fontsize=16)

    # Create throughput scaling plot
    fig2, axes = mngs.plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Select key parameters for throughput analysis
    throughput_params = ["batch_size", "n_channels", "seq_sec", "pha_n_bands"]

    for idx, param in enumerate(throughput_params):
        ax = axes[idx]
        param_data = df[df["param_name"] == param].copy()
        param_data = param_data.sort_values("param_value")

        ax.plot(
            param_data["param_value"],
            param_data["gpac_throughput"],
            "o-",
            color=colors[idx],
            linewidth=2,
            markersize=8,
            label="gPAC",
        )

        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel("Throughput (Million samples/s)")
        ax.set_title(f'Throughput vs {param.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

        if param in ["batch_size", "n_channels"]:
            ax.set_xscale("log", base=2)

    plt.suptitle("gPAC Throughput Scaling", fontsize=16)
    # plt.tight_layout()

    return fig, fig2


def main(args):
    """Main function."""

    print("\n" + "=" * 60)
    print("gPAC Parameter Sweep Benchmark")
    print("=" * 60)

    # Run parameter sweep
    df_sweep = run_parameter_sweep(args)
    mngs.io.save(
        df_sweep.to_dict(),
        "./parameter_sweep_benchmark_out/parameter_sweep_results.yaml",
    )
    mngs.io.save(
        df_sweep, "./parameter_sweep_benchmark_out/parameter_sweep_results.csv"
    )

    # Create visualizations
    print("\nCreating parameter dependency plots...")
    fig1, fig2 = plot_parameter_dependency_plots(df_sweep)
    mngs.io.save(
        fig1, "./parameter_sweep_benchmark_out/parameter_dependencies.gif"
    )
    mngs.io.save(
        fig2, "./parameter_sweep_benchmark_out/throughput_scaling.gif"
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for param in df_sweep["param_name"].unique():
        param_data = df_sweep[df_sweep["param_name"] == param]
        print(f"\n{param}:")
        print(
            f"  gPAC time range: {param_data['gpac_time_mean'].min():.4f} - "
            f"{param_data['gpac_time_mean'].max():.4f} seconds"
        )
        valid_speedups = param_data["speedup"][~param_data["speedup"].isna()]
        if len(valid_speedups) > 0:
            print(
                f"  Speedup range: {valid_speedups.min():.1f}x - {valid_speedups.max():.1f}x"
            )
        print(
            f"  Max throughput: {param_data['gpac_throughput'].max():.2f} M samples/s"
        )

    return 0


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


if __name__ == "__main__":
    run_main()

# EOF
