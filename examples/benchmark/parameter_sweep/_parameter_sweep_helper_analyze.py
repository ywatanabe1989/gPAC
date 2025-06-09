#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 00:44:15 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_analyze.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_analyze.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Standalone script to analyze gPAC benchmark results

import pickle

import numpy as np
import pandas as pd


def load_benchmark_data(file_path):
    """Load benchmark data from pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Convert to DataFrame, excluding problematic columns
        dict_for_df = {
            k: v
            for k, v in data.items()
            if k not in ["sample_comodulogram", "device_ids"]
        }

        df = pd.DataFrame(dict_for_df)

        # Add per-batch timing
        df["computation_time_per_batch"] = (
            df["computation_time_for_all_runs"] / df["n_batches"]
        )

        return df
    except Exception as e:
        mngs.str.printc(
            f"Error loading data from {file_path}: {e}", c=CC["red"]
        )
        return None


def analyze_speedup(df):
    """Analyze speedup between gPAC and TensorPAC."""

    mngs.str.printc("📊 gPAC vs TensorPAC Performance Analysis", c=CC["blue"])
    print("=" * 60)

    # Separate data by package
    gpac_data = df[df["package"] == "gpac"].copy().reset_index(drop=True)
    tensorpac_data = (
        df[df["package"] == "tensorpac"].copy().reset_index(drop=True)
    )

    mngs.str.printc("📈 Dataset Overview:", c=CC["yellow"])
    print(f"   • Total configurations: {len(df)}")
    print(f"   • gPAC configurations: {len(gpac_data)}")
    print(f"   • TensorPAC configurations: {len(tensorpac_data)}")

    # Basic performance statistics
    mngs.str.printc("\n⏱️ Performance Statistics:", c=CC["yellow"])
    mngs.str.printc("   gPAC:", c=CC["blue"])
    print(
        f"     - Mean time: {gpac_data['computation_time_for_all_runs'].mean():.4f}s"
    )
    print(
        f"     - Median time: {gpac_data['computation_time_for_all_runs'].median():.4f}s"
    )
    print(
        f"     - Min time: {gpac_data['computation_time_for_all_runs'].min():.4f}s"
    )
    print(
        f"     - Max time: {gpac_data['computation_time_for_all_runs'].max():.4f}s"
    )

    mngs.str.printc("   TensorPAC:", c=CC["orange"])
    print(
        f"     - Mean time: {tensorpac_data['computation_time_for_all_runs'].mean():.4f}s"
    )
    print(
        f"     - Median time: {tensorpac_data['computation_time_for_all_runs'].median():.4f}s"
    )
    print(
        f"     - Min time: {tensorpac_data['computation_time_for_all_runs'].min():.4f}s"
    )
    print(
        f"     - Max time: {tensorpac_data['computation_time_for_all_runs'].max():.4f}s"
    )

    # Calculate speedups for matching configurations
    speedups = []
    comparison_params = [
        "batch_size",
        "n_channels",
        "duration_sec",
        "pha_n_bands",
        "amp_n_bands",
        "n_perm",
    ]

    mngs.str.printc("\n🔍 Finding matching configurations...", c=CC["yellow"])

    matched_count = 0
    for gpac_idx, gpac_row in gpac_data.iterrows():
        for tp_idx, tp_row in tensorpac_data.iterrows():
            # Check if configurations match
            match = True
            for param in comparison_params:
                if param in gpac_row and param in tp_row:
                    if gpac_row[param] != tp_row[param]:
                        match = False
                        break

            if match:
                speedup = (
                    tp_row["computation_time_for_all_runs"]
                    / gpac_row["computation_time_for_all_runs"]
                )
                speedups.append(
                    {
                        "speedup": speedup,
                        "gpac_time": gpac_row["computation_time_for_all_runs"],
                        "tensorpac_time": tp_row[
                            "computation_time_for_all_runs"
                        ],
                        "batch_size": gpac_row["batch_size"],
                        "n_channels": gpac_row["n_channels"],
                        "duration_sec": gpac_row["duration_sec"],
                        "pha_n_bands": gpac_row["pha_n_bands"],
                        "amp_n_bands": gpac_row["amp_n_bands"],
                    }
                )
                matched_count += 1
                break

    print(f"   • Found {matched_count} matching configuration pairs")

    if speedups:
        speedup_values = [s["speedup"] for s in speedups]

        mngs.str.printc("\n🚀 Speedup Analysis:", c=CC["green"])
        print(f"   • Mean speedup: {np.mean(speedup_values):.1f}x")
        print(f"   • Median speedup: {np.median(speedup_values):.1f}x")
        print(f"   • Min speedup: {np.min(speedup_values):.1f}x")
        print(f"   • Max speedup: {np.max(speedup_values):.1f}x")
        print(f"   • Standard deviation: {np.std(speedup_values):.1f}x")

        # Show extreme cases
        best_speedup = max(speedups, key=lambda x: x["speedup"])
        worst_speedup = min(speedups, key=lambda x: x["speedup"])

        mngs.str.printc(
            f"\n🏆 Best Speedup: {best_speedup['speedup']:.1f}x", c=CC["green"]
        )
        print(
            f"   Config: batch={best_speedup['batch_size']}, ch={best_speedup['n_channels']}, "
            f"dur={best_speedup['duration_sec']}s"
        )
        print(
            f"   Times: gPAC={best_speedup['gpac_time']:.4f}s, TensorPAC={best_speedup['tensorpac_time']:.4f}s"
        )

        mngs.str.printc(
            f"\n⚠️ Worst Speedup: {worst_speedup['speedup']:.1f}x",
            c=CC["yellow"],
        )
        print(
            f"   Config: batch={worst_speedup['batch_size']}, ch={worst_speedup['n_channels']}, "
            f"dur={worst_speedup['duration_sec']}s"
        )
        print(
            f"   Times: gPAC={worst_speedup['gpac_time']:.4f}s, TensorPAC={worst_speedup['tensorpac_time']:.4f}s"
        )

    else:
        # If no exact matches, calculate overall ratios
        mngs.str.printc(
            "\n⚠️ No exact matching configurations found.", c=CC["yellow"]
        )
        print("Calculating overall performance ratios...")

        median_ratio = (
            tensorpac_data["computation_time_for_all_runs"].median()
            / gpac_data["computation_time_for_all_runs"].median()
        )
        mean_ratio = (
            tensorpac_data["computation_time_for_all_runs"].mean()
            / gpac_data["computation_time_for_all_runs"].mean()
        )

        print(f"   • Median time ratio: {median_ratio:.1f}x")
        print(f"   • Mean time ratio: {mean_ratio:.1f}x")

        speedups = [
            {
                "speedup": median_ratio,
                "gpac_time": gpac_data[
                    "computation_time_for_all_runs"
                ].median(),
                "tensorpac_time": tensorpac_data[
                    "computation_time_for_all_runs"
                ].median(),
            }
        ]

    return speedups


def create_performance_plots(df, speedups=None):
    """Create performance comparison plots."""

    mngs.str.printc(
        "\n🎨 Creating performance visualizations...", c=CC["blue"]
    )

    fig, axes = mngs.plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall performance comparison
    ax1 = axes[0, 0]
    gpac_times = df[df["package"] == "gpac"]["computation_time_for_all_runs"]
    tensorpac_times = df[df["package"] == "tensorpac"][
        "computation_time_for_all_runs"
    ]

    bp = ax1.boxplot(
        [gpac_times, tensorpac_times],
        labels=["gPAC", "TensorPAC"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")

    ax1.set_xyt(
        "",
        "Computation Time (seconds)",
        "Overall Performance Comparison",
    )
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Add median speedup annotation
    if speedups:
        median_speedup = np.median([s["speedup"] for s in speedups])
        ax1.text(
            0.5,
            0.95,
            f"Median Speedup: {median_speedup:.1f}x",
            transform=ax1.transAxes,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # 2. Performance vs duration
    ax2 = axes[0, 1]
    for package, color in [("gpac", "blue"), ("tensorpac", "red")]:
        pkg_data = df[df["package"] == package]
        if len(pkg_data) > 0:
            duration_stats = pkg_data.groupby("duration_sec")[
                "computation_time_for_all_runs"
            ].mean()
            ax2.plot(
                duration_stats.index,
                duration_stats.values,
                "o-",
                label=package,
                color=color,
                alpha=0.7,
                linewidth=2,
            )

    ax2.set_xyt(
        "Duration (seconds)",
        "Computation Time (seconds)",
        "Performance vs Signal Duration",
    )
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Performance vs frequency bands
    ax3 = axes[1, 0]
    for package, color in [("gpac", "blue"), ("tensorpac", "red")]:
        pkg_data = df[df["package"] == package]
        if len(pkg_data) > 0:
            band_stats = pkg_data.groupby("pha_n_bands")[
                "computation_time_for_all_runs"
            ].mean()
            ax3.plot(
                band_stats.index,
                band_stats.values,
                "s-",
                label=package,
                color=color,
                alpha=0.7,
                linewidth=2,
            )

    ax3.set_xyt(
        "Number of Phase Bands",
        "Computation Time (seconds)",
        "Performance vs Phase Bands",
    )
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Speedup distribution (if available)
    ax4 = axes[1, 1]
    if speedups and len(speedups) > 1:
        speedup_values = [s["speedup"] for s in speedups]
        ax4.hist(
            speedup_values,
            bins=20,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        ax4.axvline(
            np.median(speedup_values),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(speedup_values):.1f}x",
        )
        ax4.set_xyt(
            "Speedup Factor",
            "Frequency",
            "Speedup Distribution",
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Show data size vs performance instead
        df_copy = df.copy()
        df_copy["data_size"] = (
            df_copy["batch_size"]
            * df_copy["n_channels"]
            * df_copy["duration_sec"]
            * df_copy["fs"]
        )

        for package, color in [("gpac", "blue"), ("tensorpac", "red")]:
            pkg_data = df_copy[df_copy["package"] == package]
            if len(pkg_data) > 0:
                ax4.scatter(
                    pkg_data["data_size"],
                    pkg_data["computation_time_for_all_runs"],
                    alpha=0.6,
                    label=package,
                    color=color,
                    s=30,
                )

        ax4.set_xyt(
            "Data Size (total samples)",
            "Computation Time (seconds)",
            "Performance vs Data Size",
        )
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("gPAC vs TensorPAC Performance Analysis", fontsize=16, y=1.02)

    # Save the plot using mngs
    mngs.io.save(fig, "gpac_performance_analysis.gif")
    mngs.str.printc(
        "   • Saved plot as: gpac_performance_analysis.gif", c=CC["green"]
    )

    return fig


def main(args):
    """Main analysis function."""

    # Default file path - adjust as needed
    default_path = os.path.join(
        __DIR__, "parameter_sweep_benchmark_out", "benchmark_results.pkl"
    )

    # Check if file exists
    if hasattr(args, "file") and args.file:
        file_path = args.file
    else:
        file_path = default_path

    if not os.path.exists(file_path):
        mngs.str.printc(f"❌ File not found: {file_path}", c=CC["red"])
        mngs.str.printc(
            "Please provide the correct path to your benchmark results file.",
            c=CC["yellow"],
        )
        return 1

    mngs.str.printc(f"📂 Loading data from: {file_path}", c=CC["blue"])

    # Load data
    df = load_benchmark_data(file_path)
    if df is None:
        return 1

    mngs.str.printc(
        f"✅ Successfully loaded {len(df)} configurations", c=CC["green"]
    )

    # Analyze speedup
    speedups = analyze_speedup(df)

    # Create visualizations
    fig = create_performance_plots(df, speedups)
    plt.show()

    mngs.str.printc("\n✅ Analysis complete!", c=CC["green"])
    print("=" * 60)

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze gPAC benchmark results"
    )
    parser.add_argument(
        "--file", type=str, help="Path to benchmark results file"
    )
    args = parser.parse_args()
    return args


def run_main():
    global CONFIG, CC, sys, plt, mngs

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
