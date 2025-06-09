#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 18:32:01 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_analyze.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_analyze.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import mngs
import numpy as np
import pandas as pd


def load_and_prepare_data(file_path):
    PATH = "./examples/benchmark/parameter_sweep/parameter_sweep_benchmark_out/benchmark_results_quick.pkl"
    data = mngs.io.load(PATH)

    dict_for_df = {
        k: v
        for k, v in data.items()
        if k not in ["sample_comodulogram", "device_ids"]
    }

    df = pd.DataFrame(dict_for_df)
    df["computation_time_per_batch"] = (
        df["computation_time_for_all_runs"] / df["n_batches"]
    )

    return df


def analyze_performance_trends(df):
    results = {}

    scaling_params = [
        "batch_size",
        "n_channels",
        "duration_sec",
        "pha_n_bands",
        "amp_n_bands",
    ]

    for param in scaling_params:
        param_analysis = {}

        for package in ["gpac", "tensorpac"]:
            pkg_data = df[df["package"] == package]
            if len(pkg_data) == 0:
                continue

            grouped = pkg_data.groupby(param)[
                "computation_time_per_batch"
            ].mean()

            if len(grouped) > 1:
                scaling_factor = grouped.max() / grouped.min()
                param_range = grouped.index.max() / grouped.index.min()

                param_analysis[f"{package}_scaling_factor"] = scaling_factor
                param_analysis[f"{package}_param_range"] = param_range
                param_analysis[f"{package}_scaling_efficiency"] = np.log(
                    scaling_factor
                ) / np.log(param_range)

        results[param] = param_analysis

    return results


def find_optimal_configurations(df):
    best_overall = df.loc[df["computation_time_per_batch"].idxmin()]

    large_batch = df[df["batch_size"] >= 8]
    if len(large_batch) > 0:
        best_large_batch = large_batch.loc[
            large_batch["computation_time_per_batch"].idxmin()
        ]
    else:
        best_large_batch = None

    high_res = df[(df["pha_n_bands"] >= 64) & (df["amp_n_bands"] >= 64)]
    if len(high_res) > 0:
        best_high_res = high_res.loc[
            high_res["computation_time_per_batch"].idxmin()
        ]
    else:
        best_high_res = None

    return {
        "best_overall": best_overall.to_dict(),
        "best_large_batch": (
            best_large_batch.to_dict()
            if best_large_batch is not None
            else None
        ),
        "best_high_resolution": (
            best_high_res.to_dict() if best_high_res is not None else None
        ),
    }


def analyze_memory_efficiency(df):
    df_analysis = df.copy()

    df_analysis["estimated_memory_mb"] = (
        df_analysis["batch_size"]
        * df_analysis["n_channels"]
        * df_analysis["duration_sec"]
        * df_analysis["fs"]
        * 4
        * 2
    ) / (1024 * 1024)

    df_analysis["memory_efficiency"] = (
        df_analysis["computation_time_per_batch"]
        / df_analysis["estimated_memory_mb"]
    )

    memory_stats = (
        df_analysis.groupby("package")["memory_efficiency"]
        .agg(["mean", "median", "std", "min", "max"])
        .to_dict()
    )

    return memory_stats


def main(args):
    df = load_and_prepare_data(args.input)

    mngs.os.makedirs(args.output_dir, exist_ok=True)

    scaling_analysis = analyze_performance_trends(df)
    optimal_configs = find_optimal_configurations(df)
    memory_analysis = analyze_memory_efficiency(df)

    analysis_results = {
        "scaling_analysis": scaling_analysis,
        "optimal_configurations": optimal_configs,
        "memory_efficiency": memory_analysis,
    }

    mngs.io.save(
        analysis_results, f"{args.output_dir}/performance_analysis.yaml"
    )

    print(f"Analysis complete. Results saved to {args.output_dir}")

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="benchmark_results.pkl", help="Input file path"
    )
    parser.add_argument(
        "--output-dir", default="./analysis_output", help="Output directory"
    )
    args = parser.parse_args()
    return args


def run_main():
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
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
