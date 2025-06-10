#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 15:45:00 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/pac_values_comparison_with_tensorpac/run_all_comparisons.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/pac_values_comparison_with_tensorpac/run_all_comparisons.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Runs comprehensive PAC comparison between gPAC and Tensorpac
  - Tests multiple parameter configurations
  - Generates summary report with all metrics
  - Creates visualizations for different scenarios

Dependencies:
  - scripts:
    - ./compare_comodulograms.py
  - packages:
    - subprocess, pandas, matplotlib, mngs
IO:
  - input-files:
    - None
  - output-files:
    - comparison_summary_report.pdf
    - all_comparisons_results.csv
"""

"""Imports"""
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mngs
import pandas as pd

"""Functions & Classes"""
def run_comparison_test(
    test_name: str,
    params: dict,
    output_dir: Path,
) -> dict:
    """Run a single comparison test with given parameters."""
    print(f"\n{'='*60}")
    print(f"Running test: {test_name}")
    print(f"{'='*60}")
    
    # Create test directory
    test_dir = output_dir / test_name
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Build command
    cmd = ["python", "compare_comodulograms.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Run comparison
    original_cwd = Path.cwd()
    try:
        os.chdir(test_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in {test_name}:")
            print(result.stderr)
            return None
        
        # Load results
        if Path("comparison_results.yaml").exists():
            results = mngs.io.load("comparison_results.yaml")
            results["test_name"] = test_name
            return results
        else:
            print(f"No results found for {test_name}")
            return None
            
    finally:
        os.chdir(original_cwd)


def create_summary_report(all_results: list, output_dir: Path):
    """Create summary report with all comparison results."""
    fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    test_names = [r["test_name"] for r in all_results if r is not None]
    correlations = [r["pac_correlation"] for r in all_results if r is not None]
    rmses = [r["pac_rmse"] for r in all_results if r is not None]
    z_correlations = [r.get("z_correlation", None) for r in all_results if r is not None]
    
    # Correlation comparison
    ax = axes[0, 0]
    ax.bar(range(len(test_names)), correlations, alpha=0.7)
    ax.set_xyt("Test Case", "Correlation", "PAC Correlation (gPAC vs Tensorpac)")
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.axhline(y=0.9, color="r", linestyle="--", label="Good correlation (0.9)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE comparison
    ax = axes[0, 1]
    ax.bar(range(len(test_names)), rmses, alpha=0.7, color="orange")
    ax.set_xyt("Test Case", "RMSE", "Root Mean Square Error")
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    
    # Mean PAC values comparison
    ax = axes[1, 0]
    gpac_means = [r["gpac_pac_mean"] for r in all_results if r is not None]
    tensorpac_means = [r["tensorpac_pac_mean"] for r in all_results if r is not None]
    
    xx = range(len(test_names))
    width = 0.35
    ax.bar([x - width/2 for x in xx], gpac_means, width, label="gPAC", alpha=0.7)
    ax.bar([x + width/2 for x in xx], tensorpac_means, width, label="Tensorpac", alpha=0.7)
    ax.set_xyt("Test Case", "Mean PAC Value", "Mean PAC Values")
    ax.set_xticks(xx)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Z-score correlation (if available)
    ax = axes[1, 1]
    z_corr_valid = [z for z in z_correlations if z is not None]
    if z_corr_valid:
        test_names_z = [test_names[i] for i, z in enumerate(z_correlations) if z is not None]
        ax.bar(range(len(test_names_z)), z_corr_valid, alpha=0.7, color="green")
        ax.set_xyt("Test Case", "Z-score Correlation", "Z-score Correlation")
        ax.set_xticks(range(len(test_names_z)))
        ax.set_xticklabels(test_names_z, rotation=45, ha="right")
        ax.axhline(y=0.9, color="r", linestyle="--", label="Good correlation (0.9)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Z-score data available", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("gPAC vs Tensorpac Comparison Summary", fontsize=14)
    plt.tight_layout()
    
    mngs.io.save(fig, output_dir / "comparison_summary_report.gif")
    plt.close()


def main(args):
    """Main function to run all comparisons."""
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define test configurations
    test_configs = {
        "low_noise": {
            "batch_size": 2,
            "n_channels": 4,
            "duration": 10,
            "fs": 512,
            "phase_freq": 10,
            "amp_freq": 80,
            "pac_strength": 0.8,
            "noise_level": 0.1,
            "n_perm": 100 if not args.quick else 0,
        },
        "high_noise": {
            "batch_size": 2,
            "n_channels": 4,
            "duration": 10,
            "fs": 512,
            "phase_freq": 10,
            "amp_freq": 80,
            "pac_strength": 0.5,
            "noise_level": 1.0,
            "n_perm": 100 if not args.quick else 0,
        },
        "multi_pac": {
            "batch_size": 2,
            "n_channels": 4,
            "duration": 10,
            "fs": 512,
            "phase_freq": 8,
            "amp_freq": 60,
            "pac_strength": 0.6,
            "noise_level": 0.5,
            "n_perm": 100 if not args.quick else 0,
        },
        "high_freq": {
            "batch_size": 2,
            "n_channels": 4,
            "duration": 10,
            "fs": 1024,
            "phase_freq": 20,
            "amp_freq": 120,
            "pac_strength": 0.7,
            "noise_level": 0.3,
            "n_perm": 100 if not args.quick else 0,
        },
    }
    
    # Run all tests
    all_results = []
    for test_name, params in test_configs.items():
        result = run_comparison_test(test_name, params, output_dir)
        if result:
            all_results.append(result)
    
    # Create summary report
    if all_results:
        create_summary_report(all_results, output_dir)
        
        # Save all results to CSV
        df_data = []
        for result in all_results:
            df_data.append({
                "test_name": result["test_name"],
                "pac_correlation": result["pac_correlation"],
                "pac_rmse": result["pac_rmse"],
                "z_correlation": result.get("z_correlation", None),
                "gpac_mean": result["gpac_pac_mean"],
                "gpac_std": result["gpac_pac_std"],
                "tensorpac_mean": result["tensorpac_pac_mean"],
                "tensorpac_std": result["tensorpac_pac_std"],
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_dir / "all_comparisons_results.csv", index=False)
        
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        print(df.to_string())
        
        # Calculate overall statistics
        mean_correlation = df["pac_correlation"].mean()
        mean_rmse = df["pac_rmse"].mean()
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Correlation: {mean_correlation:.4f}")
        print(f"  Mean RMSE: {mean_rmse:.4f}")
        
        if mean_correlation > 0.9:
            print("  ✓ Excellent agreement between gPAC and Tensorpac")
        elif mean_correlation > 0.8:
            print("  ✓ Good agreement between gPAC and Tensorpac")
        else:
            print("  ⚠ Moderate agreement - investigate differences")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive PAC comparison tests"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests without permutation testing",
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt

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