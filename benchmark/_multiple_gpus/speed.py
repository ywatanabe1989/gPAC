#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-03 15:15:55 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/performance/multiple_gpus/speed.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/multiple_gpus/speed.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests computational speed comparison between single and multi-GPU
  - Measures speedup for identical workloads (same data size)
  - Focuses on computational acceleration, not VRAM scaling

Dependencies:
  - scripts:
    - ./utils.py
  - packages:
    - torch, numpy, mngs, gpac
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - speed_comparison.gif
    - speed_results.yaml
"""

"""Imports"""
import argparse

import torch
from utils import (calculate_efficiency, calculate_speedup,
                   create_comparison_plot, create_speed_comparison_plot, create_pac_model,
                   create_test_config, generate_test_data,
                   measure_execution_time, print_config, print_gpu_info,
                   print_results_summary, print_test_header)

"""Functions & Classes"""
def run_speed_test(config: dict) -> dict:
    """
    Run speed comparison test with identical workloads.

    Tests the same data size on single vs multi-GPU to measure
    computational speedup, not VRAM scaling benefits.
    """
    print_test_header(
        "Speed Test", "Computational acceleration for identical workloads"
    )

    # Test different batch sizes
    batch_multipliers = [1, 2, 3, 4, 6, 8, 12]
    base_batch = config["batch_size"]

    results = {
        "batch_sizes": [],
        "single_times": [],
        "multi_times": [],
        "single_memory": [],
        "multi_memory": [],
        "single_success": [],
        "multi_success": [],
    }

    for multiplier in batch_multipliers:
        batch_size = base_batch * multiplier
        print(f"\n--- Testing batch size: {batch_size} ---")

        # Generate identical test data for both tests
        data = generate_test_data(
            batch_size, config["n_channels"], config["seq_len"]
        )
        print(f"Data shape: {data.shape}")

        # Test 1: Single GPU
        pac_single = create_pac_model(config, multi_gpu=False)
        single_time, single_success, single_mem = measure_execution_time(
            pac_single, data
        )

        if single_success:
            print(f"✅ Single GPU: {single_time:.2f}s, {single_mem:.1f} MB")
        else:
            print(f"❌ Single GPU: FAILED")

        # Test 2: Multi-GPU (same data)
        pac_multi = create_pac_model(config, multi_gpu=True)

        if torch.cuda.device_count() > 1:
            print(
                f"Multi-GPU: Using {torch.cuda.device_count()} GPUs: {pac_multi.device_ids}"
            )

        multi_time, multi_success, multi_mem = measure_execution_time(
            pac_multi, data
        )

        if multi_success:
            print(f"✅ Multi-GPU: {multi_time:.2f}s, {multi_mem:.1f} MB")
        else:
            print(f"❌ Multi-GPU: FAILED")

        # Calculate and display metrics
        if single_success and multi_success:
            speedup = calculate_speedup(single_time, multi_time)
            efficiency = calculate_efficiency(
                speedup, torch.cuda.device_count()
            )
            print(f"📈 Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1%}")

        # Store results
        results["batch_sizes"].append(batch_size)
        results["single_times"].append(single_time)
        results["multi_times"].append(multi_time)
        results["single_memory"].append(single_mem)
        results["multi_memory"].append(multi_mem)
        results["single_success"].append(single_success)
        results["multi_success"].append(multi_success)

        # Stop if both fail (memory limit reached)
        if not single_success and not multi_success:
            print("⚠️  Both configurations failed - stopping test")
            break

    return results


def main(args):
    """Main speed test function."""
    print_gpu_info()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1

    # Create test configuration
    config = create_test_config()
    config["n_perm"] = args.n_perm
    print_config(config)

    # Run speed test
    results = run_speed_test(config)

    # Find best speedup result
    valid_results = [
        i
        for i in range(len(results["batch_sizes"]))
        if results["single_success"][i] and results["multi_success"][i]
    ]

    if valid_results:
        # Show best speedup result
        best_idx = max(
            valid_results,
            key=lambda i: calculate_speedup(
                results["single_times"][i], results["multi_times"][i]
            ),
        )
        print_results_summary(
            results["single_times"][best_idx],
            results["multi_times"][best_idx],
            results["batch_sizes"][best_idx],
            torch.cuda.device_count(),
        )

        # Create visualization with new speed comparison plot
        fig = create_speed_comparison_plot(
            results,
            "Multi-GPU Speed Test: Calculation Time vs Sample Size",
        )

        # Save results
        import mngs

        mngs.io.save(results, "speed_results.yaml")
        mngs.io.save(fig, "speed_comparison.gif")

    else:
        print("❌ No successful tests completed")
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Multi-GPU speed test")
    parser.add_argument(
        "--n_perm",
        "-p",
        type=int,
        default=0,
        help="Number of permutations for statistical testing (default: %(default)s)",
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
