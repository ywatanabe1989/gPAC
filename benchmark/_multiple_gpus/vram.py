#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-03 18:50:23 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/performance/multiple_gpus/vram.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/multiple_gpus/vram.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests VRAM scaling capabilities of multi-GPU setup
  - Finds maximum dataset size for single GPU
  - Tests proportionally larger datasets on multi-GPU
  - Demonstrates memory capacity benefits

Dependencies:
  - scripts:
    - ./utils.py
  - packages:
    - torch, numpy, scitex, gpac
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - ./vram_scaling.gif
    - ./vram_results.yaml
"""

"""Imports"""
import argparse

import torch
from utils import (clear_gpu_cache, create_comparison_plot, create_pac_model,
                   create_test_config, generate_test_data,
                   get_gpu_memory_capacity, measure_execution_time,
                   print_config, print_gpu_info, print_test_header)

"""Functions & Classes"""
def find_max_batch_size(config: dict, max_attempts: int = 15) -> int:
    """
    Find maximum batch size that fits in single GPU memory.
    Uses binary search approach for efficiency.
    """
    print("🔍 Finding maximum single GPU batch size...")

    # Start with exponential search to find upper bound
    base_batch = 8
    max_working = base_batch

    for attempt in range(max_attempts):
        test_batch = base_batch * (2**attempt)

        try:
            clear_gpu_cache()
            pac_single = create_pac_model(config, multi_gpu=False)
            test_data = generate_test_data(
                test_batch, config["n_channels"], config["seq_len"]
            )

            print(f"   Testing batch size {test_batch}...")
            _, success, memory_used = measure_execution_time(
                pac_single, test_data
            )

            if success:
                max_working = test_batch
                print(
                    f"   ✅ Batch {test_batch}: SUCCESS ({memory_used:.2f} GiB)"
                )
            else:
                print(f"   ❌ Batch {test_batch}: FAILED")
                break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ Batch {test_batch}: OUT OF MEMORY")
                break
            else:
                raise e

    print(f"🎯 Maximum single GPU batch size: {max_working}")
    return max_working


def run_vram_scaling_test(config: dict) -> dict:
    """
    Test VRAM scaling by comparing maximum single GPU workload
    vs proportionally scaled multi-GPU workload.
    """
    print_test_header(
        "VRAM Scaling Test", "Memory capacity scaling with multiple GPUs"
    )

    # Find single GPU limit
    single_max_batch = find_max_batch_size(config)
    gpu_count = torch.cuda.device_count()

    # Calculate scaled batch sizes
    scaling_factors = [1, 2, 3, 4] if gpu_count >= 4 else [1, 2]
    if gpu_count > 4:
        scaling_factors.extend([6, 8])

    results = {
        "scaling_factors": [],
        "batch_sizes": [],
        "single_times": [],
        "multi_times": [],
        "single_memory": [],
        "multi_memory": [],
        "single_success": [],
        "multi_success": [],
        "single_max_batch": single_max_batch,
    }

    for factor in scaling_factors:
        scaled_batch = single_max_batch * factor

        print(
            f"\n--- Testing scaling factor {factor}x (batch size: {scaled_batch}) ---"
        )

        # Test 1: Single GPU with base workload
        if factor == 1:
            # Only test single GPU at its maximum capacity
            print(f"Single GPU max capacity test:")
            data_single = generate_test_data(
                single_max_batch, config["n_channels"], config["seq_len"]
            )
            pac_single = create_pac_model(config, multi_gpu=False)
            single_time, single_success, single_mem = measure_execution_time(
                pac_single, data_single
            )

            if single_success:
                print(
                    f"✅ Single GPU: {single_time:.2f}s, {single_mem:.2f} GiB"
                )
            else:
                print(f"❌ Single GPU: FAILED")
        else:
            # Single GPU cannot handle scaled workloads
            single_time, single_success, single_mem = (
                float("inf"),
                False,
                float("inf"),
            )
            print(
                f"❌ Single GPU: Cannot handle {factor}x workload (would exceed VRAM)"
            )

        # Test 2: Multi-GPU with scaled workload
        print(f"Multi-GPU {factor}x scaling test:")
        data_multi = generate_test_data(
            scaled_batch, config["n_channels"], config["seq_len"]
        )
        pac_multi = create_pac_model(config, multi_gpu=True)

        print(f"Multi-GPU: Using {gpu_count} GPUs: {pac_multi.device_ids}")
        print(f"Data shape: {data_multi.shape}")

        multi_time, multi_success, multi_mem = measure_execution_time(
            pac_multi, data_multi
        )

        if multi_success:
            print(f"✅ Multi-GPU: {multi_time:.2f}s, {multi_mem:.2f} GiB")

            # Calculate throughput metrics
            if factor == 1 and single_success:
                speedup = single_time / multi_time
                print(f"📈 Speedup at base workload: {speedup:.2f}x")

            single_throughput = (
                single_max_batch / single_time if single_success else 0
            )
            multi_throughput = scaled_batch / multi_time
            print(
                f"📊 Throughput: Multi-GPU {multi_throughput:.1f} vs Single {single_throughput:.1f} samples/sec"
            )

        else:
            print(f"❌ Multi-GPU: FAILED")

        # Store results
        results["scaling_factors"].append(factor)
        results["batch_sizes"].append(scaled_batch)
        results["single_times"].append(single_time)
        results["multi_times"].append(multi_time)
        results["single_memory"].append(single_mem)
        results["multi_memory"].append(multi_mem)
        results["single_success"].append(single_success)
        results["multi_success"].append(multi_success)

        # Stop if multi-GPU fails (hit total VRAM limit)
        if not multi_success:
            print("⚠️  Multi-GPU failed - reached total VRAM limit")
            break

    return results


def print_vram_summary(results: dict):
    """Print VRAM scaling test summary."""
    gpu_count = torch.cuda.device_count()
    gpu_memory = get_gpu_memory_capacity()
    single_max = results["single_max_batch"]

    print(f"\n{'='*60}")
    print(f"📋 VRAM Scaling Summary")
    print(f"{'='*60}")
    print(f"Single GPU Memory:     {gpu_memory:.1f} GiB")
    print(f"Total Multi-GPU Memory: {gpu_memory * gpu_count:.1f} GiB")
    print(f"Single GPU Max Batch:   {single_max}")

    # Find maximum successful multi-GPU scaling
    max_scaling = 0
    max_batch = 0
    for i, success in enumerate(results["multi_success"]):
        if success:
            max_scaling = results["scaling_factors"][i]
            max_batch = results["batch_sizes"][i]

    if max_scaling > 0:
        print(f"Max Multi-GPU Scaling:  {max_scaling}x (batch {max_batch})")
        print(
            f"Memory Utilization:     {max_scaling/gpu_count:.1%} of theoretical maximum"
        )

        # Calculate effective throughput improvement
        base_idx = 0  # First result should be 1x scaling
        if (
            results["single_success"][base_idx]
            and results["multi_success"][base_idx]
        ):
            base_single_throughput = (
                single_max / results["single_times"][base_idx]
            )
            max_idx = results["scaling_factors"].index(max_scaling)
            max_multi_throughput = (
                results["batch_sizes"][max_idx]
                / results["multi_times"][max_idx]
            )

            print(
                f"Throughput Improvement: {max_multi_throughput/base_single_throughput:.1f}x"
            )
    else:
        print("❌ No successful multi-GPU scaling achieved")


def main(args):
    """Main VRAM scaling test function."""
    print_gpu_info()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1

    if torch.cuda.device_count() == 1:
        print(
            "⚠️  Only 1 GPU available - VRAM scaling test requires multiple GPUs"
        )
        return 1

    # Create test configuration
    config = create_test_config()
    config["n_perm"] = args.n_perm
    print_config(config)

    # Run VRAM scaling test
    results = run_vram_scaling_test(config)

    # Print summary
    print_vram_summary(results)

    # Create visualization
    fig = create_comparison_plot(results, "Multi-GPU VRAM Scaling Test")

    # Save results
    import scitex

    scitex.io.save(results, "vram_results.yaml")
    scitex.io.save(fig, "vram_scaling.gif")

    print(f"\n✅ Results saved to ./")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import scitex

    script_mode = scitex.gen.is_script()
    parser = argparse.ArgumentParser(description="Multi-GPU VRAM scaling test")
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
    import scitex

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
