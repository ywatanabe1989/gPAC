#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-05 13:50:16 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/performance/multiple_gpus/throughput.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/multiple_gpus/throughput.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests throughput scaling: process 4x more data in reasonable time
  - Demonstrates the key multi-GPU value proposition
  - Compares throughput (samples/second) rather than just speedup
  - Shows real-world benefit: handle larger datasets efficiently

Dependencies:
  - scripts:
    - ./utils.py
  - packages:
    - torch, numpy, mngs, gpac
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - throughput_scaling.gif
    - throughput_results.yaml
"""

"""Imports"""
import argparse

import mngs
import torch
from utils import (calculate_throughput, create_pac_model, create_test_config,
                   create_throughput_comparison_plot, generate_test_data,
                   measure_execution_time, print_config, print_gpu_info,
                   print_test_header)

"""Functions & Classes"""
def find_optimal_single_gpu_batch(config: dict) -> int:
    """Find optimal batch size for single GPU (not maximum, but efficient)."""
    print("🔍 Finding optimal single GPU batch size...")

    # Test various batch sizes to find efficiency sweet spot
    test_batches = [8, 16, 24, 32, 48, 64, 96, 128]
    best_throughput = 0
    best_batch = 8

    print("   Testing batch sizes for optimal throughput...")

    for batch_size in test_batches:
        try:
            pac_single = create_pac_model(config, multi_gpu=False)
            test_data = generate_test_data(
                batch_size, config["n_channels"], config["seq_len"]
            )

            execution_time, success, memory_used = measure_execution_time(
                pac_single, test_data
            )

            if success:
                throughput = calculate_throughput(batch_size, execution_time)
                print(
                    f"   Batch {batch_size}: {throughput:.1f} samples/sec ({execution_time:.2f}s)"
                )

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch = batch_size
            else:
                print(f"   Batch {batch_size}: FAILED")
                break

        except Exception as e:
            print(f"   Batch {batch_size}: ERROR - {str(e)}")
            break

    print(
        f"🎯 Optimal single GPU batch: {best_batch} ({best_throughput:.1f} samples/sec)"
    )
    return best_batch


def run_throughput_scaling_test(config: dict) -> dict:
    """
    Test throughput scaling across different GPU configurations (1, 2, 3, 4 GPUs)
    and various batch sizes to measure samples per second.
    """
    print_test_header(
        "Throughput Scaling Test",
        "Multi-GPU throughput across different configurations",
    )

    # Find optimal single GPU batch size
    optimal_batch = find_optimal_single_gpu_batch(config)
    total_gpu_count = torch.cuda.device_count()

    # Test different GPU configurations
    gpu_configs = (
        [1, 2, 3, 4] if total_gpu_count >= 4 else [1, min(2, total_gpu_count)]
    )

    # Define batch size scenarios for each GPU configuration
    batch_scenarios = [
        {"name": "Small", "batch": optimal_batch // 2},
        {"name": "Optimal", "batch": optimal_batch},
        {"name": "2x", "batch": optimal_batch * 2},
        {"name": "4x", "batch": optimal_batch * 4},
    ]

    results = {
        "gpu_configs": [],
        "batch_sizes": [],
        "execution_times": [],
        "throughputs": [],
        "memory_usage": [],
        "success": [],
        "scenario_names": [],
    }

    for gpu_config in gpu_configs:
        print(f"\n{'='*60}")
        print(f"🔧 Testing with {gpu_config} GPU(s)")
        print(f"{'='*60}")

        # Set up device configuration for this test
        if gpu_config == 1:
            use_multi_gpu = False
        else:
            use_multi_gpu = True
            # Limit visible GPUs for this configuration
            import os

            original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            device_list = ",".join(str(i) for i in range(gpu_config))
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list

        for scenario in batch_scenarios:
            batch_size = scenario["batch"]
            scenario_name = f"{scenario['name']} ({batch_size})"

            print(f"\n--- {gpu_config} GPU(s): {scenario_name} ---")

            try:
                # Generate test data
                data = generate_test_data(
                    batch_size, config["n_channels"], config["seq_len"]
                )
                print(f"Data shape: {data.shape}")

                # Create PAC model with current GPU configuration
                pac_model = create_pac_model(config, multi_gpu=use_multi_gpu)

                # Measure execution with multiple runs for stability
                execution_times = []
                for run in range(3):  # Multiple runs for stable measurements
                    exec_time, success, memory_used = measure_execution_time(
                        pac_model, data
                    )
                    if success:
                        execution_times.append(exec_time)
                
                if execution_times:
                    # Use median time for stability
                    execution_time = np.median(execution_times)
                    success = True
                    throughput = calculate_throughput(
                        batch_size, execution_time
                    )
                    print(
                        f"✅ {gpu_config} GPU(s): {execution_time:.2f}s "
                        f"(median of {len(execution_times)} runs), "
                        f"{throughput:.1f} samples/sec, {memory_used:.1f} MB"
                    )

                    # Store results
                    results["gpu_configs"].append(gpu_config)
                    results["batch_sizes"].append(batch_size)
                    results["execution_times"].append(execution_time)
                    results["throughputs"].append(throughput)
                    results["memory_usage"].append(memory_used)
                    results["success"].append(True)
                    results["scenario_names"].append(scenario_name)

                else:
                    print(f"❌ {gpu_config} GPU(s): FAILED")
                    results["gpu_configs"].append(gpu_config)
                    results["batch_sizes"].append(batch_size)
                    results["execution_times"].append(float("inf"))
                    results["throughputs"].append(0)
                    results["memory_usage"].append(float("inf"))
                    results["success"].append(False)
                    results["scenario_names"].append(scenario_name)

            except Exception as e:
                print(f"❌ {gpu_config} GPU(s): ERROR - {str(e)}")
                results["gpu_configs"].append(gpu_config)
                results["batch_sizes"].append(batch_size)
                results["execution_times"].append(float("inf"))
                results["throughputs"].append(0)
                results["memory_usage"].append(float("inf"))
                results["success"].append(False)
                results["scenario_names"].append(scenario_name)

        # Restore original CUDA_VISIBLE_DEVICES if we changed it
        if gpu_config > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices

    return results


def print_throughput_summary(results: dict):
    """Print throughput scaling summary."""
    print(f"\n{'='*60}")
    print(f"📊 Throughput Scaling Summary")
    print(f"{'='*60}")

    # Group results by GPU configuration
    gpu_configs = sorted(set(results["gpu_configs"]))

    for gpu_config in gpu_configs:
        # Find results for this GPU configuration
        config_results = []
        for i, gc in enumerate(results["gpu_configs"]):
            if gc == gpu_config and results["success"][i]:
                config_results.append(
                    {
                        "batch_size": results["batch_sizes"][i],
                        "throughput": results["throughputs"][i],
                        "scenario": results["scenario_names"][i],
                    }
                )

        if config_results:
            max_throughput = max(r["throughput"] for r in config_results)
            best_result = max(config_results, key=lambda x: x["throughput"])

            print(f"\n🔧 {gpu_config} GPU(s):")
            print(f"   Best throughput: {max_throughput:.1f} samples/sec")
            print(f"   Best scenario: {best_result['scenario']}")

            # Show all results for this GPU config
            for result in config_results:
                print(
                    f"     {result['scenario']}: {result['throughput']:.1f} samples/sec"
                )

    # Calculate scaling efficiency
    if len(gpu_configs) > 1:
        single_gpu_best = 0
        multi_gpu_best = 0

        for i, gc in enumerate(results["gpu_configs"]):
            if results["success"][i]:
                if gc == 1:
                    single_gpu_best = max(
                        single_gpu_best, results["throughputs"][i]
                    )
                else:
                    multi_gpu_best = max(
                        multi_gpu_best, results["throughputs"][i]
                    )

        if single_gpu_best > 0:
            scaling_factor = multi_gpu_best / single_gpu_best
            print(f"\n📈 Overall Scaling:")
            print(f"   Single GPU peak: {single_gpu_best:.1f} samples/sec")
            print(f"   Multi-GPU peak: {multi_gpu_best:.1f} samples/sec")
            print(f"   Scaling factor: {scaling_factor:.2f}x")


def main(args):
    """Main throughput scaling test function."""
    print_gpu_info()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1

    # Can test with any number of GPUs (including 1)

    # Create test configuration
    config = create_test_config()
    config["n_perm"] = args.n_perm
    print_config(config)

    # Run throughput scaling test
    results = run_throughput_scaling_test(config)

    # Print summary
    print_throughput_summary(results)

    # Create visualization with new throughput comparison plot
    fig = create_throughput_comparison_plot(
        results,
        "Multi-GPU Throughput Scaling: Batch Size vs Samples/sec",
    )

    # Save results
    mngs.io.save(results, "throughput_results.yaml")
    mngs.io.save(fig, "throughput_scaling.gif")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Multi-GPU throughput scaling test"
    )
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
