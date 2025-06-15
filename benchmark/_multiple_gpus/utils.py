#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-05 14:37:40 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/performance/multiple_gpus/utils.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/multiple_gpus/utils.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Shared utilities for multi-GPU performance testing
  - GPU memory monitoring and management
  - Data generation and configuration
  - Common measurement and visualization functions

Dependencies:
  - packages:
    - torch, numpy, matplotlib, scitex, gpac
"""

"""Imports"""
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append("../../../src")
import scitex
from gpac import PAC

"""GPU Memory Utilities"""


def get_gpu_memory_info() -> float:
    """Get current GPU memory usage in GiB."""
    if torch.cuda.is_available():
        # Use memory_reserved() which shows actual GPU memory reserved by PyTorch
        # This gives a better indication of actual memory consumption
        memory_bytes = torch.cuda.memory_reserved()
        memory_gib = memory_bytes / (1024**3)
        return memory_gib
    return 0.0


def clear_gpu_cache():
    """Clear GPU cache and synchronize."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_capacity() -> float:
    """Get total GPU memory capacity in GiB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


"""Data Generation"""


def create_test_config() -> Dict[str, Any]:
    """Create standard test configuration."""
    batch_size = 8
    n_channels = 16
    fs = 512
    seq_sec = 8.0  # Sequence duration in seconds (more intuitive)
    n_perm = 0  # Start with no permutations for faster testing
    seq_len = int(fs * seq_sec)

    return {
        "batch_size": batch_size,
        "n_channels": n_channels,
        "seq_sec": seq_sec,  # Human-readable duration
        "n_perm": n_perm,
        "seq_len": seq_len,  # Computed length for internal use
        "fs": fs,
    }


def generate_test_data(
    batch_size: int, n_channels: int, seq_len: int
) -> torch.Tensor:
    """Generate synthetic test data for PAC analysis."""
    return torch.randn(batch_size, n_channels, seq_len)


"""Performance Measurement"""


def measure_execution_time(
    pac_model: PAC, data: torch.Tensor
) -> Tuple[float, bool, float]:
    """Measure execution time, success status, and memory usage."""
    try:
        clear_gpu_cache()

        # Reset peak memory stats for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Get baseline memory before computation
            baseline_mem = torch.cuda.memory_allocated() / (1024**3)

        # Ensure model and data are on GPU if CUDA is available
        if torch.cuda.is_available():
            # Check if model is already on GPU
            if hasattr(pac_model, "device"):
                print(f"   DEBUG: PAC model device: {pac_model.device}")
            
            # Move data to the same device as the model
            if hasattr(pac_model, "device"):
                data = data.to(pac_model.device)
            else:
                data = data.cuda()

            # Force synchronization
            torch.cuda.synchronize()
            
            # Get memory after moving data to GPU
            data_on_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            print(f"   DEBUG: Memory after data transfer: {data_on_gpu_mem:.3f} GiB")

        start_time = time.time()
        result = pac_model(data)

        # Synchronize for accurate timing and memory measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Measure peak memory during computation
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            # Calculate memory used during computation (excluding initial data transfer)
            memory_used = peak_mem - baseline_mem
            
            # Debug print to diagnose 0.0 issue
            if memory_used < 0.01:  # Less than 10 MB
                print(f"   DEBUG: baseline_mem={baseline_mem:.3f} GiB, peak_mem={peak_mem:.3f} GiB, current_mem={current_mem:.3f} GiB")
        else:
            memory_used = 0.0

        execution_time = time.time() - start_time

        return execution_time, True, memory_used

    except Exception as e:
        print(f"Execution failed: {str(e)}")
        return float("inf"), False, float("inf")


def create_pac_model(config: Dict[str, Any], multi_gpu: bool = False) -> PAC:
    """Create PAC model with standard configuration."""
    # Set device_ids based on multi_gpu flag
    if multi_gpu and torch.cuda.is_available():
        device_ids = "all"  # Use all available GPUs
    else:
        device_ids = [0] if torch.cuda.is_available() else []
    
    return PAC(
        seq_len=config["seq_len"],
        fs=config["fs"],
        pha_n_bands=config.get("pha_n_bands", 10),
        amp_n_bands=config.get("amp_n_bands", 10),
        n_perm=config["n_perm"],
        device_ids=device_ids,
        enable_caching=True,
        compile_mode=False,  # Disable compilation for debugging
    )


"""Metrics Calculation"""


def calculate_speedup(single_time: float, multi_time: float) -> float:
    """Calculate speedup ratio."""
    return single_time / multi_time if multi_time > 0 else 0.0


def calculate_efficiency(speedup: float, gpu_count: int) -> float:
    """Calculate efficiency (speedup per GPU)."""
    return speedup / gpu_count if gpu_count > 0 else 0.0


def calculate_throughput(batch_size: int, execution_time: float) -> float:
    """Calculate throughput in samples per second."""
    return batch_size / execution_time if execution_time > 0 else 0.0


"""Visualization"""


def create_comparison_plot(
    results: Dict[str, List],
    title: str,
):
    """Create comparison visualization plot."""

    fig, (ax1, ax2) = scitex.plt.subplots(1, 2, figsize=(12, 5))

    # Speedup/Efficiency plot
    valid_indices = [
        i
        for i, (s_success, m_success) in enumerate(
            zip(results["single_success"], results["multi_success"])
        )
        if s_success and m_success
    ]

    if valid_indices:
        batch_sizes = [results["batch_sizes"][i] for i in valid_indices]
        speedups = [
            results["single_times"][i] / results["multi_times"][i]
            for i in valid_indices
        ]

        ax1.plot(batch_sizes, speedups, "o-", label="Speedup", color="blue")
        ax1.axhline(
            y=1, color="r", linestyle="--", alpha=0.5, label="No speedup"
        )
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Speedup (x)")
        ax1.set_title("Multi-GPU Speedup")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Throughput comparison
    single_indices = [
        i for i, success in enumerate(results["single_success"]) if success
    ]
    multi_indices = [
        i for i, success in enumerate(results["multi_success"]) if success
    ]

    if single_indices:
        single_batches = [results["batch_sizes"][i] for i in single_indices]
        single_throughput = [
            results["batch_sizes"][i] / results["single_times"][i]
            for i in single_indices
        ]
        ax2.plot(
            single_batches,
            single_throughput,
            "o-",
            label="Single GPU",
            color="blue",
        )

    if multi_indices:
        multi_batches = [results["batch_sizes"][i] for i in multi_indices]
        multi_throughput = [
            results["batch_sizes"][i] / results["multi_times"][i]
            for i in multi_indices
        ]
        ax2.plot(
            multi_batches,
            multi_throughput,
            "s-",
            label="Multi-GPU",
            color="red",
        )

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (samples/sec)")
    ax2.set_title("Processing Throughput")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def create_speed_comparison_plot(results: Dict[str, List], title: str):
    """Create speed comparison plot with x=calculation_time, y=n_samples, hue=device_config."""

    # Prepare data for plotting
    plot_data = []

    # Get valid results
    for i, (
        batch_size,
        single_time,
        multi_time,
        single_success,
        multi_success,
    ) in enumerate(
        zip(
            results["batch_sizes"],
            results["single_times"],
            results["multi_times"],
            results["single_success"],
            results["multi_success"],
        )
    ):
        if single_success:
            plot_data.append(
                {
                    "calculation_time": single_time,
                    "n_samples": batch_size,
                    "device_config": "1 GPU",
                    "successful": True,
                }
            )

        if multi_success:
            gpu_count = torch.cuda.device_count()
            plot_data.append(
                {
                    "calculation_time": multi_time,
                    "n_samples": batch_size,
                    "device_config": f"{gpu_count} GPUs",
                    "successful": True,
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    if df.empty:
        print("⚠️  No data available for plotting")
        return None

    # Create the plot
    fig, ax = scitex.plt.subplots(1, 1, figsize=(10, 6))

    # Create scatter plot with different colors for each device configuration
    colors = {
        "1 GPU": "blue",
        "2 GPUs": "orange",
        "3 GPUs": "green",
        "4 GPUs": "red",
        "8 GPUs": "purple",
    }

    for device_config in df["device_config"].unique():
        subset = df[df["device_config"] == device_config]
        color = colors.get(device_config, "black")
        ax.scatter(
            subset["calculation_time"],
            subset["n_samples"],
            label=device_config,
            color=color,
            s=60,
            alpha=0.7,
        )

        # Add trend line
        if len(subset) > 1:
            z = np.polyfit(subset["calculation_time"], subset["n_samples"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                subset["calculation_time"].min(),
                subset["calculation_time"].max(),
                100,
            )
            ax.plot(
                x_trend, p(x_trend), color=color, linestyle="--", alpha=0.5
            )

    ax.set_xlabel("Actual Calculation Time (s)")
    ax.set_ylabel("Number of Samples (batch size)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_throughput_comparison_plot(results: Dict[str, List], title: str):
    """Create throughput comparison plot with x=batch_size, y=samples/sec, hue=device_config."""

    # Prepare data for plotting
    plot_data = []

    for i, (
        batch_size,
        single_time,
        multi_time,
        single_success,
        multi_success,
    ) in enumerate(
        zip(
            results["batch_sizes"],
            results["single_times"],
            results["multi_times"],
            results["single_success"],
            results["multi_success"],
        )
    ):
        if single_success:
            throughput = batch_size / single_time
            plot_data.append(
                {
                    "batch_size": batch_size,
                    "samples_per_sec": throughput,
                    "device_config": "1 GPU",
                }
            )

        if multi_success:
            gpu_count = torch.cuda.device_count()
            throughput = batch_size / multi_time
            plot_data.append(
                {
                    "batch_size": batch_size,
                    "samples_per_sec": throughput,
                    "device_config": f"{gpu_count} GPUs",
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    if df.empty:
        print("⚠️  No data available for plotting")
        return None

    # Create the plot
    fig, ax = scitex.plt.subplots(1, 1, figsize=(10, 6))

    # Create line plot with different colors for each device configuration
    colors = {
        "1 GPU": "blue",
        "2 GPUs": "orange",
        "3 GPUs": "green",
        "4 GPUs": "red",
        "8 GPUs": "purple",
    }

    for device_config in df["device_config"].unique():
        subset = df[df["device_config"] == device_config].sort_values(
            "batch_size"
        )
        color = colors.get(device_config, "black")
        ax.plot(
            subset["batch_size"],
            subset["samples_per_sec"],
            "o-",
            label=device_config,
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Samples per Second")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


"""Reporting"""


def print_test_header(test_name: str, description: str = ""):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"🚀 {test_name}")
    if description:
        print(f"📋 {description}")
    print(f"{'='*60}")


def print_gpu_info():
    """Print GPU configuration information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = get_gpu_memory_capacity()
        print(f"🔧 GPU Count: {gpu_count}")
        print(f"🔧 GPU Memory: {gpu_memory:.1f} GiB per GPU")
        print(f"🔧 Total VRAM: {gpu_memory * gpu_count:.1f} GiB")
        # Show current GPU devices
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
    else:
        print("❌ CUDA not available")


def print_config(config: Dict[str, Any]):
    """Print test configuration."""
    print(f"\n📋 Test Configuration:")
    for key, value in config.items():
        if key == "seq_sec":
            print(f"   {key}: {value}s (sequence duration)")
        elif key == "seq_len":
            print(f"   {key}: {value} samples (computed from seq_sec * fs)")
        else:
            print(f"   {key}: {value}")


def print_results_summary(
    single_time: float, multi_time: float, batch_size: int, gpu_count: int
):
    """Print formatted results summary."""
    if single_time == float("inf") or multi_time == float("inf"):
        print("❌ Test failed")
        return

    speedup = calculate_speedup(single_time, multi_time)
    efficiency = calculate_efficiency(speedup, gpu_count)
    single_throughput = calculate_throughput(batch_size, single_time)
    multi_throughput = calculate_throughput(batch_size, multi_time)

    print(f"\n📊 Results Summary:")
    print(f"   Single GPU Time:  {single_time:.2f}s")
    print(f"   Multi-GPU Time:   {multi_time:.2f}s")
    print(f"   Speedup:          {speedup:.2f}x")
    print(f"   Efficiency:       {efficiency:.1%} per GPU")
    print(f"   Single Throughput: {single_throughput:.1f} samples/sec")
    print(f"   Multi Throughput:  {multi_throughput:.1f} samples/sec")

# EOF
