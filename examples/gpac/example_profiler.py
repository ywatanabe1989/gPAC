#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 10:00:00 (ywatanabe)"
# File: example_profiler.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_profiler.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Using gPAC's Profiler for performance monitoring
  - Tracking GPU memory usage (VRAM)
  - Measuring computation time for different batch sizes
  - Comparing performance across different configurations

Dependencies:
  - scripts: None
  - packages: gpac, torch, numpy, matplotlib

IO:
  - input-files: None
  - output-files: ./example_profiler_out/profiling_results.png, ./example_profiler_out/profiling_data.yaml, ./example_profiler_out/performance_report.txt
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""


def visualize_profiling_results(results):
    """Visualize profiling results."""
    import matplotlib.pyplot as plt

    # Create figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), facecolor="white")
    axes = axes.flatten()

    # Extract data
    batch_sizes = results["batch_sizes"]
    compute_times = results["compute_times"]
    peak_memory_mb = results["peak_memory_mb"]
    throughput = results["throughput"]

    # Plot computation time vs batch size
    ax = axes[0]
    ax.plot(batch_sizes, compute_times, "b-o", linewidth=2, markersize=8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Computation Time (s)")
    ax.set_title("Computation Time vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Plot memory usage vs batch size
    ax = axes[1]
    ax.plot(batch_sizes, peak_memory_mb, "r-s", linewidth=2, markersize=8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak Memory Usage (MB)")
    ax.set_title("GPU Memory Usage vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Plot throughput vs batch size
    ax = axes[2]
    ax.plot(batch_sizes, throughput, "g-^", linewidth=2, markersize=8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Processing Throughput vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Plot efficiency (throughput per MB)
    ax = axes[3]
    efficiency = np.array(throughput) / np.array(peak_memory_mb)
    ax.plot(batch_sizes, efficiency, "m-d", linewidth=2, markersize=8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Efficiency (samples/sec/MB)")
    ax.set_title("Memory Efficiency vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    return fig


def profile_pac_computation(batch_size, signal_length=2500, n_channels=1):
    """Profile PAC computation for a given batch size."""
    from gpac import PAC, SyntheticDataGenerator, Profiler

    # Parameters
    fs = 250
    phase_freqs = np.arange(4, 20, 2)
    amp_freqs = np.arange(30, 100, 5)

    # Generate synthetic data
    generator = SyntheticDataGenerator(fs=fs, duration_sec=signal_length / fs)
    signals = []
    for _ in range(batch_size):
        signal = generator.generate_pac_signal(
            phase_freq=6.0, amp_freq=60.0, coupling_strength=0.7, noise_level=0.1
        )
        signals.append(signal)

    # Convert to tensor
    signals_tensor = torch.stack([torch.from_numpy(s).float() for s in signals])
    if n_channels == 1:
        signals_tensor = signals_tensor.unsqueeze(1)  # Add channel dimension

    # Initialize PAC with profiler
    pac = PAC(
        low_freq_range=phase_freqs,
        high_freq_range=amp_freqs,
        low_freq_width=2.0,
        high_freq_width=20.0,
        fs=fs,
        n_jobs=1,
    )

    # Create profiler
    profiler = Profiler()

    # Warm-up GPU
    if torch.cuda.is_available():
        _ = pac(signals_tensor[:1].cuda())
        torch.cuda.synchronize()

    # Profile computation
    with profiler:
        if torch.cuda.is_available():
            signals_tensor = signals_tensor.cuda()
        pac_values = pac(signals_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Get profiling results
    profile_data = profiler.get_summary()

    return profile_data


def main(args):
    """Main example function."""
    import mngs

    # Set random seed
    mngs.gen.fix_seeds(42)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    if device == "cpu":
        batch_sizes = [1, 2, 4, 8]  # Smaller sizes for CPU

    print(f"\nProfiling PAC computation for batch sizes: {batch_sizes}")

    results = {
        "batch_sizes": [],
        "compute_times": [],
        "peak_memory_mb": [],
        "throughput": [],
        "device": device,
    }

    # Profile each batch size
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Profiling batch size: {batch_size}")

        try:
            profile_data = profile_pac_computation(batch_size)

            # Extract metrics
            total_time = profile_data["total_time"]
            peak_memory = profile_data.get("peak_memory_mb", 0)
            throughput = batch_size / total_time

            # Store results
            results["batch_sizes"].append(batch_size)
            results["compute_times"].append(total_time)
            results["peak_memory_mb"].append(peak_memory)
            results["throughput"].append(throughput)

            # Print summary
            print(f"  Total time: {total_time:.3f} s")
            print(f"  Peak memory: {peak_memory:.2f} MB")
            print(f"  Throughput: {throughput:.2f} samples/s")
            print(f"  Time per sample: {total_time/batch_size*1000:.2f} ms")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Create output directory
    sdir = mngs.io.mk_spath(__file__)

    # Visualize results if we have data
    if results["batch_sizes"]:
        print("\nVisualizing profiling results...")
        fig = visualize_profiling_results(results)
        spath = sdir / "profiling_results.png"
        mngs.io.save(fig, spath)
        print(f"  Figure saved to: {spath}")

        # Save detailed results
        spath = sdir / "profiling_data.yaml"
        mngs.io.save(results, spath)
        print(f"  Data saved to: {spath}")

        # Create performance report
        report = []
        report.append(f"gPAC Performance Profiling Report")
        report.append(f"{'='*50}")
        report.append(f"Device: {device}")
        if device == "cuda":
            report.append(f"GPU: {torch.cuda.get_device_name(0)}")
        report.append(f"")
        report.append(
            f"{'Batch Size':<12} {'Time (s)':<12} {'Memory (MB)':<12} {'Throughput':<12}"
        )
        report.append(f"{'-'*48}")

        for i in range(len(results["batch_sizes"])):
            report.append(
                f"{results['batch_sizes'][i]:<12} "
                f"{results['compute_times'][i]:<12.3f} "
                f"{results['peak_memory_mb'][i]:<12.2f} "
                f"{results['throughput'][i]:<12.2f}"
            )

        # Find optimal batch size (best throughput)
        optimal_idx = np.argmax(results["throughput"])
        report.append(
            f"\nOptimal batch size: {results['batch_sizes'][optimal_idx]} "
            f"(throughput: {results['throughput'][optimal_idx]:.2f} samples/s)"
        )

        # Save report
        report_text = "\n".join(report)
        spath = sdir / "performance_report.txt"
        mngs.io.save(report_text, spath, is_pickle=False)
        print(f"  Report saved to: {spath}")

        print("\n" + report_text)

    print("\nExample completed successfully!")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Performance profiling example for gPAC"
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
