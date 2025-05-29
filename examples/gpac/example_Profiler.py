#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 04:45:00 (ywatanabe)"
# File: ./examples/gpac/example_Profiler.py

"""
Functionalities:
  - Demonstrates the _Profiler module usage for performance monitoring
  - Shows CPU, RAM, and GPU resource tracking
  - Compares performance with and without profiling overhead
  - Demonstrates nested profiling contexts
  - Shows how to generate performance reports

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch
    - numpy
    - matplotlib
    - mngs
    
IO:
  - input-files:
    - None
    
  - output-files:
    - ./example_Profiler_out/profiling_results.png
    - ./example_Profiler_out/performance_report.txt
"""

"""Imports"""
import argparse
import sys
import time
import numpy as np
import torch

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""

def heavy_computation(size=1000, iterations=10, use_gpu=True):
    """Simulate a computationally intensive task."""
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Matrix operations
    result = torch.randn(size, size, device=device)
    for _ in range(iterations):
        result = torch.matmul(result, result)
        result = torch.nn.functional.relu(result)
    
    return result


def main(args):
    """Run Profiler demonstration."""
    import mngs
    from gpac._Profiler import create_profiler, Profiler
    from gpac import PAC, BandPassFilter
    
    mngs.str.printc("🚀 Profiler Demonstration", c="green")
    mngs.str.printc("=" * 60, c="green")
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mngs.str.printc(f"Using device: {device}", c="cyan")
    
    # Part 1: Basic profiling
    mngs.str.printc("\n🎯 Part 1: Basic Profiling", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    # Create profiler
    profiler = create_profiler(enable_gpu=torch.cuda.is_available())
    
    # Profile a simple computation
    with profiler.profile("Simple Matrix Multiplication"):
        result = heavy_computation(size=500, iterations=5)
    
    # Profile PAC computation
    mngs.str.printc("\n🔄 Profiling PAC computation...", c="blue")
    
    # Generate test signal
    fs = 1000
    duration = 2
    signal = torch.randn(1, 1, int(fs * duration)).to(device)
    
    # Initialize PAC
    pac = PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_start_hz=4,
        pha_end_hz=10,
        pha_n_bands=3,
        amp_start_hz=30,
        amp_end_hz=100,
        amp_n_bands=5
    ).to(device)
    
    # Profile PAC computation
    with profiler.profile("PAC Computation"):
        pac_results = pac(signal)
    
    # Part 2: Nested profiling
    mngs.str.printc("\n🎯 Part 2: Nested Profiling", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    with profiler.profile("Complete Pipeline"):
        # Signal generation
        with profiler.profile("Signal Generation"):
            test_signal = torch.randn(10, 1, 1000).to(device)
        
        # Filtering
        with profiler.profile("Bandpass Filtering"):
            filter = BandPassFilter(
                seq_len=1000,
                fs=256,
                pha_start_hz=4,
                pha_end_hz=10,
                pha_n_bands=2,
                amp_start_hz=30,
                amp_end_hz=80,
                amp_n_bands=3
            ).to(device)
            
            filtered = filter(test_signal)
        
        # Analysis
        with profiler.profile("Signal Analysis"):
            mean_power = (filtered ** 2).mean()
            peak_freq = torch.fft.rfft(filtered).abs().argmax()
    
    # Part 3: Comparing with/without profiler overhead
    mngs.str.printc("\n🎯 Part 3: Profiler Overhead Analysis", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    # Without profiler
    start = time.time()
    for _ in range(10):
        _ = heavy_computation(size=200, iterations=2)
    time_without = time.time() - start
    
    # With profiler
    start = time.time()
    for i in range(10):
        with profiler.profile(f"Iteration {i}"):
            _ = heavy_computation(size=200, iterations=2)
    time_with = time.time() - start
    
    overhead_percent = ((time_with - time_without) / time_without) * 100
    
    mngs.str.printc(f"\nTime without profiler: {time_without:.3f}s", c="cyan")
    mngs.str.printc(f"Time with profiler: {time_with:.3f}s", c="cyan")
    mngs.str.printc(f"Profiler overhead: {overhead_percent:.1f}%", c="yellow")
    
    # Generate summary
    mngs.str.printc("\n📊 Profiling Summary", c="green")
    mngs.str.printc("=" * 60, c="green")
    profiler.print_summary()
    
    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get results
    results = profiler.results
    
    # Plot timing distribution
    ax = axes[0, 0]
    names = [r.name for r in results[:10]]  # Top 10
    times = [r.duration for r in results[:10]]
    ax.barh(names, times)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Top 10 Operations by Time")
    ax.grid(True, alpha=0.3)
    
    # Plot CPU usage
    ax = axes[0, 1]
    cpu_usage = [r.cpu_percent for r in results[:10]]
    ax.barh(names, cpu_usage, color='orange')
    ax.set_xlabel("CPU Usage (%)")
    ax.set_title("CPU Usage by Operation")
    ax.grid(True, alpha=0.3)
    
    # Plot memory usage
    ax = axes[1, 0]
    ram_usage = [r.ram_used_gb for r in results[:10]]
    ax.barh(names, ram_usage, color='green')
    ax.set_xlabel("RAM Usage (GB)")
    ax.set_title("Memory Usage by Operation")
    ax.grid(True, alpha=0.3)
    
    # Plot GPU usage if available
    ax = axes[1, 1]
    if torch.cuda.is_available() and results[0].gpu_memory_used_gb is not None:
        gpu_usage = [r.gpu_memory_used_gb for r in results[:10] if r.gpu_memory_used_gb]
        ax.barh(names[:len(gpu_usage)], gpu_usage, color='red')
        ax.set_xlabel("GPU Memory (GB)")
        ax.set_title("GPU Memory Usage by Operation")
    else:
        ax.text(0.5, 0.5, "GPU not available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("GPU Memory Usage")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    spath = "profiling_results.png"
    mngs.io.save(fig, spath)
    
    # Save detailed report
    report_lines = [
        "=== gPAC Profiler Report ===",
        f"Total operations profiled: {len(results)}",
        f"Total time: {sum(r.duration for r in results):.3f}s",
        f"Device: {device}",
        f"Profiler overhead: {overhead_percent:.1f}%",
        "",
        "Top 5 Slowest Operations:",
        "-" * 40
    ]
    
    for i, result in enumerate(sorted(results, key=lambda x: x.duration, reverse=True)[:5]):
        report_lines.append(f"\n{i+1}. {result}")
    
    report_text = "\n".join(report_lines)
    mngs.io.save(report_text, "performance_report.txt")
    
    mngs.str.printc("\n✅ Profiler demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to:", c="green")
    mngs.str.printc(f"   - {spath}", c="cyan")
    mngs.str.printc(f"   - performance_report.txt", c="cyan")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Profiler demonstration")
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt
    
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mngs
    
    args = parse_args()
    
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix=None,
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