#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 09:42:13 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/gpac/example__Hilbert.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example__Hilbert.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates Hilbert transform functionality with differentiability testing
  - Shows phase and amplitude extraction from signals
  - Validates gradient flow through the transform
  - Visualizes instantaneous frequency analysis
  - Tests batch processing and performance

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, stx

IO:
  - input-files: None (generates synthetic test signals)
  - output-files:
    - 01_hilbert_analysis.gif
    - 02_hilbert_performance.gif
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpac.core._Hilbert import Hilbert

"""Functions & Classes"""
def unwrap_phase(phase):
    """Unwrap phase to avoid discontinuities."""
    diff = torch.diff(phase)
    diff = ((diff + torch.pi) % (2 * torch.pi)) - torch.pi
    return torch.cat([phase[:1], phase[:1] + torch.cumsum(diff, dim=0)])


def demo_hilbert_analysis(args):
    """Demonstrate Hilbert transform and signal analysis."""
    import scitex as stx

    stx.str.printc("=== Demo Hilbert Transform Analysis ===", c="yellow")

    # Generate test signal
    sample_rate = args.sample_rate
    duration = args.duration
    time_steps = int(sample_rate * duration)
    t_vals = torch.linspace(0, duration, time_steps)

    # Complex signal with multiple frequency components
    signal = (
        torch.sin(2 * np.pi * 10 * t_vals)
        + 0.5 * torch.sin(2 * np.pi * 30 * t_vals)
        + 0.3 * torch.cos(2 * np.pi * 50 * t_vals)
    )
    signal += 0.1 * torch.randn_like(signal)

    # Create Hilbert transformer
    hilbert = Hilbert(seq_len=time_steps, dim=-1, fp16=False)

    # Test gradient flow
    signal_grad = signal.clone().requires_grad_(True)
    result = hilbert(signal_grad)
    phase, amplitude = result[..., 0], result[..., 1]

    loss = amplitude.mean()
    loss.backward()

    # Get analytic signal and instantaneous frequency
    analytic = hilbert.get_analytic_signal(signal)
    instantaneous_freq = (
        torch.diff(unwrap_phase(phase)) * sample_rate / (2 * np.pi)
    )

    # Create visualization
    fig, axes = stx.plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(t_vals.numpy(), signal.detach().numpy())
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_vals.numpy(), amplitude.detach().numpy())
    axes[1].set_title("Instantaneous Amplitude")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_vals.numpy(), phase.detach().numpy())
    axes[2].set_title("Instantaneous Phase")
    axes[2].set_ylabel("Phase [rad]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_vals[1:].numpy(), instantaneous_freq.detach().numpy())
    axes[3].set_title("Instantaneous Frequency")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Frequency [Hz]")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    stx.io.save(fig, "01_hilbert_analysis.gif")
    plt.close()

    # Print analysis results
    print(f"Signal shape: {signal.shape}")
    print(f"Phase shape: {phase.shape}")
    print(f"Amplitude shape: {amplitude.shape}")
    print(f"Gradient exists: {signal_grad.grad is not None}")
    print(f"Gradient norm: {signal_grad.grad.norm().item():.6f}")
    print(
        f"Real part matches original: {torch.allclose(analytic.real, signal, atol=1e-5)}"
    )
    print(f"Amplitude is positive: {(amplitude >= 0).all()}")


def demo_batch_processing_and_performance(args):
    """Demonstrate batch processing and performance metrics."""
    import scitex as stx

    stx.str.printc("=== Demo Batch Processing & Performance ===", c="yellow")

    # Test batch processing
    sample_rate = args.sample_rate
    duration = 1.0  # Shorter for performance test
    time_steps = int(sample_rate * duration)

    # Create batch of signals
    batch_size = 10
    signal = torch.randn(batch_size, time_steps)

    hilbert = Hilbert(seq_len=time_steps)

    # Process batch
    batch_result = hilbert(signal)

    print(f"Batch input shape: {signal.shape}")
    print(f"Batch output shape: {batch_result.shape}")

    # Performance test
    large_signal = torch.randn(10_000)
    hilbert_large = Hilbert(seq_len=10_000)

    # Warmup
    for _ in range(10):
        _ = hilbert_large(large_signal)

    # Timing
    start_time = time.time()
    n_iterations = 100
    for _ in range(n_iterations):
        _ = hilbert_large(large_signal)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / n_iterations * 1000

    # Visualize performance metrics
    fig, (ax1, ax2) = stx.plt.subplots(1, 2, figsize=(12, 5))

    # Batch processing visualization
    ax1.bar(
        ["Input", "Output Phase", "Output Amplitude"],
        [
            batch_size * time_steps,
            batch_size * time_steps,
            batch_size * time_steps,
        ],
        color=["blue", "green", "red"],
        alpha=0.7,
    )
    ax1.set_title("Batch Processing Dimensions")
    ax1.set_ylabel("Total Elements")
    ax1.grid(True, alpha=0.3)

    # Performance metrics
    throughput_samples_per_sec = 10_000 / (avg_time_ms / 1000)
    ax2.bar(
        ["Processing Time", "Throughput"],
        [
            avg_time_ms,
            throughput_samples_per_sec / 1000,
        ],  # Scale for visualization
        color=["orange", "purple"],
        alpha=0.7,
    )
    ax2.set_title("Performance Metrics")
    ax2.set_ylabel("Time [ms] / Throughput [k samples/s]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    stx.io.save(fig, "02_hilbert_performance.gif")
    plt.close()

    print(f"\nPerformance Summary:")
    print(f"Average processing time (10k samples): {avg_time_ms:.2f} ms")
    print(f"Throughput: {throughput_samples_per_sec:.0f} samples/second")


def main(args):
    """Main function to demonstrate Hilbert transform functionality."""
    demo_hilbert_analysis(args)
    demo_batch_processing_and_performance(args)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import scitex as stx

    parser = argparse.ArgumentParser(
        description="Demonstrate Hilbert transform functionality"
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1000.0,
        help="Sampling rate [Hz] (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Signal duration [s] (default: %(default)s)",
    )

    args = parser.parse_args()
    stx.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize stx framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = stx.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
