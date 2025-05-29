#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 01:00:00 (ywatanabe)"
# File: ./examples/gpac/example_Hilbert.py

"""
Functionalities:
  - Demonstrates Hilbert transform usage in gPAC
  - Shows extraction of instantaneous phase and amplitude
  - Visualizes analytical signal components
  - Compares with scipy.signal.hilbert for validation

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch 
    - numpy
    - scipy
    - matplotlib
    - mngs
    
IO:
  - input-files:
    - None (generates synthetic signals)
    
  - output-files:
    - ./scripts/example_Hilbert/hilbert_transform_demo.png
    - ./scripts/example_Hilbert/hilbert_results.npz
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
from scipy import signal as sp_signal

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""

def main(args):
    """Run Hilbert transform demonstration."""
    import mngs
    from gpac import Hilbert
    
    mngs.str.printc("🚀 Hilbert Transform Demonstration", c="green")
    mngs.str.printc("=" * 50, c="green")
    
    # Parameters
    fs = 256  # Sampling frequency (Hz)
    duration = 2.0  # seconds
    freq1 = 10  # Hz
    freq2 = 40  # Hz
    
    # Create composite signal
    mngs.str.printc("\n📡 Creating test signal...", c="cyan")
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    signal += 0.1 * np.random.randn(len(t))  # Add noise
    
    mngs.str.printc(f"Signal: {freq1} Hz + {freq2} Hz components", c="cyan")
    mngs.str.printc(f"Duration: {duration} s, Sampling rate: {fs} Hz", c="cyan")
    
    # Convert to torch tensor (add batch dimension)
    signal_torch = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    
    # Apply gPAC Hilbert transform
    mngs.str.printc("\n🔄 Applying Hilbert transform...", c="blue")
    hilbert = Hilbert(seq_len=len(t))
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hilbert = hilbert.to(device)
    signal_torch = signal_torch.to(device)
    
    # Apply transform
    analytic_signal = hilbert(signal_torch)
    
    # Extract phase and amplitude
    phase = analytic_signal[..., 0].cpu().numpy().squeeze()
    amplitude = analytic_signal[..., 1].cpu().numpy().squeeze()
    
    # Compare with scipy
    analytic_scipy = sp_signal.hilbert(signal)
    phase_scipy = np.angle(analytic_scipy)
    amplitude_scipy = np.abs(analytic_scipy)
    
    mngs.str.printc("✅ Hilbert transform completed", c="green")
    
    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    fig, axes = mngs.plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot original signal
    ax = axes[0, 0]
    ax.plot(t[:256], signal[:256], 'b-', linewidth=1)
    ax.set_xyt("Time (s)", "Amplitude", "Original Signal")
    ax.grid(True, alpha=0.3)
    
    # Plot amplitude (gPAC)
    ax = axes[0, 1]
    ax.plot(t[:256], amplitude[:256], 'r-', linewidth=1, label='gPAC')
    ax.plot(t[:256], amplitude_scipy[:256], 'k--', linewidth=1, alpha=0.7, label='SciPy')
    ax.set_xyt("Time (s)", "Amplitude", "Instantaneous Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot phase (gPAC)
    ax = axes[1, 0]
    ax.plot(t[:256], phase[:256], 'g-', linewidth=1, label='gPAC')
    ax.plot(t[:256], phase_scipy[:256], 'k--', linewidth=1, alpha=0.7, label='SciPy')
    ax.set_xyt("Time (s)", "Phase (rad)", "Instantaneous Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot phase unwrapped
    ax = axes[1, 1]
    phase_unwrapped = np.unwrap(phase)
    phase_scipy_unwrapped = np.unwrap(phase_scipy)
    ax.plot(t[:256], phase_unwrapped[:256], 'g-', linewidth=1, label='gPAC')
    ax.plot(t[:256], phase_scipy_unwrapped[:256], 'k--', linewidth=1, alpha=0.7, label='SciPy')
    ax.set_xyt("Time (s)", "Phase (rad)", "Unwrapped Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot analytic signal in complex plane
    ax = axes[2, 0]
    real_part = signal[:256]
    imag_part = amplitude[:256] * np.sin(phase[:256])
    ax.plot(real_part, imag_part, 'b-', linewidth=0.5, alpha=0.7)
    ax.scatter(real_part[0], imag_part[0], c='r', s=50, marker='o', label='Start')
    ax.scatter(real_part[-1], imag_part[-1], c='g', s=50, marker='s', label='End')
    ax.set_xyt("Real", "Imaginary", "Analytic Signal (Complex Plane)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot error comparison
    ax = axes[2, 1]
    amplitude_error = np.abs(amplitude - amplitude_scipy)
    phase_error = np.abs(phase - phase_scipy)
    ax.semilogy(t[:256], amplitude_error[:256], 'r-', linewidth=1, label='Amplitude error')
    ax.semilogy(t[:256], phase_error[:256], 'g-', linewidth=1, label='Phase error')
    ax.set_xyt("Time (s)", "Absolute Error", "gPAC vs SciPy Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    spath = "hilbert_transform_demo.png"
    mngs.io.save(fig, spath)
    
    # Report statistics
    mngs.str.printc("\n📊 Results Summary", c="yellow")
    mngs.str.printc("=" * 50, c="yellow")
    mngs.str.printc(f"Signal shape: {signal_torch.shape}", c="cyan")
    mngs.str.printc(f"Amplitude shape: {amplitude.shape}", c="cyan")
    mngs.str.printc(f"Phase shape: {phase.shape}", c="cyan")
    mngs.str.printc(f"Mean amplitude: {amplitude.mean():.4f}", c="cyan")
    mngs.str.printc(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}]", c="cyan")
    
    # Compare with scipy
    mngs.str.printc("\n🔍 Comparison with SciPy:", c="yellow")
    mngs.str.printc(f"Mean amplitude error: {amplitude_error.mean():.2e}", c="cyan")
    mngs.str.printc(f"Max amplitude error: {amplitude_error.max():.2e}", c="cyan")
    mngs.str.printc(f"Mean phase error: {phase_error.mean():.2e}", c="cyan")
    mngs.str.printc(f"Max phase error: {phase_error.max():.2e}", c="cyan")
    
    # Save results
    results = {
        'signal': signal,
        'amplitude': amplitude,
        'phase': phase,
        'amplitude_scipy': amplitude_scipy,
        'phase_scipy': phase_scipy,
        't': t,
        'fs': fs
    }
    mngs.io.save(results, "hilbert_results.npz")
    
    mngs.str.printc("\n✅ Hilbert transform demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to: {spath}", c="green")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Hilbert transform demonstration")
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
