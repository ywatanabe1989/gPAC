#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 10:00:00 (ywatanabe)"
# File: example_pac_analysis.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_pac_analysis.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Basic PAC computation using gPAC
  - Synthetic data generation with known PAC
  - Visualization of PAC comodulogram
  - Performance comparison with TensorPAC (if available)

Dependencies:
  - scripts: None
  - packages: gpac, tensorpac, torch, numpy, matplotlib

IO:
  - input-files: None
  - output-files: ./example_pac_analysis_out/pac_analysis.png, ./example_pac_analysis_out/pac_results.pkl
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch

# Optional: Import TensorPAC for comparison
try:
    from tensorpac import Pac as TensorPAC

    HAS_TENSORPAC = True
except ImportError:
    HAS_TENSORPAC = False


"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""


def generate_pac_signal(
    fs=250,
    duration=5.0,
    phase_freq=6.0,
    amp_freq=60.0,
    coupling_strength=0.7,
    noise_level=0.1,
):
    """Generate synthetic signal with known PAC."""
    from gpac import SyntheticDataGenerator

    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level,
    )
    return signal


def compute_pac(signal, fs, phase_freqs, amp_freqs):
    """Compute PAC using gPAC."""
    from gpac import PAC
    import mngs

    # Convert to tensor if needed
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float()

    # Add batch and channel dimensions if needed
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(0)

    # Get sequence length
    seq_len = signal.shape[-1]
    
    # Initialize PAC calculator with correct API
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=phase_freqs[0],
        pha_end_hz=phase_freqs[-1],
        pha_n_bands=len(phase_freqs),
        amp_start_hz=amp_freqs[0],
        amp_end_hz=amp_freqs[-1],
        amp_n_bands=len(amp_freqs),
        n_perm=None,  # No surrogates for this example
        trainable=False,
        fp16=False,
    )

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pac = pac.to(device)
    signal = signal.to(device)

    # Compute PAC
    with torch.no_grad():
        pac_results = pac(signal)
    
    # Extract PAC values
    pac_values = pac_results['pac'].squeeze().cpu().numpy()

    return pac_values


def compute_pac_tensorpac(signal, fs, phase_freqs, amp_freqs):
    """Compute PAC using TensorPAC for comparison."""
    import mngs

    if not HAS_TENSORPAC:
        return None, False

    pac = TensorPAC(idpac=(2, 0, 0), f_pha=phase_freqs, f_amp=amp_freqs)

    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    pac_values = pac.filterfit(fs, signal)
    return pac_values.squeeze(), False


def visualize_pac_results(
    signal, pac_gpac, pac_tensorpac, phase_freqs, amp_freqs, phase_freq, amp_freq
):
    """Visualize PAC results."""
    import mngs
    import matplotlib.pyplot as plt

    # Create figure
    nrows = 1
    ncols = 2 if HAS_TENSORPAC else 1
    figsize = (12 if HAS_TENSORPAC else 6, 5)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, facecolor="white"
    )
    if ncols == 1:
        axes = [axes]

    # Plot gPAC results
    ax = axes[0]
    im = ax.imshow(
        pac_gpac,
        aspect="auto",
        origin="lower",
        extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
        cmap="hot",
        vmin=0,
        vmax=0.8,
    )
    ax.set_xlabel("Amplitude Frequency (Hz)")
    ax.set_ylabel("Phase Frequency (Hz)")
    ax.set_title("gPAC Results")

    # Mark the true coupling
    ax.plot(
        amp_freq,
        phase_freq,
        "co",
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="cyan",
        fillstyle="none",
    )
    ax.text(amp_freq + 5, phase_freq, "True\nCoupling", color="cyan", fontsize=10)

    # Add colorbar
    fig.colorbar(im, ax=ax, label="PAC Strength")

    # Plot TensorPAC results if available
    if HAS_TENSORPAC and pac_tensorpac is not None:
        ax = axes[1]
        im = ax.imshow(
            pac_tensorpac,
            aspect="auto",
            origin="lower",
            extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
            cmap="hot",
            vmin=0,
            vmax=0.8,
        )
        ax.set_xlabel("Amplitude Frequency (Hz)")
        ax.set_ylabel("Phase Frequency (Hz)")
        ax.set_title("TensorPAC Results")

        # Mark the true coupling
        ax.plot(
            amp_freq,
            phase_freq,
            "co",
            markersize=10,
            markeredgewidth=2,
            markeredgecolor="cyan",
            fillstyle="none",
        )

        # Add colorbar
        fig.colorbar(im, ax=ax, label="PAC Strength")

    plt.tight_layout()
    return fig


def main(args):
    """Main example function."""
    import mngs

    # Set random seed for reproducibility
    mngs.gen.fix_seeds(42)

    # Parameters
    fs = 250  # Sampling frequency
    duration = 10.0  # Duration in seconds
    phase_freq = 6.0  # Phase frequency (theta)
    amp_freq = 60.0  # Amplitude frequency (gamma)
    coupling_strength = 0.7
    noise_level = 0.1

    # Frequency ranges for PAC computation
    phase_freqs = np.arange(2, 20, 1)
    amp_freqs = np.arange(30, 100, 2)

    # Generate synthetic signal
    print("Generating synthetic PAC signal...")
    signal = generate_pac_signal(
        fs=fs,
        duration=duration,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level,
    )

    # Compute PAC using gPAC
    print("Computing PAC with gPAC...")
    import time

    start_time = time.time()
    pac_gpac = compute_pac(signal, fs, phase_freqs, amp_freqs)
    gpac_time = time.time() - start_time
    print(f"  Time: {gpac_time:.3f} seconds")

    # Compute PAC using TensorPAC if available
    pac_tensorpac = None
    tensorpac_time = None
    if HAS_TENSORPAC:
        print("Computing PAC with TensorPAC...")
        start_time = time.time()
        pac_tensorpac, from_cache = compute_pac_tensorpac(
            signal, fs, phase_freqs, amp_freqs
        )
        tensorpac_time = time.time() - start_time
        if not from_cache:
            print(f"  Time: {tensorpac_time:.3f} seconds")
            print(f"  Speedup: {tensorpac_time/gpac_time:.2f}x")

    # Visualize results
    print("\nVisualizing results...")
    fig = visualize_pac_results(
        signal, pac_gpac, pac_tensorpac, phase_freqs, amp_freqs, phase_freq, amp_freq
    )

    # Save outputs
    # The output directory is already created by mngs.gen.start
    output_dir = "./example_pac_analysis_out"
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    fig_path = os.path.join(output_dir, "pac_analysis.png")
    mngs.io.save(fig, fig_path)
    print(f"  Figure saved to: {fig_path}")

    # Save numerical results
    results = {
        "pac_gpac": pac_gpac,
        "pac_tensorpac": pac_tensorpac,
        "parameters": {
            "fs": fs,
            "duration": duration,
            "phase_freq": phase_freq,
            "amp_freq": amp_freq,
            "coupling_strength": coupling_strength,
            "noise_level": noise_level,
        },
        "computation_time": {
            "gpac": gpac_time,
            "tensorpac": tensorpac_time if HAS_TENSORPAC else None,
        },
    }
    import pickle
    results_path = os.path.join(output_dir, "pac_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Results saved to: {results_path}")

    # Find peak PAC value in gPAC results
    peak_idx = np.unravel_index(np.argmax(pac_gpac), pac_gpac.shape)
    peak_phase = phase_freqs[peak_idx[0]]
    peak_amp = amp_freqs[peak_idx[1]]
    peak_value = pac_gpac[peak_idx]

    print(f"\nPeak PAC coupling detected:")
    print(f"  Phase frequency: {peak_phase:.1f} Hz (true: {phase_freq} Hz)")
    print(f"  Amplitude frequency: {peak_amp:.1f} Hz (true: {amp_freq} Hz)")
    print(f"  PAC strength: {peak_value:.3f}")

    print("\nExample completed successfully!")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="PAC analysis example using gPAC")
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
