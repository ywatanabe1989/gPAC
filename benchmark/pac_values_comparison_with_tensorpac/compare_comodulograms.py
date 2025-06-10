#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 15:40:00 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/pac_values_comparison_with_tensorpac/compare_comodulograms.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/pac_values_comparison_with_tensorpac/compare_comodulograms.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Compares PAC comodulograms between gPAC and Tensorpac
  - Calculates correlations and differences between implementations
  - Generates visualization comparing the two methods
  - Computes z-scores and p-values for both implementations

Dependencies:
  - scripts:
    - None
  - packages:
    - torch, numpy, matplotlib, mngs, gpac, tensorpac
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - comodulogram_comparison_gpac_tensorpac.gif
    - comparison_results.yaml
    - comparison_statistics.csv
"""

"""Imports"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from tensorpac import Pac as TensorPac

import gpac

warnings.filterwarnings("ignore", category=RuntimeWarning)

"""Functions & Classes"""
def generate_test_signal(
    batch_size: int = 2,
    n_channels: int = 4,
    duration_sec: float = 10,
    fs: float = 512,
    phase_freq: float = 10.0,
    amp_freq: float = 80.0,
    pac_strength: float = 0.5,
    noise_level: float = 0.5,
) -> torch.Tensor:
    """Generate synthetic PAC signal for comparison."""
    seq_len = int(duration_sec * fs)
    time = np.linspace(0, duration_sec, seq_len)
    
    # Generate phase and amplitude signals
    phase_signal = np.sin(2 * np.pi * phase_freq * time)
    amp_carrier = np.sin(2 * np.pi * amp_freq * time)
    
    # Create PAC modulation
    modulation = (1 + pac_strength * phase_signal) / 2
    modulated_signal = amp_carrier * modulation
    
    # Add noise
    noise = np.random.randn(batch_size, n_channels, seq_len) * noise_level
    
    # Broadcast signal and add noise
    signal = modulated_signal[np.newaxis, np.newaxis, :] + noise
    
    return torch.tensor(signal, dtype=torch.float32)


def compute_gpac_comodulogram(
    signal: torch.Tensor,
    fs: float,
    pha_range: tuple = (2, 30),
    amp_range: tuple = (30, 150),
    n_perm: int = 0,
) -> dict:
    """Compute PAC using gPAC."""
    seq_len = signal.shape[-1]
    
    # Create PAC model
    pac = gpac.PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=pha_range,
        amp_range_hz=amp_range,
        pha_n_bands=30,
        amp_n_bands=30,
        n_perm=n_perm,
        fp16=False,
    )
    
    if torch.cuda.is_available():
        pac = pac.cuda()
        signal = signal.cuda()
    
    # Compute PAC
    results = pac(signal)
    
    return {
        "pac": results["pac"].cpu().numpy(),
        "pac_z": results["pac_z"].cpu().numpy() if results["pac_z"] is not None else None,
        "pha_freqs": results["phase_frequencies"].cpu().numpy(),
        "amp_freqs": results["amplitude_frequencies"].cpu().numpy(),
    }


def compute_tensorpac_comodulogram(
    signal: np.ndarray,
    fs: float,
    pha_range: tuple = (2, 30),
    amp_range: tuple = (30, 150),
    n_perm: int = 0,
) -> dict:
    """Compute PAC using Tensorpac."""
    # Initialize Tensorpac
    pac = TensorPac(
        idpac=(1, 0, 0),  # MVL method
        f_pha=pha_range,
        f_amp=amp_range,
        n_bins=18,
    )
    
    # Get frequency vectors
    pha_freqs = pac.f_pha.mean(axis=1)
    amp_freqs = pac.f_amp.mean(axis=1)
    
    # Compute PAC for each sample and channel
    batch_size, n_channels, seq_len = signal.shape
    pac_values = []
    pac_pvalues = []
    
    for b in range(batch_size):
        batch_pac = []
        batch_pval = []
        for c in range(n_channels):
            if n_perm > 0:
                pac_result = pac.filterfit(
                    sf=fs,
                    x_pha=signal[b, c],
                    x_amp=signal[b, c],
                    n_perm=n_perm,
                    n_jobs=1,
                )
                batch_pac.append(pac_result)
                # Get p-values from surrogates
                pval = pac.pvalues
                batch_pval.append(pval)
            else:
                pac_result = pac.filterfit(
                    sf=fs,
                    x_pha=signal[b, c],
                    x_amp=signal[b, c],
                )
                batch_pac.append(pac_result)
                batch_pval.append(None)
        
        pac_values.append(batch_pac)
        pac_pvalues.append(batch_pval)
    
    pac_values = np.array(pac_values)
    
    # Convert p-values to z-scores if available
    pac_z = None
    if n_perm > 0 and pac_pvalues[0][0] is not None:
        pac_z = []
        for b in range(batch_size):
            batch_z = []
            for c in range(n_channels):
                if pac_pvalues[b][c] is not None:
                    # Convert p-values to z-scores
                    from scipy import stats
                    z_scores = stats.norm.ppf(1 - pac_pvalues[b][c])
                    batch_z.append(z_scores)
            pac_z.append(batch_z)
        pac_z = np.array(pac_z)
    
    return {
        "pac": pac_values,
        "pac_z": pac_z,
        "pha_freqs": pha_freqs,
        "amp_freqs": amp_freqs,
    }


def plot_comparison(
    gpac_results: dict,
    tensorpac_results: dict,
    sample_idx: int = 0,
    channel_idx: int = 0,
) -> tuple:
    """Create comparison visualization."""
    fig, axes = mngs.plt.subplots(2, 4, figsize=(20, 10))
    
    # Extract data
    gpac_pac = gpac_results["pac"][sample_idx, channel_idx]
    tensorpac_pac = tensorpac_results["pac"][sample_idx, channel_idx]
    
    # Common colormap parameters
    vmin = min(gpac_pac.min(), tensorpac_pac.min())
    vmax = max(gpac_pac.max(), tensorpac_pac.max())
    
    # gPAC comodulogram
    ax = axes[0, 0]
    im1 = ax.imshow(
        gpac_pac.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        extent=[
            gpac_results["pha_freqs"][0],
            gpac_results["pha_freqs"][-1],
            gpac_results["amp_freqs"][0],
            gpac_results["amp_freqs"][-1],
        ],
    )
    ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "gPAC")
    plt.colorbar(im1, ax=ax, label="PAC Value")
    
    # Tensorpac comodulogram
    ax = axes[0, 1]
    im2 = ax.imshow(
        tensorpac_pac.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        extent=[
            tensorpac_results["pha_freqs"][0],
            tensorpac_results["pha_freqs"][-1],
            tensorpac_results["amp_freqs"][0],
            tensorpac_results["amp_freqs"][-1],
        ],
    )
    ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "Tensorpac")
    plt.colorbar(im2, ax=ax, label="PAC Value")
    
    # Difference plot
    ax = axes[0, 2]
    diff = gpac_pac - tensorpac_pac
    im3 = ax.imshow(
        diff.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-np.abs(diff).max(),
        vmax=np.abs(diff).max(),
        extent=[
            gpac_results["pha_freqs"][0],
            gpac_results["pha_freqs"][-1],
            gpac_results["amp_freqs"][0],
            gpac_results["amp_freqs"][-1],
        ],
    )
    ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "Difference (gPAC - Tensorpac)")
    plt.colorbar(im3, ax=ax, label="Difference")
    
    # Scatter plot
    ax = axes[0, 3]
    ax.scatter(tensorpac_pac.flatten(), gpac_pac.flatten(), alpha=0.5, s=10)
    ax.plot(
        [tensorpac_pac.min(), tensorpac_pac.max()],
        [tensorpac_pac.min(), tensorpac_pac.max()],
        "r--",
        label="y=x",
    )
    ax.set_xyt("Tensorpac PAC", "gPAC PAC", "Correlation Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    correlation = np.corrcoef(gpac_pac.flatten(), tensorpac_pac.flatten())[0, 1]
    rmse = np.sqrt(np.mean((gpac_pac - tensorpac_pac) ** 2))
    
    # Z-score comparison if available
    if gpac_results["pac_z"] is not None and tensorpac_results["pac_z"] is not None:
        gpac_z = gpac_results["pac_z"][sample_idx, channel_idx]
        tensorpac_z = tensorpac_results["pac_z"][sample_idx, channel_idx]
        
        # gPAC z-scores
        ax = axes[1, 0]
        im4 = ax.imshow(
            gpac_z.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
        )
        ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "gPAC Z-scores")
        plt.colorbar(im4, ax=ax, label="Z-score")
        
        # Tensorpac z-scores
        ax = axes[1, 1]
        im5 = ax.imshow(
            tensorpac_z.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
        )
        ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "Tensorpac Z-scores")
        plt.colorbar(im5, ax=ax, label="Z-score")
        
        # Z-score difference
        ax = axes[1, 2]
        z_diff = gpac_z - tensorpac_z
        im6 = ax.imshow(
            z_diff.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-np.abs(z_diff).max(),
            vmax=np.abs(z_diff).max(),
        )
        ax.set_xyt("Phase Frequency [Hz]", "Amplitude Frequency [Hz]", "Z-score Difference")
        plt.colorbar(im6, ax=ax, label="Difference")
        
        # Z-score scatter
        ax = axes[1, 3]
        ax.scatter(tensorpac_z.flatten(), gpac_z.flatten(), alpha=0.5, s=10)
        ax.plot([-3, 3], [-3, 3], "r--", label="y=x")
        ax.set_xyt("Tensorpac Z-score", "gPAC Z-score", "Z-score Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        z_correlation = np.corrcoef(gpac_z.flatten(), tensorpac_z.flatten())[0, 1]
    else:
        z_correlation = None
        # Clear unused subplots
        for ax in axes[1, :]:
            ax.axis("off")
    
    plt.suptitle(
        f"gPAC vs Tensorpac Comparison\n"
        f"PAC Correlation: {correlation:.4f}, RMSE: {rmse:.4f}"
        + (f", Z-score Correlation: {z_correlation:.4f}" if z_correlation else ""),
        fontsize=14,
    )
    plt.tight_layout()
    
    return fig, correlation, rmse, z_correlation


def main(args):
    """Main comparison function."""
    mngs.str.printc("=== gPAC vs Tensorpac Comparison ===", c=CC["orange"])
    
    # Generate test signal
    print("\n==> Generating test signal...")
    signal = generate_test_signal(
        batch_size=args.batch_size,
        n_channels=args.n_channels,
        duration_sec=args.duration,
        fs=args.fs,
        phase_freq=args.phase_freq,
        amp_freq=args.amp_freq,
        pac_strength=args.pac_strength,
        noise_level=args.noise_level,
    )
    print(f"Signal shape: {signal.shape}")
    
    # Compute gPAC
    print("\n==> Computing gPAC comodulogram...")
    gpac_results = compute_gpac_comodulogram(
        signal,
        fs=args.fs,
        pha_range=(2, 30),
        amp_range=(30, 150),
        n_perm=args.n_perm,
    )
    print(f"gPAC shape: {gpac_results['pac'].shape}")
    print(f"gPAC range: [{gpac_results['pac'].min():.4f}, {gpac_results['pac'].max():.4f}]")
    
    # Compute Tensorpac
    print("\n==> Computing Tensorpac comodulogram...")
    tensorpac_results = compute_tensorpac_comodulogram(
        signal.numpy(),
        fs=args.fs,
        pha_range=(2, 30),
        amp_range=(30, 150),
        n_perm=args.n_perm,
    )
    print(f"Tensorpac shape: {tensorpac_results['pac'].shape}")
    print(f"Tensorpac range: [{tensorpac_results['pac'].min():.4f}, {tensorpac_results['pac'].max():.4f}]")
    
    # Create comparison plots
    print("\n==> Creating comparison visualization...")
    fig, correlation, rmse, z_correlation = plot_comparison(
        gpac_results,
        tensorpac_results,
        sample_idx=0,
        channel_idx=0,
    )
    
    # Save results
    mngs.io.save(fig, "comodulogram_comparison_gpac_tensorpac.gif")
    
    # Save statistics
    results = {
        "pac_correlation": float(correlation),
        "pac_rmse": float(rmse),
        "z_correlation": float(z_correlation) if z_correlation else None,
        "gpac_pac_mean": float(gpac_results["pac"].mean()),
        "gpac_pac_std": float(gpac_results["pac"].std()),
        "tensorpac_pac_mean": float(tensorpac_results["pac"].mean()),
        "tensorpac_pac_std": float(tensorpac_results["pac"].std()),
        "signal_params": {
            "batch_size": args.batch_size,
            "n_channels": args.n_channels,
            "duration_sec": args.duration,
            "fs": args.fs,
            "phase_freq": args.phase_freq,
            "amp_freq": args.amp_freq,
            "pac_strength": args.pac_strength,
            "noise_level": args.noise_level,
        },
    }
    
    mngs.io.save(results, "comparison_results.yaml")
    
    # Save CSV for further analysis
    comparison_df = {
        "metric": ["correlation", "rmse", "gpac_mean", "gpac_std", "tensorpac_mean", "tensorpac_std"],
        "value": [
            correlation,
            rmse,
            gpac_results["pac"].mean(),
            gpac_results["pac"].std(),
            tensorpac_results["pac"].mean(),
            tensorpac_results["pac"].std(),
        ],
    }
    mngs.io.save(comparison_df, "comparison_statistics.csv")
    
    # Print summary
    print("\n==> Comparison Results:")
    print(f"   PAC Correlation: {correlation:.4f}")
    print(f"   PAC RMSE: {rmse:.4f}")
    if z_correlation:
        print(f"   Z-score Correlation: {z_correlation:.4f}")
    print(f"   gPAC mean: {gpac_results['pac'].mean():.4f} ± {gpac_results['pac'].std():.4f}")
    print(f"   Tensorpac mean: {tensorpac_results['pac'].mean():.4f} ± {tensorpac_results['pac'].std():.4f}")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PAC comodulograms between gPAC and Tensorpac"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=4,
        help="Number of channels (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Signal duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=512.0,
        help="Sampling frequency (default: %(default)s)",
    )
    parser.add_argument(
        "--phase_freq",
        type=float,
        default=10.0,
        help="Phase frequency for synthetic PAC (default: %(default)s)",
    )
    parser.add_argument(
        "--amp_freq",
        type=float,
        default=80.0,
        help="Amplitude frequency for synthetic PAC (default: %(default)s)",
    )
    parser.add_argument(
        "--pac_strength",
        type=float,
        default=0.5,
        help="PAC coupling strength (default: %(default)s)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.5,
        help="Noise level (default: %(default)s)",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=100,
        help="Number of permutations for surrogate testing (default: %(default)s)",
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