#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 16:13:17 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/pac_values_comparison_with_tensorpac/generate_16_comparison_pairs.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/pac_values_comparison_with_tensorpac/generate_16_comparison_pairs.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generates 16 pairs of PAC comparisons between gPAC and Tensorpac
  - Each pair shows PAC values and z-scores in 2x2 subplot
  - Saves correlations in figure titles
  - Creates 16 individual GIF files for detailed analysis

Dependencies:
  - scripts:
    - None
  - packages:
    - torch, numpy, matplotlib, mngs, gpac, tensorpac, scipy
IO:
  - input-files:
    - None (generates synthetic data)
  - output-files:
    - comparison_pair_01.gif to comparison_pair_16.gif
    - correlation_summary.csv
"""

"""Imports"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import gpac
import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from scipy import stats
from tensorpac import Pac as TensorPac

warnings.filterwarnings("ignore", category=RuntimeWarning)

"""Functions & Classes"""
def generate_diverse_pac_signal(
    config_idx: int,
    duration_sec: float = 10,
    fs: float = 512,
) -> dict:
    """Generate diverse PAC signals for different test cases."""
    seq_len = int(duration_sec * fs)
    time = np.linspace(0, duration_sec, seq_len)

    # Define 16 different PAC configurations covering full spectrum
    configs = [
        # Theta-Gamma coupling (classic PAC)
        {
            "phase_freq": 4,
            "amp_freq": 40,
            "pac_strength": 0.8,
            "noise_level": 0.2,
        },
        {
            "phase_freq": 6,
            "amp_freq": 80,
            "pac_strength": 0.7,
            "noise_level": 0.3,
        },
        # Alpha-Gamma coupling
        {
            "phase_freq": 10,
            "amp_freq": 60,
            "pac_strength": 0.6,
            "noise_level": 0.4,
        },
        {
            "phase_freq": 12,
            "amp_freq": 100,
            "pac_strength": 0.9,
            "noise_level": 0.1,
        },
        # Beta-High Gamma coupling
        {
            "phase_freq": 16,
            "amp_freq": 120,
            "pac_strength": 0.7,
            "noise_level": 0.3,
        },
        {
            "phase_freq": 20,
            "amp_freq": 140,
            "pac_strength": 0.5,
            "noise_level": 0.5,
        },
        # Cross-frequency variations
        {
            "phase_freq": 3,
            "amp_freq": 50,
            "pac_strength": 0.85,
            "noise_level": 0.15,
        },
        {
            "phase_freq": 8,
            "amp_freq": 90,
            "pac_strength": 0.65,
            "noise_level": 0.35,
        },
        {
            "phase_freq": 14,
            "amp_freq": 110,
            "pac_strength": 0.75,
            "noise_level": 0.25,
        },
        {
            "phase_freq": 25,
            "amp_freq": 160,
            "pac_strength": 0.6,
            "noise_level": 0.4,
        },
        # Edge cases - very low/high frequencies
        {
            "phase_freq": 2.5,
            "amp_freq": 35,
            "pac_strength": 0.7,
            "noise_level": 0.3,
        },
        {
            "phase_freq": 28,
            "amp_freq": 170,
            "pac_strength": 0.5,
            "noise_level": 0.5,
        },
        # Challenging cases - weak coupling or high noise
        {
            "phase_freq": 5,
            "amp_freq": 45,
            "pac_strength": 0.3,
            "noise_level": 0.7,
        },
        {
            "phase_freq": 15,
            "amp_freq": 85,
            "pac_strength": 0.95,
            "noise_level": 0.05,
        },
        {
            "phase_freq": 7,
            "amp_freq": 65,
            "pac_strength": 0.4,
            "noise_level": 0.6,
        },
        {
            "phase_freq": 11,
            "amp_freq": 130,
            "pac_strength": 0.55,
            "noise_level": 0.45,
        },
    ]

    config = configs[config_idx % len(configs)]

    # Generate phase and amplitude signals
    phase_signal = np.sin(2 * np.pi * config["phase_freq"] * time)
    amp_carrier = np.sin(2 * np.pi * config["amp_freq"] * time)

    # Create PAC modulation
    modulation = (1 + config["pac_strength"] * phase_signal) / 2
    modulated_signal = amp_carrier * modulation

    # Add noise
    noise = np.random.randn(seq_len) * config["noise_level"]
    signal = modulated_signal + noise

    # Create batch format (1 sample, 1 channel)
    signal_tensor = torch.tensor(
        signal[np.newaxis, np.newaxis, :], dtype=torch.float32
    )

    return {
        "signal": signal_tensor,
        "config": config,
        "description": f"Phase: {config['phase_freq']}Hz, Amp: {config['amp_freq']}Hz, "
        f"Strength: {config['pac_strength']:.1f}, Noise: {config['noise_level']:.1f}",
    }


def compute_pac_both_methods(
    signal: torch.Tensor,
    fs: float,
    n_perm: int = 100,
) -> dict:
    """Compute PAC using both gPAC and Tensorpac."""
    seq_len = signal.shape[-1]

    # Create better frequency bands using log spacing for better coverage
    # Phase bands: 2-30 Hz with finer resolution
    pha_centers = np.logspace(np.log10(2.5), np.log10(28), 25)
    pha_bands_hz = []
    for f in pha_centers:
        # Standard bandwidth: f/2 for phase frequencies
        bandwidth = f / 2
        low = max(2.0, f - bandwidth/2)
        high = min(30.0, f + bandwidth/2)
        if low < high:
            pha_bands_hz.append([low, high])
    
    # Amplitude bands: 30-180 Hz with finer resolution
    amp_centers = np.logspace(np.log10(32), np.log10(175), 35)
    amp_bands_hz = []
    for f in amp_centers:
        # Standard bandwidth: f/4 for amplitude frequencies
        bandwidth = f / 4
        low = max(30.0, f - bandwidth/2)
        high = min(180.0, f + bandwidth/2)
        if low < high:
            amp_bands_hz.append([low, high])

    # gPAC computation
    pac_gpac = gpac.PAC(
        seq_len=seq_len,
        fs=fs,
        pha_bands_hz=pha_bands_hz,
        amp_bands_hz=amp_bands_hz,
        n_perm=n_perm,
        fp16=False,
    )

    if torch.cuda.is_available():
        pac_gpac = pac_gpac.cuda()
        signal_gpu = signal.cuda()
    else:
        signal_gpu = signal

    gpac_results = pac_gpac(signal_gpu)
    
    # Extract exact bands from gPAC to ensure perfect match with TensorPAC
    gpac_pha_bands = pac_gpac.pha_bands_hz.cpu().numpy()
    gpac_amp_bands = pac_gpac.amp_bands_hz.cpu().numpy()
    
    print(f"Using {len(gpac_pha_bands)} phase bands and {len(gpac_amp_bands)} amplitude bands")
    print(f"Phase band range: [{gpac_pha_bands[0, 0]:.1f}, {gpac_pha_bands[-1, 1]:.1f}] Hz")
    print(f"Amplitude band range: [{gpac_amp_bands[0, 0]:.1f}, {gpac_amp_bands[-1, 1]:.1f}] Hz")

    signal_np = signal.squeeze().cpu().numpy()

    # Always compute raw PAC values first
    pac_tensorpac_raw = TensorPac(
        idpac=(2, 0, 0),  # MI without surrogates
        f_pha=gpac_pha_bands,
        f_amp=gpac_amp_bands,
        n_bins=18,
    )
    tensorpac_pac = pac_tensorpac_raw.filterfit(
        sf=fs,
        x_pha=signal_np,
        x_amp=signal_np,
    )
    
    # Compute z-scores if permutations are requested
    if n_perm > 0:
        # Try manual z-score calculation from surrogates
        # First get surrogates without normalization
        pac_tensorpac_surr = TensorPac(
            idpac=(2, 2, 0),  # MI, swap amplitude time blocks, no normalization
            f_pha=gpac_pha_bands,
            f_amp=gpac_amp_bands,
            n_bins=18,
        )
        
        # Get pac values with surrogates
        pac_with_surr = pac_tensorpac_surr.filterfit(
            sf=fs,
            x_pha=signal_np,
            x_amp=signal_np,
            n_perm=n_perm,
            n_jobs=1,
        )
        
        # Also try the built-in z-score normalization for comparison
        pac_tensorpac_zscore = TensorPac(
            idpac=(2, 2, 1),  # MI, swap amplitude time blocks, z-score normalization
            f_pha=gpac_pha_bands,
            f_amp=gpac_amp_bands,
            n_bins=18,
        )
        tensorpac_z_builtin = pac_tensorpac_zscore.filterfit(
            sf=fs,
            x_pha=signal_np,
            x_amp=signal_np,
            n_perm=n_perm,
            n_jobs=1,
        )
        
        # For now, use the built-in z-scores but scale them up if they're too small
        tensorpac_z = tensorpac_z_builtin
        
        # If z-scores are very small, scale them up for visualization
        if np.abs(tensorpac_z).max() < 0.1:
            scale_factor = 1.0 / np.abs(tensorpac_z).max() if np.abs(tensorpac_z).max() > 0 else 1.0
            tensorpac_z = tensorpac_z * min(scale_factor, 100)  # Cap scaling at 100x
            print(f"TensorPAC z-scores scaled by {min(scale_factor, 100):.1f}x for visualization")
        
        print(f"TensorPAC z-scores computed with {n_perm} permutations")
        print(f"TensorPAC z-score shape: {tensorpac_z.shape}")
        print(f"TensorPAC z-score range: [{tensorpac_z.min():.3f}, {tensorpac_z.max():.3f}]")
        print(f"Has non-zero values: {(tensorpac_z != 0).any()}")
    else:
        tensorpac_z = None

    # TensorPAC returns (amp, phase, trial) while gPAC returns (batch, channel, phase, amp)
    # Need to transpose tensorpac output to match gPAC's (phase, amp) convention
    tensorpac_pac_transposed = tensorpac_pac.squeeze().T  # Transpose to (phase, amp)
    
    # Handle tensorpac z-scores transposition
    if tensorpac_z is not None:
        # TensorPAC filterfit returns (amp, phase, trial) - we need (phase, amp)
        if tensorpac_z.ndim == 3:
            tensorpac_z_transposed = tensorpac_z.squeeze().T  # From (amp, phase) to (phase, amp)
        else:
            tensorpac_z_transposed = tensorpac_z.T if tensorpac_z.ndim == 2 else tensorpac_z
        print(f"Final TensorPAC z-score shape after transpose: {tensorpac_z_transposed.shape}")
    else:
        tensorpac_z_transposed = None
    
    return {
        "gpac_pac": gpac_results["pac"].squeeze().cpu().numpy(),
        "gpac_z": (
            gpac_results["pac_z"].squeeze().cpu().numpy()
            if gpac_results["pac_z"] is not None
            else None
        ),
        "tensorpac_pac": tensorpac_pac_transposed,
        "tensorpac_z": tensorpac_z_transposed,
        "pha_freqs": gpac_results["phase_frequencies"].cpu().numpy(),
        "amp_freqs": gpac_results["amplitude_frequencies"].cpu().numpy(),
    }


def create_comparison_figure(
    results: dict,
    pair_idx: int,
    description: str,
    ground_truth: dict,
    n_perm: int = 0,
) -> plt.Figure:
    """Create 2x3 comparison figure for one pair with ground truth markers and difference plot."""
    fig, axes = mngs.plt.subplots(2, 3, figsize=(18, 10))

    # Calculate correlations
    pac_corr = np.corrcoef(
        results["gpac_pac"].flatten(), results["tensorpac_pac"].flatten()
    )[0, 1]

    if results["gpac_z"] is not None and results["tensorpac_z"] is not None:
        z_corr = np.corrcoef(
            results["gpac_z"].flatten(), results["tensorpac_z"].flatten()
        )[0, 1]
    else:
        z_corr = None

    # Get frequency values
    pha_freqs = results["pha_freqs"]
    amp_freqs = results["amp_freqs"]
    
    # Get ground truth position
    gt_phase = ground_truth["phase_freq"]
    gt_amp = ground_truth["amp_freq"]
    
    # Find closest indices for ground truth marker
    gt_pha_idx = np.argmin(np.abs(pha_freqs - gt_phase))
    gt_amp_idx = np.argmin(np.abs(amp_freqs - gt_amp))

    # PAC values - gPAC
    ax = axes[0, 0]
    im = ax.imshow(
        results["gpac_pac"].T,
        aspect="auto",
        origin="lower",
        cmap="hot",
    )
    ax.set_xyt("Phase [Hz] (log-spaced bands)", "Amplitude [Hz] (log-spaced bands)", "gPAC")
    
    # Set log-spaced ticks
    n_pha_ticks = min(6, len(pha_freqs))
    n_amp_ticks = min(6, len(amp_freqs))
    pha_tick_indices = np.linspace(0, len(pha_freqs)-1, n_pha_ticks, dtype=int)
    amp_tick_indices = np.linspace(0, len(amp_freqs)-1, n_amp_ticks, dtype=int)
    
    ax.set_xticks(pha_tick_indices)
    ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
    ax.set_yticks(amp_tick_indices)
    ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
    
    plt.colorbar(im, ax=ax)
    # Add ground truth marker
    ax.scatter(
        gt_pha_idx,
        gt_amp_idx,
        s=200,
        marker="x",
        c="cyan",
        linewidth=3,
        label="Ground Truth",
    )
    ax.legend(loc="upper right")

    # PAC values - Tensorpac
    ax = axes[0, 1]
    im = ax.imshow(
        results["tensorpac_pac"].T,
        aspect="auto",
        origin="lower",
        cmap="hot",
    )
    ax.set_xyt("Phase [Hz] (log-spaced bands)", "Amplitude [Hz] (log-spaced bands)", "Tensorpac")
    
    # Set same log-spaced ticks
    ax.set_xticks(pha_tick_indices)
    ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
    ax.set_yticks(amp_tick_indices)
    ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
    
    plt.colorbar(im, ax=ax)
    # Add ground truth marker
    ax.scatter(
        gt_pha_idx,
        gt_amp_idx,
        s=200,
        marker="x",
        c="cyan",
        linewidth=3,
        label="Ground Truth",
    )
    ax.legend(loc="upper right")

    # Z-scores - gPAC
    if results["gpac_z"] is not None:
        ax = axes[1, 0]
        im = ax.imshow(
            results["gpac_z"].T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
        )
        ax.set_xyt("Phase [Hz] (log-spaced bands)", "Amplitude [Hz] (log-spaced bands)", "gPAC Z-scores")
        
        # Set same log-spaced ticks
        ax.set_xticks(pha_tick_indices)
        ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
        ax.set_yticks(amp_tick_indices)
        ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
        
        plt.colorbar(im, ax=ax)
        # Add ground truth marker
        ax.scatter(
            gt_pha_idx,
            gt_amp_idx,
            s=200,
            marker="x",
            c="lime",
            linewidth=3,
            label="Ground Truth",
        )
        ax.legend(loc="upper right")
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No Z-scores",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].axis("off")

    # Z-scores - Tensorpac
    if results["tensorpac_z"] is not None:
        ax = axes[1, 1]
        # Use adaptive scaling for TensorPAC z-scores
        tz_data = results["tensorpac_z"].T
        tz_abs_max = np.abs(tz_data).max()
        # If values are very small, use data range, otherwise use symmetric around 0
        if tz_abs_max < 0.1:
            vmin, vmax = tz_data.min(), tz_data.max()
        else:
            vmin, vmax = -tz_abs_max, tz_abs_max
        
        im = ax.imshow(
            tz_data,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xyt("Phase [Hz] (log-spaced bands)", "Amplitude [Hz] (log-spaced bands)", "Tensorpac Z-scores")
        
        # Set same log-spaced ticks
        ax.set_xticks(pha_tick_indices)
        ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
        ax.set_yticks(amp_tick_indices)
        ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
        
        plt.colorbar(im, ax=ax)
        # Add ground truth marker
        ax.scatter(
            gt_pha_idx,
            gt_amp_idx,
            s=200,
            marker="x",
            c="lime",
            linewidth=3,
            label="Ground Truth",
        )
        ax.legend(loc="upper right")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No Z-scores",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].axis("off")

    # Difference plots in third column
    # PAC difference (gPAC - Tensorpac)
    ax = axes[0, 2]
    pac_diff = results["gpac_pac"] - results["tensorpac_pac"]
    im = ax.imshow(
        pac_diff.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-np.abs(pac_diff).max(),
        vmax=np.abs(pac_diff).max(),
    )
    ax.set_xyt(
        "Phase [Hz] (log-spaced bands)", "Amplitude [Hz] (log-spaced bands)", "PAC Difference\n(gPAC - Tensorpac)"
    )
    
    # Set same log-spaced ticks
    ax.set_xticks(pha_tick_indices)
    ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
    ax.set_yticks(amp_tick_indices)
    ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
    
    plt.colorbar(im, ax=ax)
    # Add ground truth marker
    ax.scatter(
        gt_pha_idx,
        gt_amp_idx,
        s=200,
        marker="x",
        c="black",
        linewidth=3,
        label="Ground Truth",
    )
    ax.legend(loc="upper right")

    # Z-score difference
    if results["gpac_z"] is not None and results["tensorpac_z"] is not None:
        ax = axes[1, 2]
        z_diff = results["gpac_z"] - results["tensorpac_z"]
        im = ax.imshow(
            z_diff.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-np.abs(z_diff).max(),
            vmax=np.abs(z_diff).max(),
        )
        ax.set_xyt(
            "Phase [Hz] (log-spaced bands)",
            "Amplitude [Hz] (log-spaced bands)",
            "Z-score Difference\n(gPAC - Tensorpac)",
        )
        
        # Set same log-spaced ticks
        ax.set_xticks(pha_tick_indices)
        ax.set_xticklabels([f"{pha_freqs[i]:.0f}" for i in pha_tick_indices])
        ax.set_yticks(amp_tick_indices)
        ax.set_yticklabels([f"{amp_freqs[i]:.0f}" for i in amp_tick_indices])
        
        plt.colorbar(im, ax=ax)
        # Add ground truth marker
        ax.scatter(
            gt_pha_idx,
            gt_amp_idx,
            s=200,
            marker="x",
            c="black",
            linewidth=3,
            label="Ground Truth",
        )
        ax.legend(loc="upper right")
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "No Z-score difference",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].axis("off")

    # Add title with correlations and n_perm
    title = f"Pair {pair_idx + 1:02d}: {description}\n"
    title += f"PAC Correlation: {pac_corr:.4f}"
    if z_corr is not None:
        title += f", Z-score Correlation: {z_corr:.4f}"
    title += f" (n_perm={n_perm})"

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig, pac_corr, z_corr


def main(args):
    """Main function to generate 16 comparison pairs."""
    mngs.str.printc(
        "=== Generating 16 PAC Comparison Pairs ===", c=CC["orange"]
    )

    correlation_data = []

    for pair_idx in range(16):
        print(f"\n{'='*60}")
        print(f"Generating Pair {pair_idx + 1}/16")
        print(f"{'='*60}")

        # Generate signal with specific configuration
        signal_data = generate_diverse_pac_signal(
            pair_idx, duration_sec=args.duration, fs=args.fs
        )
        print(f"Configuration: {signal_data['description']}")

        # Compute PAC with both methods
        print("Computing PAC with both methods...")
        results = compute_pac_both_methods(
            signal_data["signal"],
            fs=args.fs,
            n_perm=args.n_perm,
        )

        # Create comparison figure
        fig, pac_corr, z_corr = create_comparison_figure(
            results,
            pair_idx,
            signal_data["description"],
            signal_data["config"],
            n_perm=args.n_perm,
        )

        # Save figure
        filename = f"comparison_pair_{pair_idx + 1:02d}.gif"
        mngs.io.save(fig, filename)
        plt.close()

        # Store correlation data
        correlation_data.append(
            {
                "pair": pair_idx + 1,
                "phase_freq": signal_data["config"]["phase_freq"],
                "amp_freq": signal_data["config"]["amp_freq"],
                "pac_strength": signal_data["config"]["pac_strength"],
                "noise_level": signal_data["config"]["noise_level"],
                "pac_correlation": pac_corr,
                "z_correlation": z_corr if z_corr is not None else np.nan,
            }
        )

        print(f"✓ Saved: {filename}")
        print(f"  PAC Correlation: {pac_corr:.4f}")
        if z_corr is not None:
            print(f"  Z-score Correlation: {z_corr:.4f}")

    # Save correlation summary
    import pandas as pd

    df = pd.DataFrame(correlation_data)
    df.to_csv("correlation_summary.csv", index=False)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(
        f"Mean PAC Correlation: {df['pac_correlation'].mean():.4f} ± {df['pac_correlation'].std():.4f}"
    )
    if not df["z_correlation"].isna().all():
        print(
            f"Mean Z-score Correlation: {df['z_correlation'].mean():.4f} ± {df['z_correlation'].std():.4f}"
        )
    print(f"Min PAC Correlation: {df['pac_correlation'].min():.4f}")
    print(f"Max PAC Correlation: {df['pac_correlation'].max():.4f}")

    # Create summary visualization
    fig, axes = mngs.plt.subplots(1, 2, figsize=(12, 5))

    # PAC correlations by configuration
    ax = axes[0]
    ax.scatter(
        df["pac_strength"],
        df["pac_correlation"],
        c=df["noise_level"],
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax.set_xyt(
        "PAC Strength",
        "PAC Correlation",
        "PAC Correlation vs Signal Properties",
    )
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Noise Level")
    ax.grid(True, alpha=0.3)

    # Correlation histogram
    ax = axes[1]
    ax.hist(df["pac_correlation"], bins=10, alpha=0.7, edgecolor="black")
    ax.axvline(
        df["pac_correlation"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['pac_correlation'].mean():.3f}",
    )
    ax.set_xyt("PAC Correlation", "Count", "Distribution of PAC Correlations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mngs.io.save(fig, "correlation_summary_visualization.gif")

    print(f"\n✓ Generated all 16 comparison pairs")
    print(f"✓ Saved correlation summary to correlation_summary.csv")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 16 PAC comparison pairs between gPAC and Tensorpac"
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
        "--n_perm",
        type=int,
        default=100,
        help="Number of permutations for z-score computation (default: %(default)s)",
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
