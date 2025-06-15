#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 11:20:05 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/gpac/core/example__ModulationIndex.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/core/example__ModulationIndex.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates ModulationIndex functionality for PAC analysis
  - Shows MI calculation with soft binning for differentiability
  - Visualizes phase-amplitude coupling strength
  - Tests gradient flow and surrogate statistics
  - Compares different phase bin configurations

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, stx

IO:
  - input-files: None (generates synthetic PAC signals)
  - output-files:
    - 01_modulation_index_basic.gif
    - 02_modulation_index_surrogates.gif
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import gpac
import matplotlib.pyplot as plt
import scitex as stx
import torch

"""Functions & Classes"""
def demo_modulation_index_basic(args):
    """Demonstrate basic ModulationIndex functionality."""

    stx.str.printc("=== Demo ModulationIndex Basic ===", c="yellow")

    # Generate synthetic PAC data
    pac_config = gpac.dataset.multi_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=2,
        n_channels=4,
        n_segments=3,
        duration_sec=2,
        fs=512,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    # Create bandpass filter and Hilbert transform
    bpf = gpac.core.BandPassFilter(
        fs=metadata["fs"][0],
        pha_range_hz=(4.0, 12.0),
        amp_range_hz=(30.0, 100.0),
        pha_n_bands=100,
        amp_n_bands=100,
        trainable=False,
    )

    hilbert = gpac.core.Hilbert(seq_len=signal.shape[-1])

    # Create ModulationIndex calculator
    mi_calc = gpac.core.ModulationIndex(n_bins=18, temperature=0.01)

    if torch.cuda.is_available():
        bpf = bpf.cuda()
        hilbert = hilbert.cuda()
        mi_calc = mi_calc.cuda()
        signal = signal.cuda()

    # Filter signals
    filtered = bpf(signal)
    n_pha = bpf.info["pha_n_bands"]
    n_amp = bpf.info["amp_n_bands"]

    # Split phase and amplitude bands
    pha_filtered = filtered[:, :, :, :n_pha, :]
    amp_filtered = filtered[:, :, :, n_pha:, :]

    # Extract phase and amplitude using Hilbert transform
    batch_size, n_channels, n_segments = pha_filtered.shape[:3]

    # Reshape for Hilbert transform
    pha_flat = pha_filtered.reshape(-1, pha_filtered.shape[-1])
    amp_flat = amp_filtered.reshape(-1, amp_filtered.shape[-1])

    pha_hilbert = hilbert(pha_flat)
    amp_hilbert = hilbert(amp_flat)

    # Extract phase and amplitude
    phase = pha_hilbert[..., 0].reshape(
        batch_size, n_channels, n_segments, n_pha, -1
    )
    amplitude = amp_hilbert[..., 1].reshape(
        batch_size, n_channels, n_segments, n_amp, -1
    )

    # Reshape for ModulationIndex: (batch, channels, freqs, segments, time)
    phase_mi = phase.permute(0, 1, 3, 2, 4)
    amplitude_mi = amplitude.permute(0, 1, 3, 2, 4)

    # Test gradient flow
    phase_grad = phase_mi.clone().requires_grad_(True)
    amplitude_grad = amplitude_mi.clone().requires_grad_(True)

    # Compute modulation index
    mi_result = mi_calc(
        phase_grad,
        amplitude_grad,
        compute_distributions=True,
        return_per_segment=True,
    )

    mi_values = mi_result["mi"]
    mi_per_segment = mi_result["mi_per_segment"]
    amp_distributions = mi_result["amplitude_distributions"]

    # Test gradient flow
    loss = mi_values.mean()
    loss.backward()

    # Create visualization
    fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))

    # MI heatmap
    ax_mi = axes[0, 0]
    mi_mean = mi_values[0, 0].cpu().detach().numpy()
    im1 = ax_mi.imshow(mi_mean, aspect="auto", cmap="viridis")
    ax_mi.set_xyt("Amplitude Frequency", "Phase Frequency", "Modulation Index")
    ax_mi.set_xticks(range(n_amp))
    ax_mi.set_xticklabels([f"{f:.1f}" for f in bpf.amp_mids])
    ax_mi.set_yticks(range(n_pha))
    ax_mi.set_yticklabels([f"{f:.1f}" for f in bpf.pha_mids])
    plt.colorbar(im1, ax=ax_mi)

    # MI per segment
    ax_seg = axes[0, 1]
    mi_seg_data = mi_per_segment[0, 0, 0, 0].cpu().detach().numpy()
    ax_seg.plot(mi_seg_data, "o-")
    ax_seg.set_xyt("Segment", "MI Value", "MI per Segment")
    ax_seg.grid(True, alpha=0.3)

    # Amplitude distribution for one phase-amplitude pair
    ax_dist = axes[1, 0]
    amp_dist = amp_distributions[0, 0, 0, 0].cpu().detach().numpy()
    phase_bin_centers = mi_result["phase_bin_centers"].cpu().numpy()
    ax_dist.bar(range(len(amp_dist)), amp_dist, alpha=0.7)
    ax_dist.set_xyt(
        "Phase Bin", "Normalized Amplitude", "Amplitude Distribution"
    )
    ax_dist.grid(True, alpha=0.3)

    # MI distribution across frequency pairs
    ax_hist = axes[1, 1]
    mi_flat = mi_values[0, 0].cpu().detach().numpy().flatten()
    ax_hist.hist(mi_flat, bins=20, alpha=0.7, color="blue")
    ax_hist.set_xyt("MI Value", "Count", "MI Distribution")
    ax_hist.grid(True, alpha=0.3)

    # fig.tight_layout()
    stx.io.save(fig, "01_modulation_index_basic.gif")
    plt.close()

    # Print results
    print(f"Input signal shape: {signal.shape}")
    print(f"Phase shape: {phase_mi.shape}")
    print(f"Amplitude shape: {amplitude_mi.shape}")
    print(f"MI values shape: {mi_values.shape}")
    print(f"MI per segment shape: {mi_per_segment.shape}")
    print(f"Gradient flow successful: {phase_grad.grad is not None}")
    print(f"Mean MI: {mi_values.mean().item():.4f}")
    print(f"Max MI: {mi_values.max().item():.4f}")


def demo_modulation_index_surrogates(args):
    """Demonstrate surrogate statistics for significance testing."""
    import scitex as stx

    stx.str.printc("=== Demo ModulationIndex Surrogates ===", c="yellow")

    # Generate smaller data for surrogate testing
    pac_config = gpac.dataset.multi_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=1,
        n_channels=2,
        n_segments=2,
        duration_sec=1,
        fs=256,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    # Create components
    bpf = gpac.core.BandPassFilter(
        fs=metadata["fs"][0],
        pha_range_hz=(6.0, 10.0),
        amp_range_hz=(40.0, 80.0),
        pha_n_bands=100,
        amp_n_bands=100,
        trainable=False,
    )

    hilbert = gpac.core.Hilbert(seq_len=signal.shape[-1])
    mi_calc = gpac.core.ModulationIndex(n_bins=12, temperature=0.01)

    if torch.cuda.is_available():
        bpf = bpf.cuda()
        hilbert = hilbert.cuda()
        mi_calc = mi_calc.cuda()
        signal = signal.cuda()

    # Process signal
    filtered = bpf(signal)
    n_pha = bpf.info["pha_n_bands"]
    n_amp = bpf.info["amp_n_bands"]

    pha_filtered = filtered[:, :, :, :n_pha, :]
    amp_filtered = filtered[:, :, :, n_pha:, :]

    batch_size, n_channels, n_segments = pha_filtered.shape[:3]

    pha_flat = pha_filtered.reshape(-1, pha_filtered.shape[-1])
    amp_flat = amp_filtered.reshape(-1, amp_filtered.shape[-1])

    pha_hilbert = hilbert(pha_flat)
    amp_hilbert = hilbert(amp_flat)

    phase = pha_hilbert[..., 0].reshape(
        batch_size, n_channels, n_segments, n_pha, -1
    )
    amplitude = amp_hilbert[..., 1].reshape(
        batch_size, n_channels, n_segments, n_amp, -1
    )

    phase_mi = phase.permute(0, 1, 3, 2, 4)
    amplitude_mi = amplitude.permute(0, 1, 3, 2, 4)

    # Compute original MI
    original_result = mi_calc(
        phase_mi,
        amplitude_mi,
        compute_distributions=False,
        return_per_segment=False,
    )
    original_mi = original_result["mi"]

    # Compute surrogates
    surrogate_result = mi_calc.compute_surrogates(
        phase_mi,
        amplitude_mi,
        n_perm=100,
        chunk_size=20,
        pac_values=original_mi,
        return_surrogates=False,
    )

    surrogate_mean = surrogate_result["surrogate_mean"]
    surrogate_std = surrogate_result["surrogate_std"]
    pac_z = surrogate_result["pac_z"]

    # Create visualization
    fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))

    # Original MI vs surrogate mean
    ax_orig = axes[0, 0]
    orig_data = original_mi[0, 0].cpu().numpy()
    surr_data = surrogate_mean[0, 0].cpu().numpy()
    im1 = ax_orig.imshow(orig_data, aspect="auto", cmap="viridis")
    ax_orig.set_xyt("Amplitude Freq", "Phase Freq", "Original MI")
    plt.colorbar(im1, ax=ax_orig)

    ax_surr = axes[0, 1]
    im2 = ax_surr.imshow(surr_data, aspect="auto", cmap="viridis")
    ax_surr.set_xyt("Amplitude Freq", "Phase Freq", "Surrogate Mean MI")
    plt.colorbar(im2, ax=ax_surr)

    # Z-scores
    ax_z = axes[1, 0]
    z_data = pac_z[0, 0].cpu().numpy()
    im3 = ax_z.imshow(z_data, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax_z.set_xyt("Amplitude Freq", "Phase Freq", "Z-scores")
    plt.colorbar(im3, ax=ax_z)

    # Comparison plot
    ax_comp = axes[1, 1]
    orig_flat = orig_data.flatten()
    surr_flat = surr_data.flatten()
    z_flat = z_data.flatten()

    ax_comp.scatter(
        orig_flat, surr_flat, c=z_flat, cmap="RdBu_r", vmin=-3, vmax=3
    )
    ax_comp.plot([0, orig_flat.max()], [0, orig_flat.max()], "k--", alpha=0.5)
    ax_comp.set_xyt(
        "Original MI", "Surrogate Mean MI", "Original vs Surrogate"
    )
    ax_comp.grid(True, alpha=0.3)

    # fig.tight_layout()
    stx.io.save(fig, "02_modulation_index_surrogates.gif")
    plt.close()

    # Print statistics
    significant_pairs = (torch.abs(pac_z) > 2).sum().item()
    total_pairs = pac_z.numel()

    print(
        f"Original MI range: {original_mi.min().item():.4f} - {original_mi.max().item():.4f}"
    )
    print(
        f"Surrogate mean range: {surrogate_mean.min().item():.4f} - {surrogate_mean.max().item():.4f}"
    )
    print(
        f"Z-score range: {pac_z.min().item():.2f} - {pac_z.max().item():.2f}"
    )
    print(f"Significant pairs (|z| > 2): {significant_pairs}/{total_pairs}")


def main(args):
    """Main function to demonstrate ModulationIndex functionality."""
    demo_modulation_index_basic(args)
    demo_modulation_index_surrogates(args)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import scitex as stx

    parser = argparse.ArgumentParser(
        description="Demonstrate ModulationIndex functionality"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=18,
        help="Number of phase bins (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Soft binning temperature (default: %(default)s)",
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
