#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 11:46:23 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/gpac/example__PAC.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example__PAC.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates PAC class functionality for phase-amplitude coupling analysis
  - Shows both static and trainable PAC computation
  - Visualizes PAC comodulograms and surrogate statistics
  - Tests gradient flow for trainable components
  - Compares performance and accuracy between modes

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, mngs

IO:
  - input-files: None (generates synthetic PAC signals)
  - output-files:
    - 01_static_pac_analysis.gif
    - 02_trainable_pac_analysis.gif
    - 03_pac_comparison.gif
"""

"""Imports"""
import argparse

import matplotlib

matplotlib.use("Agg")
import gpac
import matplotlib.pyplot as plt
import numpy as np
import torch

"""Functions & Classes"""
def demo_static_pac(args):
    """Demonstrate static PAC analysis."""
    import mngs

    mngs.str.printc("=== Demo Static PAC ===", c="yellow")

    # Generate synthetic PAC data
    pac_config = gpac.dataset.multi_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=2,
        n_channels=8,
        n_segments=4,
        duration_sec=2,
        fs=512,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    # ipdb> metadata
    # {'fs': tensor([512., 512.]), 'class_name': ['no_pac', 'single_pac'], 'noise_level': tensor([0.2000, 0.2000]), 'n_pac_components': tensor([0., 1.]), 'pac_components': [[], [{'phase_hz': 8.0, 'amp_hz': 80.0, 'strength': 0.5}]]}

    # Create static PAC
    pac_analyzer = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=4.0,
        pha_end_hz=20.0,
        pha_n_bands=100,
        amp_start_hz=30.0,
        amp_end_hz=120.0,
        amp_n_bands=100,
        trainable=False,
        n_perm=50,
        fp16=False,
    )

    if torch.cuda.is_available():
        pac_analyzer = pac_analyzer.cuda()
        signal = signal.cuda()

    # Compute PAC
    pac_result = pac_analyzer(signal)

    # Create visualization with ground truth
    fig, axes = mngs.plt.subplots(2, 3, figsize=(18, 10))

    sample_idx, channel_idx = 0, 0

    # Ground truth PAC
    ax_gt = axes[0, 0]
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        # Extract ground truth info
        gt_info = metadata["pac_coupling"][sample_idx]
        pha_freqs = pac_analyzer.pha_mids.cpu().numpy()
        amp_freqs = pac_analyzer.amp_mids.cpu().numpy()

        # Create ground truth matrix
        gt_matrix = np.zeros((len(pha_freqs), len(amp_freqs)))
        for coupling in gt_info:
            pha_hz = coupling["pha_hz"]
            amp_hz = coupling["amp_hz"]
            strength = coupling.get("strength", 1.0)

            # Find closest frequency indices
            pha_idx = np.argmin(np.abs(pha_freqs - pha_hz))
            amp_idx = np.argmin(np.abs(amp_freqs - amp_hz))
            gt_matrix[pha_idx, amp_idx] = strength

        im_gt = ax_gt.imshow(
            gt_matrix, aspect="auto", cmap="viridis", origin="lower"
        )
        ax_gt.set_xyt(
            "Amplitude Frequency", "Phase Frequency", "Ground Truth PAC"
        )
        ax_gt.set_xticks(range(0, len(amp_freqs), 20))
        ax_gt.set_xticklabels([f"{f:.0f}" for f in amp_freqs[::20]])
        ax_gt.set_yticks(range(0, len(pha_freqs), 20))
        ax_gt.set_yticklabels([f"{f:.0f}" for f in pha_freqs[::20]])
        plt.colorbar(im_gt, ax=ax_gt)
    else:
        ax_gt.text(
            0.5,
            0.5,
            "No Ground Truth",
            ha="center",
            va="center",
            transform=ax_gt.transAxes,
        )
        ax_gt.set_title("Ground Truth PAC")

    # Detected PAC comodulogram
    ax_pac = axes[0, 1]
    pac_data = pac_result["pac"][sample_idx, channel_idx].cpu().numpy()
    im1 = ax_pac.imshow(
        pac_data, aspect="auto", cmap="viridis", origin="lower"
    )
    ax_pac.set_xyt("Amplitude Frequency", "Phase Frequency", "Detected PAC")
    ax_pac.set_xticks(range(0, len(pac_analyzer.amp_mids), 20))
    ax_pac.set_xticklabels([f"{f:.0f}" for f in pac_analyzer.amp_mids[::20]])
    ax_pac.set_yticks(range(0, len(pac_analyzer.pha_mids), 20))
    ax_pac.set_yticklabels([f"{f:.0f}" for f in pac_analyzer.pha_mids[::20]])
    plt.colorbar(im1, ax=ax_pac)

    # Z-scores
    ax_z = axes[0, 2]
    if pac_result["pac_z"] is not None:
        z_data = pac_result["pac_z"][sample_idx, channel_idx].cpu().numpy()
        im2 = ax_z.imshow(
            z_data,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            origin="lower",
        )
        ax_z.set_xyt("Amplitude Frequency", "Phase Frequency", "Z-scores")
        ax_z.set_xticks(range(0, len(pac_analyzer.amp_mids), 20))
        ax_z.set_xticklabels([f"{f:.0f}" for f in pac_analyzer.amp_mids[::20]])
        ax_z.set_yticks(range(0, len(pac_analyzer.pha_mids), 20))
        ax_z.set_yticklabels([f"{f:.0f}" for f in pac_analyzer.pha_mids[::20]])
        plt.colorbar(im2, ax=ax_z)

    # Frequency profiles
    ax_pha = axes[1, 0]
    pac_mean_amp = pac_data.mean(axis=1)
    ax_pha.plot(
        pac_analyzer.pha_mids.cpu(), pac_mean_amp, "o-", label="Detected"
    )
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        gt_pha_freqs = [
            c["pha_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]
        gt_strengths = [
            c.get("strength", 1.0)
            for c in metadata["pac_coupling"][sample_idx]
        ]
        ax_pha.scatter(
            gt_pha_freqs,
            gt_strengths,
            color="red",
            s=100,
            marker="x",
            label="Ground Truth",
        )
    ax_pha.set_xyt(
        "Phase Frequency [Hz]", "PAC Strength", "Phase Frequency Profile"
    )
    ax_pha.legend()
    ax_pha.grid(True, alpha=0.3)

    ax_amp = axes[1, 1]
    pac_mean_pha = pac_data.mean(axis=0)
    ax_amp.plot(
        pac_analyzer.amp_mids.cpu(), pac_mean_pha, "s-", label="Detected"
    )
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        gt_amp_freqs = [
            c["amp_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]
        gt_strengths = [
            c.get("strength", 1.0)
            for c in metadata["pac_coupling"][sample_idx]
        ]
        ax_amp.scatter(
            gt_amp_freqs,
            gt_strengths,
            color="red",
            s=100,
            marker="x",
            label="Ground Truth",
        )
    ax_amp.set_xyt(
        "Amplitude Frequency [Hz]",
        "PAC Strength",
        "Amplitude Frequency Profile",
    )
    ax_amp.legend()
    ax_amp.grid(True, alpha=0.3)

    # Ground truth vs detected scatter
    ax_scatter = axes[1, 2]
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        for coupling in metadata["pac_coupling"][sample_idx]:
            pha_hz = coupling["pha_hz"]
            amp_hz = coupling["amp_hz"]
            gt_strength = coupling.get("strength", 1.0)

            # Find closest detected PAC value
            pha_idx = np.argmin(
                np.abs(pac_analyzer.pha_mids.cpu().numpy() - pha_hz)
            )
            amp_idx = np.argmin(
                np.abs(pac_analyzer.amp_mids.cpu().numpy() - amp_hz)
            )
            detected_strength = pac_data[pha_idx, amp_idx]

            ax_scatter.scatter(
                gt_strength, detected_strength, s=100, alpha=0.7
            )
            ax_scatter.text(
                gt_strength,
                detected_strength,
                f"({pha_hz:.0f},{amp_hz:.0f})",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        # Add diagonal line
        max_val = max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
        ax_scatter.plot([0, max_val], [0, max_val], "k--", alpha=0.5)
        ax_scatter.set_xyt(
            "Ground Truth PAC", "Detected PAC", "GT vs Detected"
        )
        ax_scatter.grid(True, alpha=0.3)
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No Ground Truth",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
        )

    plt.tight_layout()
    mngs.io.save(fig, "01_static_pac_analysis.gif")
    plt.close()

    # Print statistics with ground truth comparison
    print(f"Input shape: {signal.shape}")
    print(f"PAC shape: {pac_result['pac'].shape}")
    print(f"Mean PAC: {pac_result['pac'].mean().item():.4f}")
    print(f"Max PAC: {pac_result['pac'].max().item():.4f}")
    if pac_result["pac_z"] is not None:
        significant_pairs = (torch.abs(pac_result["pac_z"]) > 2).sum().item()
        total_pairs = pac_result["pac_z"].numel()
        print(
            f"Significant pairs (|z| > 2): {significant_pairs}/{total_pairs}"
        )

    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        print(f"Ground truth couplings for sample {sample_idx}:")
        for coupling in metadata["pac_coupling"][sample_idx]:
            print(
                f"  {coupling['pha_hz']:.1f} Hz -> {coupling['amp_hz']:.1f} Hz (strength: {coupling.get('strength', 1.0):.3f})"
            )


def demo_trainable_pac(args):
    """Demonstrate trainable PAC analysis with optimization."""
    import mngs

    mngs.str.printc("=== Demo Trainable PAC ===", c="yellow")

    # Generate synthetic PAC data
    pac_config = gpac.dataset.multi_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=3,
        n_channels=6,
        n_segments=3,
        duration_sec=1.5,
        fs=256,
        pac_config=pac_config,
    )
    signal, label, metadata = batch
    # ipdb> metadata
    # {'fs': tensor([256., 256., 256.]), 'class_name': ['no_pac', 'dual_pac', 'single_pac'], 'noise_level': tensor([0.2000, 0.2000, 0.2000]), 'n_pac_components': tensor([0., 2., 1.]), 'pac_components': [[], [{'phase_hz': 8.0, 'amp_hz': 80.0, 'strength': 0.4}, {'phase_hz': 12.0, 'amp_hz': 120.0, 'strength': 0.3}], [{'phase_hz': 8.0, 'amp_hz': 80.0, 'strength': 0.5}]]}
    # ipdb>

    # Create trainable PAC
    pac_analyzer = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=6.0,
        pha_end_hz=15.0,
        pha_n_bands=100,
        amp_start_hz=40.0,
        amp_end_hz=100.0,
        amp_n_bands=100,
        trainable=True,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        temperature=1.0,
        hard_selection=False,
        n_perm=None,
        fp16=False,
    )

    if torch.cuda.is_available():
        pac_analyzer = pac_analyzer.cuda()
        signal = signal.cuda()

    signal = signal.requires_grad_(False)
    optimizer = torch.optim.Adam(pac_analyzer.parameters(), lr=0.01)

    initial_freqs = pac_analyzer.get_selected_frequencies()
    print(f"Initial selected frequencies:")
    print(f"  Phase: {initial_freqs[0]}")
    print(f"  Amplitude: {initial_freqs[1]}")

    trainable_params = list(pac_analyzer.parameters())
    print(f"Trainable parameters: {len(trainable_params)}")
    for param_idx, param in enumerate(trainable_params):
        print(
            f"  Param {param_idx}: {param.shape}, requires_grad={param.requires_grad}"
        )

    # Optimization loop
    losses = []
    pac_analyzer.train()

    for epoch_idx in range(10):
        optimizer.zero_grad()
        pac_result = pac_analyzer(signal)
        loss = -pac_result["pac"].mean()
        reg_loss = pac_analyzer.get_filter_regularization_loss()
        total_loss = loss + 0.1 * reg_loss

        print(
            f"Epoch {epoch_idx}: loss={loss.item():.4f}, reg={reg_loss.item():.4f}, requires_grad={total_loss.requires_grad}"
        )

        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        else:
            print("Warning: Loss does not require gradients")

        losses.append(total_loss.item())

        if epoch_idx % 3 == 0:
            print(f"Epoch {epoch_idx}: Loss = {total_loss.item():.4f}")

    final_freqs = pac_analyzer.get_selected_frequencies()
    print(f"Final selected frequencies:")
    print(f"  Phase: {final_freqs[0]}")
    print(f"  Amplitude: {final_freqs[1]}")

    # Final PAC computation
    pac_analyzer.eval()
    with torch.no_grad():
        final_pac_result = pac_analyzer(signal)

    # Create visualization with ground truth
    fig, axes = mngs.plt.subplots(2, 3, figsize=(18, 10))

    sample_idx, channel_idx = 0, 0

    # Ground truth PAC
    ax_gt = axes[0, 0]
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        gt_info = metadata["pac_coupling"][sample_idx]

        # Create simplified ground truth visualization for trainable case
        gt_pha_freqs = [c["pha_hz"] for c in gt_info]
        gt_amp_freqs = [c["amp_hz"] for c in gt_info]
        gt_strengths = [c.get("strength", 1.0) for c in gt_info]

        # Scatter plot for ground truth
        for pha_hz, amp_hz, strength in zip(
            gt_pha_freqs, gt_amp_freqs, gt_strengths
        ):
            ax_gt.scatter(amp_hz, pha_hz, s=strength * 500, alpha=0.7, c="red")
            ax_gt.text(
                amp_hz,
                pha_hz,
                f"{strength:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

        ax_gt.set_xyt(
            "Amplitude Frequency [Hz]",
            "Phase Frequency [Hz]",
            "Ground Truth PAC",
        )
        ax_gt.set_xlim(40, 100)
        ax_gt.set_ylim(6, 15)
        ax_gt.grid(True, alpha=0.3)
    else:
        ax_gt.text(
            0.5,
            0.5,
            "No Ground Truth",
            ha="center",
            va="center",
            transform=ax_gt.transAxes,
        )
        ax_gt.set_title("Ground Truth PAC")

    # Detected PAC comodulogram
    ax_pac = axes[0, 1]
    pac_data = final_pac_result["pac"][sample_idx, channel_idx].cpu().numpy()
    im1 = ax_pac.imshow(
        pac_data, aspect="auto", cmap="viridis", origin="lower"
    )
    ax_pac.set_xyt("Amplitude Filter", "Phase Filter", "Trainable PAC")
    plt.colorbar(im1, ax=ax_pac)

    # Training loss
    ax_loss = axes[0, 2]
    ax_loss.plot(losses, "o-")
    ax_loss.set_xyt("Epoch", "Loss", "Training Loss")
    ax_loss.grid(True, alpha=0.3)

    # Selected frequency comparison with ground truth
    ax_freq = axes[1, 0]
    if initial_freqs[0] and final_freqs[0]:
        ax_freq.scatter(
            initial_freqs[0],
            [1] * len(initial_freqs[0]),
            label="Initial Phase",
            alpha=0.7,
            s=50,
        )
        ax_freq.scatter(
            final_freqs[0],
            [1.1] * len(final_freqs[0]),
            label="Final Phase",
            alpha=0.7,
            s=50,
        )
        ax_freq.scatter(
            initial_freqs[1],
            [2] * len(initial_freqs[1]),
            label="Initial Amplitude",
            alpha=0.7,
            s=50,
        )
        ax_freq.scatter(
            final_freqs[1],
            [2.1] * len(final_freqs[1]),
            label="Final Amplitude",
            alpha=0.7,
            s=50,
        )

    # Add ground truth frequencies
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        gt_pha_freqs = [
            c["pha_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]
        gt_amp_freqs = [
            c["amp_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]
        ax_freq.scatter(
            gt_pha_freqs,
            [0.9] * len(gt_pha_freqs),
            label="GT Phase",
            alpha=0.9,
            s=100,
            marker="x",
            color="red",
        )
        ax_freq.scatter(
            gt_amp_freqs,
            [1.9] * len(gt_amp_freqs),
            label="GT Amplitude",
            alpha=0.9,
            s=100,
            marker="x",
            color="red",
        )

    ax_freq.set_xyt("Frequency [Hz]", "Filter Type", "Selected Frequencies")
    ax_freq.set_yticks([0.9, 1, 1.1, 1.9, 2, 2.1])
    ax_freq.set_yticklabels(
        [
            "GT Phase",
            "Init Phase",
            "Final Phase",
            "GT Amp",
            "Init Amp",
            "Final Amp",
        ]
    )
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)

    # PAC distribution
    ax_dist = axes[1, 1]
    pac_flat = pac_data.flatten()
    ax_dist.hist(pac_flat, bins=15, alpha=0.7, color="green")
    ax_dist.set_xyt("PAC Value", "Count", "PAC Distribution")
    ax_dist.grid(True, alpha=0.3)

    # Ground truth vs final frequencies accuracy
    ax_accuracy = axes[1, 2]
    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        gt_pha_freqs = [
            c["pha_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]
        gt_amp_freqs = [
            c["amp_hz"] for c in metadata["pac_coupling"][sample_idx]
        ]

        # Calculate distances to ground truth
        pha_distances = []
        amp_distances = []

        for gt_pha in gt_pha_freqs:
            if final_freqs[0]:
                min_dist = min([abs(f - gt_pha) for f in final_freqs[0]])
                pha_distances.append(min_dist)

        for gt_amp in gt_amp_freqs:
            if final_freqs[1]:
                min_dist = min([abs(f - gt_amp) for f in final_freqs[1]])
                amp_distances.append(min_dist)

        if pha_distances and amp_distances:
            categories = ["Phase", "Amplitude"]
            avg_distances = [np.mean(pha_distances), np.mean(amp_distances)]
            ax_accuracy.bar(
                categories, avg_distances, alpha=0.7, color=["blue", "orange"]
            )
            ax_accuracy.set_xyt(
                "Frequency Type", "Avg Distance [Hz]", "GT Accuracy"
            )
            ax_accuracy.grid(True, alpha=0.3)
        else:
            ax_accuracy.text(
                0.5,
                0.5,
                "No accuracy calc",
                ha="center",
                va="center",
                transform=ax_accuracy.transAxes,
            )
    else:
        ax_accuracy.text(
            0.5,
            0.5,
            "No Ground Truth",
            ha="center",
            va="center",
            transform=ax_accuracy.transAxes,
        )

    plt.tight_layout()
    mngs.io.save(fig, "02_trainable_pac_analysis.gif")
    plt.close()

    print(f"Training completed with final loss: {losses[-1]:.4f}")
    print(f"Final PAC mean: {final_pac_result['pac'].mean().item():.4f}")

    if "pac_coupling" in metadata and len(metadata["pac_coupling"]) > 0:
        print(f"Ground truth couplings for sample {sample_idx}:")
        for coupling in metadata["pac_coupling"][sample_idx]:
            print(
                f"  {coupling['pha_hz']:.1f} Hz -> {coupling['amp_hz']:.1f} Hz (strength: {coupling.get('strength', 1.0):.3f})"
            )


def demo_pac_comparison(args):
    """Compare static vs trainable PAC performance."""
    import mngs

    mngs.str.printc("=== Demo PAC Comparison ===", c="yellow")

    # Generate test data
    pac_config = gpac.dataset.multi_class_multi_pac_config
    batch = gpac.dataset.generate_pac_batch(
        batch_size=2,
        n_channels=4,
        n_segments=2,
        duration_sec=1,
        fs=256,
        pac_config=pac_config,
    )
    signal, label, metadata = batch

    if torch.cuda.is_available():
        signal = signal.cuda()

    # Static PAC
    static_pac = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=6.0,
        pha_end_hz=12.0,
        pha_n_bands=100,
        amp_start_hz=50.0,
        amp_end_hz=90.0,
        amp_n_bands=100,
        trainable=False,
        n_perm=None,
        fp16=False,
    )

    # Trainable PAC
    trainable_pac = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=metadata["fs"][0],
        pha_start_hz=6.0,
        pha_end_hz=12.0,
        pha_n_bands=100,
        amp_start_hz=50.0,
        amp_end_hz=90.0,
        amp_n_bands=100,
        trainable=True,
        pha_n_pool_ratio=1.5,
        amp_n_pool_ratio=1.5,
        n_perm=None,
        fp16=False,
    )

    if torch.cuda.is_available():
        static_pac = static_pac.cuda()
        trainable_pac = trainable_pac.cuda()

    # Compute PAC with timing
    import time

    # Static PAC timing
    start_time = time.time()
    static_result = static_pac(signal)
    static_time = time.time() - start_time

    # Trainable PAC timing
    start_time = time.time()
    trainable_result = trainable_pac(signal)
    trainable_time = time.time() - start_time

    # Create comparison visualization
    fig, axes = mngs.plt.subplots(2, 3, figsize=(15, 10))

    sample_idx, channel_idx = 0, 0

    # Static PAC
    ax_static = axes[0, 0]
    static_data = static_result["pac"][sample_idx, channel_idx].cpu().numpy()
    im1 = ax_static.imshow(
        static_data, aspect="auto", cmap="viridis", origin="lower"
    )
    ax_static.set_xyt("Amplitude", "Phase", "Static PAC")
    plt.colorbar(im1, ax=ax_static)

    # Trainable PAC
    ax_trainable = axes[0, 1]
    trainable_data = (
        trainable_result["pac"][sample_idx, channel_idx].cpu().numpy()
    )
    im2 = ax_trainable.imshow(
        trainable_data, aspect="auto", cmap="viridis", origin="lower"
    )
    ax_trainable.set_xyt("Amplitude", "Phase", "Trainable PAC")
    plt.colorbar(im2, ax=ax_trainable)

    # Difference
    ax_diff = axes[0, 2]
    diff_data = trainable_data - static_data
    im3 = ax_diff.imshow(
        diff_data, aspect="auto", cmap="RdBu_r", origin="lower"
    )
    ax_diff.set_xyt("Amplitude", "Phase", "Difference")
    plt.colorbar(im3, ax=ax_diff)

    # Statistics comparison
    ax_stats = axes[1, 0]
    methods = ["Static", "Trainable"]
    means = [static_data.mean(), trainable_data.mean()]
    maxs = [static_data.max(), trainable_data.max()]
    stds = [static_data.std(), trainable_data.std()]

    xx_pos = np.arange(len(methods))
    width = 0.25

    ax_stats.bar(xx_pos - width, means, width, label="Mean", alpha=0.7)
    ax_stats.bar(xx_pos, maxs, width, label="Max", alpha=0.7)
    ax_stats.bar(xx_pos + width, stds, width, label="Std", alpha=0.7)
    ax_stats.set_xyt("Method", "PAC Value", "Statistics Comparison")
    ax_stats.set_xticks(xx_pos)
    ax_stats.set_xticklabels(methods)
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)

    # Timing comparison
    ax_time = axes[1, 1]
    times = [static_time * 1000, trainable_time * 1000]
    ax_time.bar(methods, times, alpha=0.7, color=["blue", "red"])
    ax_time.set_xyt("Method", "Time [ms]", "Computation Time")
    ax_time.grid(True, alpha=0.3)

    # Memory info
    ax_memory = axes[1, 2]
    static_memory = static_pac.get_memory_info()
    trainable_memory = trainable_pac.get_memory_info()

    memory_data = [
        static_memory["device_details"][0]["allocated_gb"],
        trainable_memory["device_details"][0]["allocated_gb"],
    ]
    ax_memory.bar(methods, memory_data, alpha=0.7, color=["green", "orange"])
    ax_memory.set_xyt("Method", "Memory [GB]", "GPU Memory Usage")
    ax_memory.grid(True, alpha=0.3)

    plt.tight_layout()
    mngs.io.save(fig, "03_pac_comparison.gif")
    plt.close()

    # Print comparison results
    print(f"=== Comparison Results ===")
    print(f"Static PAC:")
    print(f"  - Computation time: {static_time*1000:.2f} ms")
    print(f"  - Mean PAC: {static_data.mean():.4f}")
    print(f"  - Max PAC: {static_data.max():.4f}")
    print(f"Trainable PAC:")
    print(f"  - Computation time: {trainable_time*1000:.2f} ms")
    print(f"  - Mean PAC: {trainable_data.mean():.4f}")
    print(f"  - Max PAC: {trainable_data.max():.4f}")
    print(f"Performance ratio: {trainable_time/static_time:.2f}x")


def main(args):
    """Main function to demonstrate PAC functionality."""
    demo_static_pac(args)
    demo_trainable_pac(args)
    demo_pac_comparison(args)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(
        description="Demonstrate PAC functionality"
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=50,
        help="Number of permutations for surrogate testing (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing (default: %(default)s)",
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
