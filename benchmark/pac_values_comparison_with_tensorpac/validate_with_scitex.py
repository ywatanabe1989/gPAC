#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-15 19:14:33 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/pac_values_comparison_with_tensorpac/validate_with_scitex.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/pac_values_comparison_with_tensorpac/validate_with_scitex.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Lightweight validation comparing gPAC and TensorPAC using SciTeX framework.
Uses smaller parameters (20x20 bands) for stability.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import scitex as stx

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add parent directory to path for imports
sys.path.insert(0, "/data/gpfs/projects/punim2354/ywatanabe/gPAC/src")
sys.path.insert(
    0, "/data/gpfs/projects/punim2354/ywatanabe/gPAC/benchmark/parameter_sweep"
)

# Import helper functions from benchmark
from _parameter_sweep_helper_init_pac_calculator import (
    _create_frequency_bands, _init_pac_calculator_gpac,
    _init_pac_calculator_tensorpac)
from gpac.dataset import (generate_pac_batch, multi_class_multi_pac_config,
                          multi_class_single_pac_config,
                          single_class_multi_pac_config,
                          single_class_single_pac_config)


def compute_pac_values(signal, pha_bands, amp_bands, n_perm=50, fs=512):
    """Compute PAC values using both gPAC and TensorPAC."""

    n_batches, n_channels, n_segments, seq_pts = signal.shape
    signal_tensorpac = signal.reshape(-1, signal.shape[-1]).cpu().numpy()

    # gPAC
    # ---------------------------------------
    pac_calculator_gpac = _init_pac_calculator_gpac(
        duration_sec=signal.shape[-1] / fs,
        fs=fs,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        n_perm=n_perm,
        fp16=False,
        device="cpu",
        device_ids="all",
        trainable=False,
    ).cpu()
    gpac_result = pac_calculator_gpac(signal)
    pac_gpac = gpac_result["pac"].cpu().numpy()
    pac_z_gpac = gpac_result["pac_z"].cpu().numpy()

    # Tensorpac
    # ---------------------------------------
    # Without Permutation
    pac_calculator_tensorpac_no_perm = _init_pac_calculator_tensorpac(
        pha_bands, amp_bands, 0
    )
    pac_tensorpac = pac_calculator_tensorpac_no_perm.filterfit(
        sf=512,
        x_pha=signal_tensorpac,
        n_perm=n_perm,
        n_jobs=1,
        random_state=42,
        verbose=False,
    ).transpose(2, 1, 0)
    pac_tensorpac = pac_tensorpac.reshape(
        n_batches,
        n_channels,
        n_segments,
        pac_gpac.shape[-2],
        pac_gpac.shape[-1],
    )

    # With Permutation
    pac_calculator_tensorpac = _init_pac_calculator_tensorpac(
        pha_bands, amp_bands, n_perm
    )
    pac_z_tensorpac = pac_calculator_tensorpac.filterfit(
        sf=512,
        x_pha=signal_tensorpac,
        n_perm=n_perm,
        n_jobs=1,
        random_state=42,
        verbose=False,
    ).transpose(2, 1, 0)
    pac_z_tensorpac = pac_z_tensorpac.reshape(
        n_batches,
        n_channels,
        n_segments,
        pac_z_gpac.shape[-2],
        pac_z_gpac.shape[-1],
    )

    assert (
        pac_gpac.shape
        == pac_z_gpac.shape
        == pac_tensorpac.shape
        == pac_z_tensorpac.shape
    )

    return {
        "pac_gpac": pac_gpac,
        "pac_z_gpac": pac_z_gpac,
        "pac_tensorpac": pac_tensorpac,
        "pac_z_tensorpac": pac_z_tensorpac,
    }


def plot_comparison(pac_results_dict, ground_truth):
    """Plot comparison using SCITEX framework."""

    i_batch, i_channel, i_segment = 0, 0, 0
    fig, axes = stx.plt.subplots(
        2, 4, figsize=(20, 10), constrained_layout=True
    )
    vmin_raw = vmax_raw = vmin_z = vmax_z = None

    titles = [
        "gPAC",
        "TensorPAC",
        "gPAC - TensorPAC",
        "gPAC (zscore)",
        "TensorPAC (zscore)",
        "gPAC - Tensorpac (zscore)",
    ]

    cols = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    rows = [
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    pac_arrays = [
        pac_results_dict["pac_gpac"],
        pac_results_dict["pac_tensorpac"],
        pac_results_dict["pac_gpac"] - pac_results_dict["pac_tensorpac"],
        pac_results_dict["pac_z_gpac"],
        pac_results_dict["pac_z_tensorpac"],
        pac_results_dict["pac_z_gpac"] - pac_results_dict["pac_z_tensorpac"],
    ]

    vmins = [vmin_raw, vmin_raw, None, vmin_z, vmin_z, None]
    vmaxs = [vmax_raw, vmax_raw, None, vmax_z, vmax_z, None]
    cmaps = ["hot", "hot", "RdBu_r", "hot", "hot", "RdBu_r"]

    for ii, (i_col, i_row, title, pac, vmin_val, vmax_val, cmap) in enumerate(
        zip(cols, rows, titles, pac_arrays, vmins, vmaxs, cmaps)
    ):

        ax = axes[i_row, i_col]

        im = ax.imshow(
            pac[i_batch, i_channel, i_segment],
            aspect="auto",
            origin="lower",
            vmin=vmin_val,
            vmax=vmax_val,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Amplitude Frequency Band")

        if ii % 3 == 0:
            ax.set_ylabel("Phase Frequency Band")

        plt.colorbar(im, ax=ax)

    # Metrics
    axes[1, 2].axis("off")

    # Compute metrics
    corr_raw, _ = pearsonr(
        pac_results_dict["pac_gpac"].flatten(),
        pac_results_dict["pac_tensorpac"].flatten(),
    )
    corr_zscore, _ = pearsonr(
        pac_results_dict["pac_z_gpac"].flatten(),
        pac_results_dict["pac_z_tensorpac"].flatten(),
    )
    ax_corr_raw = axes[0, 3]

    ax_corr_raw.scatter(
        pac_results_dict["pac_gpac"].flatten(),
        pac_results_dict["pac_tensorpac"].flatten(),
    )
    ax_corr_z = axes[1, 3]
    ax_corr_z.scatter(
        pac_results_dict["pac_z_gpac"].flatten(),
        pac_results_dict["pac_z_tensorpac"].flatten(),
    )

    mae_raw = np.mean(
        np.abs(
            pac_results_dict["pac_gpac"] - pac_results_dict["pac_tensorpac"]
        )
    )
    mae_zscore = np.mean(
        np.abs(
            pac_results_dict["pac_z_gpac"]
            - pac_results_dict["pac_z_tensorpac"]
        )
    )

    # Peak locations
    gpac_peak = np.unravel_index(
        np.argmax(pac_results_dict["pac_gpac"]),
        pac_results_dict["pac_gpac"].shape,
    )
    tensorpac_peak = np.unravel_index(
        np.argmax(pac_results_dict["pac_tensorpac"]),
        pac_results_dict["pac_tensorpac"].shape,
    )
    peak_distance = np.sqrt(
        (gpac_peak[0] - tensorpac_peak[0]) ** 2
        + (gpac_peak[1] - tensorpac_peak[1]) ** 2
    )

    metrics_text = f"""{ground_truth}

Raw PAC Metrics:
  Correlation: {corr_raw:.3f}
  MAE: {mae_raw:.4f}
  gPAC Peak: {gpac_peak}
  TensorPAC Peak: {tensorpac_peak}
  Peak Distance: {peak_distance:.2f}

Z-score Metrics:
  Correlation: {corr_zscore:.3f}
  MAE: {mae_zscore:.2f}
    """

    print(metrics_text)

    # axes[0, 2].text(
    #     0.05,
    #     0.95,
    #     metrics_text,
    #     transform=axes[0, 2].transAxes,
    #     fontsize=9,
    #     verticalalignment="top",
    #     horizontalalignment="left",
    #     fontfamily="monospace",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    # )

    plt.suptitle(f"gPAC vs TensorPAC Comparison - {ground_truth}", fontsize=14)

    metrics = {
        "corr_raw": corr_raw,
        "corr_zscore": corr_zscore,
        "mae_raw": mae_raw,
        "mae_zscore": mae_zscore,
        "peak_distance": peak_distance,
    }

    return fig, metrics


def main():
    # Configuration
    n_pha_bands = 50
    n_amp_bands = 50
    n_perm = 50  # Reasonable number of permutations
    fs = 512
    duration = 10
    n_signals_to_process = 4  # Test with a few diverse signals

    # Create frequency bands using benchmark helper
    pha_range_hz = (2.0, 20.0)
    amp_range_hz = (60.0, 160.0)
    pha_bands, amp_bands = _create_frequency_bands(
        pha_range_hz, n_pha_bands, amp_range_hz, n_amp_bands
    )

    # Generate diverse test signals using generate_pac_batch
    all_metrics = []

    for i_signal in range(n_signals_to_process):
        print(f"\nProcessing signal {i_signal+1}/{n_signals_to_process}...")

        # Generate signal with PAC using different predefined configs
        # Cycle through different PAC configurations for diversity
        pac_configs = [
            single_class_single_pac_config,
            single_class_multi_pac_config,
            multi_class_single_pac_config,
            multi_class_multi_pac_config,
        ]
        pac_config = pac_configs[i_signal % len(pac_configs)]

        signal, labels, _metadata = generate_pac_batch(
            batch_size=8,
            n_channels=1,
            n_segments=1,
            duration_sec=duration,
            fs=fs,
            pac_config=pac_config,
        )
        # To get middle of batch to avoid no-pac data
        _i_batch = 4
        signal = signal[_i_batch].unsqueeze(0)
        labels = labels[_i_batch].unsqueeze(0)
        pha_gt = _metadata["pac_components"][_i_batch][0]["pha_hz"]
        amp_gt = _metadata["pac_components"][_i_batch][0]["amp_hz"]

        ground_truth = (
            f"Ground Trugh: Phase {pha_gt} Hz, Amplitude {amp_gt} Hz"
        )

        # Compute PAC values
        pac_results_dict = compute_pac_values(
            signal, pha_bands, amp_bands, n_perm, fs
        )

        # Plot comparison
        fig, metrics = plot_comparison(pac_results_dict, ground_truth)
        stx.io.save(fig, f"comparison_signal_{i_signal+1:02d}.gif")

        all_metrics.append(metrics)

        print(f"  Raw correlation: {metrics['corr_raw']:.3f}")
        print(f"  Z-score correlation: {metrics['corr_zscore']:.3f}")

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    raw_corrs = [m["corr_raw"] for m in all_metrics]
    zscore_corrs = [m["corr_zscore"] for m in all_metrics]

    print(
        f"Raw PAC Correlation: {np.mean(raw_corrs):.3f} ± {np.std(raw_corrs):.3f}"
    )
    print(
        f"Z-score Correlation: {np.mean(zscore_corrs):.3f} ± {np.std(zscore_corrs):.3f}"
    )

    # Save summary using scitex
    stx.io.save(
        {
            "all_metrics": all_metrics,
            "summary": {
                "raw_corr_mean": np.mean(raw_corrs),
                "raw_corr_std": np.std(raw_corrs),
                "zscore_corr_mean": np.mean(zscore_corrs),
                "zscore_corr_std": np.std(zscore_corrs),
            },
        },
        "summary_metrics.pkl",
    )

    print("\nResults saved to current directory")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

# EOF
