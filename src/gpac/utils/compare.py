#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 00:28:48 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/utils/compare.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/utils/compare.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Comparison utilities for fair benchmarking between gPAC and TensorPAC.

This module provides functions for:
- Shape verification and handling
- Frequency band extraction and matching
- Correlation and error metrics
- Performance benchmarking
"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

"""Shape Verification Utilities"""


def verify_input_shape_gpac(
    signal: torch.Tensor, expected_dims: List[int] = [3, 4]
) -> bool:
    """
    Verify gPAC input shape is correct.

    Parameters
    ----------
    signal : torch.Tensor
        Input signal to verify
    expected_dims : List[int]
        Expected number of dimensions (3 or 4)

    Expected shapes:
    - 3D: (n_batches, n_channels, n_times)
    - 4D: (n_batches, n_channels, n_epochs, n_times)

    Returns
    -------
    bool
        True if shape is valid

    Raises
    ------
    ValueError
        If shape is invalid
    """
    if signal.dim() not in expected_dims:
        raise ValueError(
            f"gPAC expects {expected_dims}D input, got {signal.dim()}D. "
            f"Shape should be (batch, channel, time) or (batch, channel, epoch, time)"
        )
    return True


def verify_input_shape_tensorpac(signal: np.ndarray) -> bool:
    """
    Verify TensorPAC input shape is correct.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to verify

    Expected shape: (n_epochs, n_times)

    Returns
    -------
    bool
        True if shape is valid

    Raises
    ------
    ValueError
        If shape is invalid
    """
    if signal.ndim != 2:
        raise ValueError(
            f"TensorPAC expects 2D input (n_epochs, n_times), got {signal.ndim}D"
        )
    return True


def verify_output_shapes_match(
    pac_gp: np.ndarray, pac_tp: np.ndarray, verbose: bool = True
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Verify and potentially fix shape mismatch between outputs.

    Parameters
    ----------
    pac_gp : np.ndarray
        gPAC output
    pac_tp : np.ndarray
        TensorPAC output
    verbose : bool
        Print mismatch information

    Returns
    -------
    tuple
        (shapes_match, corrected_pac_tp)
    """
    if pac_gp.shape == pac_tp.shape:
        return True, pac_tp

    if verbose:
        print(f"⚠️  Shape mismatch: gPAC {pac_gp.shape} vs TensorPAC {pac_tp.shape}")

    # Try transposing TensorPAC output
    if pac_gp.shape == pac_tp.T.shape:
        if verbose:
            print("  ✅ Fixed by transposing TensorPAC output")
        return True, pac_tp.T

    # Check if it's a dimension order issue
    if sorted(pac_gp.shape) == sorted(pac_tp.shape):
        if verbose:
            print("  ⚠️  Same dimensions but different order")
        return False, pac_tp

    return False, pac_tp


"""Band Extraction and Verification"""


def extract_gpac_bands(pac_gp) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract band configuration from gPAC object.

    Parameters
    ----------
    pac_gp : PAC
        gPAC PAC object

    Returns
    -------
    tuple
        (pha_bands, amp_bands) as numpy arrays
    """
    # Access the bandpass filter
    bandpass = pac_gp.bandpass

    # Get bands from the filter
    pha_bands = bandpass.pha_bands_hz.cpu().numpy()
    amp_bands = bandpass.amp_bands_hz.cpu().numpy()

    return pha_bands, amp_bands


def verify_band_ranges(
    bands: np.ndarray,
    expected_pha_range: Tuple[float, float],
    expected_amp_range: Tuple[float, float],
    tolerance: float = 1.0,
    is_phase: bool = True,
) -> Dict[str, bool]:
    """
    Verify band ranges match expectations.

    Parameters
    ----------
    bands : np.ndarray
        Frequency bands to verify
    expected_pha_range : tuple
        Expected phase frequency range
    expected_amp_range : tuple
        Expected amplitude frequency range
    tolerance : float
        Tolerance for range matching
    is_phase : bool
        Whether these are phase bands

    Returns
    -------
    dict
        Verification results
    """
    results = {}

    # Get actual range
    actual_range = (bands.min(), bands.max())
    expected_range = expected_pha_range if is_phase else expected_amp_range

    results["match"] = (
        abs(actual_range[0] - expected_range[0]) < tolerance
        and abs(actual_range[1] - expected_range[1]) < tolerance
    )

    results["actual_range"] = actual_range
    results["expected_range"] = expected_range

    return results


def check_band_spacing(bands: np.ndarray, tolerance: float = 0.1) -> bool:
    """
    Check if band spacing is linear or non-linear.

    Parameters
    ----------
    bands : np.ndarray
        Frequency bands
    tolerance : float
        Relative tolerance for linearity check

    Returns
    -------
    bool
        True if spacing is linear
    """
    # Handle edge cases
    if bands.size == 0 or len(bands) <= 1:
        return True  # Trivially linear for empty or single band

    # Calculate centers from bands
    centers = np.mean(bands, axis=1)

    if len(centers) <= 2:
        return True  # Linear by definition for 2 or fewer points

    spacing = np.diff(centers)
    is_linear = np.allclose(spacing, spacing[0], rtol=tolerance)

    return is_linear


"""Data Preparation Utilities"""


def prepare_signal_gpac(
    signal: np.ndarray,
    batch_size: int = 1,
    n_channels: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Prepare signal for gPAC input.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    batch_size : int
        Batch size
    n_channels : int
        Number of channels
    device : str
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Prepared signal for gPAC
    """
    if signal.ndim == 1:
        # 1D signal -> (batch, channel, time)
        signal_torch = torch.from_numpy(signal).float()
        signal_torch = signal_torch.unsqueeze(0).unsqueeze(0)
    elif signal.ndim == 2:
        # 2D signal (channels, time) -> (batch, channels, time)
        signal_torch = torch.from_numpy(signal).float()
        signal_torch = signal_torch.unsqueeze(0)
    else:
        signal_torch = torch.from_numpy(signal).float()

    # Verify shape
    verify_input_shape_gpac(signal_torch)

    return signal_torch.to(device)


def prepare_signal_tensorpac(signal: np.ndarray) -> np.ndarray:
    """
    Prepare signal for TensorPAC input.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Prepared signal for TensorPAC
    """
    if signal.ndim == 1:
        # 1D signal -> (1 epoch, n_times)
        signal_2d = signal.reshape(1, -1)
    elif signal.ndim == 2:
        signal_2d = signal
    else:
        raise ValueError(f"TensorPAC expects 1D or 2D signal, got {signal.ndim}D")

    # Verify shape
    verify_input_shape_tensorpac(signal_2d)

    return signal_2d


"""Metric Computation Utilities"""


def compute_correlation_metrics(pac1: np.ndarray, pac2: np.ndarray) -> Dict[str, float]:
    """
    Compute various correlation metrics between PAC matrices.

    Parameters
    ----------
    pac1 : np.ndarray
        First PAC matrix (typically gPAC)
    pac2 : np.ndarray
        Second PAC matrix (typically TensorPAC)

    Returns
    -------
    dict
        Correlation metrics including Pearson and Spearman correlations
    """
    from scipy.stats import pearsonr, spearmanr

    # Convert to numpy if needed
    if torch.is_tensor(pac1):
        pac1 = pac1.cpu().numpy()
    if torch.is_tensor(pac2):
        pac2 = pac2.cpu().numpy()

    # Handle shape mismatch
    # gPAC: (batch, channels, n_pha, n_amp)
    # TensorPAC: (n_pha, n_amp, n_channels) or (n_pha, n_amp)
    if pac1.ndim == 4 and pac2.ndim == 3:
        # Average gPAC over batch dimension
        pac1 = pac1.mean(axis=0)  # (channels, n_pha, n_amp)
        # Transpose to match TensorPAC
        pac1 = pac1.transpose(1, 2, 0)  # (n_pha, n_amp, channels)
    elif pac1.ndim == 4 and pac2.ndim == 2:
        # Average gPAC over batch and channels
        pac1 = pac1.mean(axis=(0, 1))  # (n_pha, n_amp)

    # Ensure same shape
    if pac1.shape != pac2.shape:
        # Try transposing
        if pac1.shape == pac2.T.shape:
            pac2 = pac2.T
        else:
            raise ValueError(f"Cannot reconcile shapes: {pac1.shape} vs {pac2.shape}")

    # Flatten arrays
    pac1_flat = pac1.flatten()
    pac2_flat = pac2.flatten()

    # Raw correlation
    pearson_r, pearson_p = pearsonr(pac1_flat, pac2_flat)
    spearman_r, spearman_p = spearmanr(pac1_flat, pac2_flat)

    # Normalized correlation
    pac1_norm = pac1 / pac1.max() if pac1.max() > 0 else pac1
    pac2_norm = pac2 / pac2.max() if pac2.max() > 0 else pac2
    pearson_norm_r, _ = pearsonr(pac1_norm.flatten(), pac2_norm.flatten())

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_normalized": pearson_norm_r,
    }


def compute_error_metrics(pac1: np.ndarray, pac2: np.ndarray) -> Dict[str, float]:
    """
    Compute error metrics between PAC matrices.

    Parameters
    ----------
    pac1 : np.ndarray
        First PAC matrix (typically gPAC)
    pac2 : np.ndarray
        Second PAC matrix (typically TensorPAC)

    Returns
    -------
    dict
        Error metrics including MAE, MSE, RMSE
    """
    # Convert to numpy if needed
    if torch.is_tensor(pac1):
        pac1 = pac1.cpu().numpy()
    if torch.is_tensor(pac2):
        pac2 = pac2.cpu().numpy()

    # Handle shape mismatch
    # gPAC: (batch, channels, n_pha, n_amp)
    # TensorPAC: (n_pha, n_amp, n_channels) or (n_pha, n_amp)
    if pac1.ndim == 4 and pac2.ndim == 3:
        # Average gPAC over batch dimension
        pac1 = pac1.mean(axis=0)  # (channels, n_pha, n_amp)
        # Transpose to match TensorPAC
        pac1 = pac1.transpose(1, 2, 0)  # (n_pha, n_amp, channels)
    elif pac1.ndim == 4 and pac2.ndim == 2:
        # Average gPAC over batch and channels
        pac1 = pac1.mean(axis=(0, 1))  # (n_pha, n_amp)

    # Ensure same shape
    if pac1.shape != pac2.shape:
        # Try transposing
        if pac1.shape == pac2.T.shape:
            pac2 = pac2.T
        else:
            raise ValueError(f"Cannot reconcile shapes: {pac1.shape} vs {pac2.shape}")

    # Scale factor
    scale_factor = pac2.max() / pac1.max() if pac1.max() > 0 else np.inf

    # Raw errors
    diff = pac1 - pac2
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(diff))

    # Normalized errors
    pac1_norm = pac1 / pac1.max() if pac1.max() > 0 else pac1
    pac2_norm = pac2 / pac2.max() if pac2.max() > 0 else pac2
    mae_norm = np.mean(np.abs(pac1_norm - pac2_norm))
    rmse_norm = np.sqrt(np.mean((pac1_norm - pac2_norm) ** 2))

    return {
        "scale_factor": scale_factor,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "mae_normalized": mae_norm,
        "rmse_normalized": rmse_norm,
    }


"""Reporting Utilities"""


def print_shape_report(
    signal: np.ndarray, pac_gp: np.ndarray, pac_tp: np.ndarray, file=None
):
    """
    Print comprehensive shape report.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    pac_gp : np.ndarray
        gPAC output
    pac_tp : np.ndarray
        TensorPAC output
    file : file object
        Output file (default: stdout)
    """
    f = file or sys.stdout

    f.write("\n" + "=" * 60 + "\n")
    f.write("SHAPE VERIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"\nInput signal: {signal.shape}\n")
    f.write(f"\nOutput shapes:\n")
    f.write(f"  gPAC: {pac_gp.shape} (batch, channels, n_pha, n_amp)\n")
    f.write(f"  TensorPAC: {pac_tp.shape}\n")
    f.write(f"  Match: {'✅ YES' if pac_gp.shape[-2:] == pac_tp.shape else '❌ NO'}\n")


def print_band_report(pha_bands: np.ndarray, amp_bands: np.ndarray, file=None):
    """
    Print detailed band report.

    Parameters
    ----------
    pha_bands : np.ndarray
        Phase frequency bands
    amp_bands : np.ndarray
        Amplitude frequency bands
    file : file object
        Output file (default: stdout)
    """
    f = file or sys.stdout

    f.write("\nBand Configuration:\n")
    f.write("Phase bands:\n")
    for i, (low, high) in enumerate(pha_bands):
        f.write(
            f"  Band {i}: {low:.1f} - {high:.1f} Hz (center: {(low+high)/2:.1f} Hz)\n"
        )

    f.write("\nAmplitude bands:\n")
    for i, (low, high) in enumerate(amp_bands):
        f.write(
            f"  Band {i}: {low:.1f} - {high:.1f} Hz (center: {(low+high)/2:.1f} Hz)\n"
        )


def print_comparison_summary(
    corr_metrics: Dict[str, float], error_metrics: Dict[str, float], file=None
):
    """
    Print comparison summary with metrics.

    Parameters
    ----------
    corr_metrics : dict
        Correlation metrics
    error_metrics : dict
        Error metrics
    file : file object
        Output file (default: stdout)
    """
    f = file or sys.stdout

    f.write("Comparison Metrics:\n")
    f.write(f"  Pearson correlation: {corr_metrics['pearson_r']:.4f}\n")
    f.write(f"  Spearman correlation: {corr_metrics['spearman_r']:.4f}\n")
    f.write(f"  MAE: {error_metrics['mae']:.6f}\n")
    f.write(f"  MSE: {error_metrics['mse']:.6f}\n")
    f.write(f"  RMSE: {error_metrics['rmse']:.6f}\n")
    f.write(f"  Max absolute error: {error_metrics.get('max_error', 0):.6f}\n")


# Convenience functions for quick comparisons
def quick_compare(pac_gp_result, pac_tp_result, verbose=True):
    """
    Quick comparison between gPAC and TensorPAC results.

    Parameters
    ----------
    pac_gp_result : dict or tensor
        gPAC output (if dict, extracts 'pac' key)
    pac_tp_result : np.ndarray
        TensorPAC output
    verbose : bool
        Print comparison results

    Returns
    -------
    dict
        Comparison metrics
    """
    # Extract PAC values if needed
    if isinstance(pac_gp_result, dict):
        pac_gp = pac_gp_result["pac"]
    else:
        pac_gp = pac_gp_result

    # Compute metrics
    corr_metrics = compute_correlation_metrics(pac_gp, pac_tp_result)
    error_metrics = compute_error_metrics(pac_gp, pac_tp_result)

    if verbose:
        print("\nQuick Comparison Results:")
        print(f"  Correlation: {corr_metrics['pearson_r']:.4f}")
        print(f"  Scale factor: {error_metrics['scale_factor']:.2f}x")
        print(f"  Normalized MAE: {error_metrics['mae_normalized']:.4f}")

    return {"correlation": corr_metrics, "errors": error_metrics}


# Export all functions
__all__ = [
    # Shape verification
    "verify_input_shape_gpac",
    "verify_input_shape_tensorpac",
    "verify_output_shapes_match",
    # Band utilities
    "extract_gpac_bands",
    "verify_band_ranges",
    "check_band_spacing",
    # Data preparation
    "prepare_signal_gpac",
    "prepare_signal_tensorpac",
    # Metrics
    "compute_correlation_metrics",
    "compute_error_metrics",
    # Reporting
    "print_shape_report",
    "print_band_report",
    "print_comparison_summary",
    # Quick comparison
    "quick_compare",
]

# EOF
