#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 17:31:43 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_ModulationIndex.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_ModulationIndex.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationIndex(nn.Module):
    """Calculates Modulation Index (MI) or Amplitude Distribution per Phase Bin."""

    def __init__(
        self,
        n_bins: int = 18,  # Number of phase bins
        fp16: bool = False,  # Use float16 for intermediate/output calculations if True
        amp_prob: bool = False,  # If True, return amplitude probabilities instead of MI
    ):
        super().__init__()
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")
        self.n_bins = n_bins
        self.fp16 = fp16
        self.amp_prob = amp_prob  # Flag to return probabilities or MI

        # Define phase bin edges from -pi to pi
        bin_edges = torch.linspace(-np.pi, np.pi, n_bins + 1)
        # Register as non-persistent buffer
        self.register_buffer("pha_bin_cutoffs", bin_edges, persistent=False)

    @property
    def pha_bin_centers(self) -> torch.Tensor:
        """Returns the center points of the phase bins."""
        cutoffs = self.pha_bin_cutoffs
        return (cutoffs[1:] + cutoffs[:-1]) / 2.0

    @staticmethod
    def _phase_to_masks(
        pha: torch.Tensor, phase_bin_cutoffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts phase values into boolean masks indicating bin membership.
        """
        n_bins = len(phase_bin_cutoffs) - 1
        cutoffs = phase_bin_cutoffs.to(pha.device, pha.dtype)

        # Ensure input tensors are contiguous to avoid performance warning
        pha_cont = pha.contiguous() if not pha.is_contiguous() else pha
        cutoffs_cont = cutoffs.contiguous() if not cutoffs.is_contiguous() else cutoffs
        
        # Use torch.bucketize with contiguous tensors 
        bin_indices = torch.bucketize(pha_cont, cutoffs_cont, right=False)
        # Adjust indices and clamp
        bin_indices = (bin_indices - 1).clamp_(min=0, max=n_bins - 1)

        # Convert indices to one-hot boolean masks
        one_hot_masks = F.one_hot(bin_indices, num_classes=n_bins)

        return one_hot_masks.bool()

    def forward(
        self,
        pha: torch.Tensor,  # Phase tensor, shape e.g., (B, C, F_pha, Seg, Time)
        amp: torch.Tensor,  # Amplitude tensor, shape e.g., (B, C, F_amp, Seg, Time)
        epsilon: float = 1e-9,  # Small value for numerical stability
    ) -> torch.Tensor:
        """
        Computes Modulation Index or Amplitude Distribution.
        """
        # 1. Input Validation
        if pha.ndim != 5 or amp.ndim != 5:
            raise ValueError(
                f"Input tensors must be 5D. Got pha:{pha.ndim}, amp:{amp.ndim}"
            )
        if pha.shape[0:2] != amp.shape[0:2] or pha.shape[3:] != amp.shape[3:]:
            raise ValueError(
                f"Dimensions mismatch. pha:{pha.shape}, amp:{amp.shape}"
            )

        # 2. Prepare Tensors
        compute_dtype = torch.float16 if self.fp16 else torch.float32
        output_dtype = torch.float16 if self.fp16 else amp.dtype

        pha = pha.to(compute_dtype)
        amp = amp.to(compute_dtype)
        device = pha.device

        # 3. Get Phase Bin Masks
        # Shape: (B, C, F_pha, Seg, Time, n_bins)
        pha_masks = self._phase_to_masks(pha, self.pha_bin_cutoffs)

        # 4. Broadcasting Setup
        # Target shape: (B, C, F_pha, F_amp, Seg, Time, n_bins)
        # Expand amp: (B, C, 1, F_amp, Seg, Time, 1)
        amp_expanded = amp.unsqueeze(2).unsqueeze(-1)
        # Expand pha_masks: (B, C, F_pha, 1, Seg, Time, n_bins)
        pha_masks_expanded = pha_masks.unsqueeze(3)

        # 5. Calculate Mean Amplitude per Bin
        pha_masks_float = pha_masks_expanded.float()
        amp_float = amp_expanded.float()

        # Sum amplitudes over time dimension (dim 5)
        amp_sums_in_bins = (pha_masks_float * amp_float).sum(
            dim=5, keepdim=True
        )
        # Count samples per bin over time dimension (dim 5)
        counts_in_bins = pha_masks_float.sum(dim=5, keepdim=True)
        # Calculate mean amplitude per bin
        amp_means_per_bin = amp_sums_in_bins / (counts_in_bins + epsilon)

        # 6. Calculate Amplitude Probabilities or MI
        # Normalize to get probability distribution across bins (dim -1)
        amp_probs = amp_means_per_bin / (
            amp_means_per_bin.sum(dim=-1, keepdim=True) + epsilon
        )

        if self.amp_prob:
            # Return Amplitude Probabilities averaged over segments (dim 4)
            result = amp_probs.squeeze(5).mean(
                dim=4
            )  # Shape: (B, C, F_pha, F_amp, n_bins)
            return result.to(output_dtype)
        else:
            # Calculate Modulation Index (MI)
            n_bins_tensor = torch.tensor(
                self.n_bins, device=device, dtype=compute_dtype
            )
            log_n_bins = torch.log(n_bins_tensor)

            # Sum (P * log(P)) over bins dimension (dim -1)
            kl_div_part = (amp_probs * torch.log(amp_probs + epsilon)).sum(
                dim=-1
            )  # Shape: (B, C, F_pha, F_amp, Seg, 1)

            # Calculate MI = (log(N) + sum(P*log(P))) / log(N)
            MI = (
                log_n_bins + kl_div_part
            ) / log_n_bins  # Shape: (B, C, F_pha, F_amp, Seg, 1)

            # Average MI over segments (dim 4), remove sum dim (dim 5)
            MI = MI.squeeze(5).mean(dim=4)  # Shape: (B, C, F_pha, F_amp)

            # Handle potential NaNs
            if torch.isnan(MI).any():
                warnings.warn(
                    "NaN values detected in Modulation Index. Replacing with 0.0."
                )
                MI = torch.nan_to_num(MI, nan=0.0)

            return MI.to(output_dtype)

# EOF