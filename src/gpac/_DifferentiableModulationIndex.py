#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 00:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_DifferentiableModulationIndex.py

"""
Differentiable Modulation Index calculation using soft binning.

This module provides a fully differentiable alternative to the standard
histogram-based MI calculation for use in trainable PAC pipelines.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableModulationIndex(nn.Module):
    """
    Differentiable Modulation Index calculation using soft binning.
    
    This class provides a fully differentiable alternative to histogram-based
    MI calculation by using soft bin assignments instead of hard binning.
    
    Parameters
    ----------
    n_bins : int, optional
        Number of phase bins (default: 18)
    temperature : float, optional
        Temperature parameter for soft binning. Lower values make binning
        more discrete, higher values make it softer (default: 1.0)
    binning_method : str, optional
        Method for soft binning: 'softmax' or 'gaussian' (default: 'softmax')
    fp16 : bool, optional
        Use half precision for computation (default: False)
    """
    
    def __init__(
        self,
        n_bins: int = 18,
        temperature: float = 1.0,
        binning_method: str = 'softmax',
        fp16: bool = False,
    ):
        super().__init__()
        
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if binning_method not in ['softmax', 'gaussian']:
            raise ValueError("binning_method must be 'softmax' or 'gaussian'.")
            
        self.n_bins = n_bins
        self.temperature = temperature
        self.binning_method = binning_method
        self.fp16 = fp16
        
        # Create phase bin centers
        bin_centers = torch.linspace(-np.pi, np.pi, n_bins + 1)[:-1] + np.pi / n_bins
        self.register_buffer("pha_bin_centers", bin_centers, persistent=False)
        
    @property
    def pha_bin_cutoffs(self) -> torch.Tensor:
        """Returns bin edges for compatibility with standard MI class."""
        edges = torch.linspace(-np.pi, np.pi, self.n_bins + 1)
        return edges.to(self.pha_bin_centers.device)
    
    def _soft_phase_binning(
        self, 
        pha: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert phase values to soft bin assignments.
        
        Parameters
        ----------
        pha : torch.Tensor
            Phase tensor of shape (..., time_points)
            
        Returns
        -------
        torch.Tensor
            Soft bin weights of shape (..., time_points, n_bins)
        """
        # Compute distances/similarities to bin centers
        # pha: (..., time) -> (..., time, 1)
        # centers: (n_bins,) -> (1, ..., 1, n_bins)
        pha_expanded = pha.unsqueeze(-1)
        centers_expanded = self.pha_bin_centers.view(
            *([1] * (pha.ndim)), self.n_bins
        )
        
        if self.binning_method == 'softmax':
            # Compute circular distances (important for phase data)
            dist1 = torch.abs(pha_expanded - centers_expanded)
            dist2 = 2 * np.pi - dist1
            distances = torch.minimum(dist1, dist2)
            
            # Soft assignment with temperature scaling
            logits = -distances / self.temperature
            soft_weights = torch.softmax(logits, dim=-1)
            
        elif self.binning_method == 'gaussian':
            # Gaussian soft binning
            dist1 = (pha_expanded - centers_expanded) ** 2
            dist2 = (pha_expanded - centers_expanded + 2*np.pi) ** 2  
            dist3 = (pha_expanded - centers_expanded - 2*np.pi) ** 2
            
            # Use minimum circular distance
            distances = torch.minimum(torch.minimum(dist1, dist2), dist3)
            
            # Gaussian weights
            soft_weights = torch.exp(-distances / (2 * self.temperature**2))
            soft_weights = soft_weights / (soft_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return soft_weights
    
    def forward(
        self,
        pha: torch.Tensor,
        amp: torch.Tensor,
        epsilon: float = 1e-9,
    ) -> dict:
        """
        Compute differentiable Modulation Index and amplitude distribution.
        
        Parameters
        ----------
        pha : torch.Tensor
            Phase tensor of shape (B, C, F_pha, Seg, Time)
        amp : torch.Tensor  
            Amplitude tensor of shape (B, C, F_amp, Seg, Time)
        epsilon : float, optional
            Small value for numerical stability (default: 1e-9)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'mi': Differentiable Modulation Index values
            - 'amp_prob': Amplitude probability distribution per phase bin
            - 'pha_bin_centers': Phase bin center values
        """
        # Input validation
        if pha.ndim != 5 or amp.ndim != 5:
            raise ValueError(
                f"Input tensors must be 5D. Got pha:{pha.ndim}, amp:{amp.ndim}"
            )
        if pha.shape[0:2] != amp.shape[0:2] or pha.shape[3:] != amp.shape[3:]:
            raise ValueError(
                f"Dimensions mismatch. pha:{pha.shape}, amp:{amp.shape}"
            )
        
        # Prepare tensors
        compute_dtype = torch.float16 if self.fp16 else torch.float32
        output_dtype = torch.float16 if self.fp16 else amp.dtype
        
        pha = pha.to(compute_dtype)
        amp = amp.to(compute_dtype)
        device = pha.device
        
        # Get soft phase bin assignments
        # Shape: (B, C, F_pha, Seg, Time, n_bins)
        soft_masks = self._soft_phase_binning(pha)
        
        # Broadcasting setup for cross-frequency computation
        # Target shape: (B, C, F_pha, F_amp, Seg, Time, n_bins)
        
        # Expand amp: (B, C, 1, F_amp, Seg, Time, 1)
        amp_expanded = amp.unsqueeze(2).unsqueeze(-1)
        
        # Expand soft_masks: (B, C, F_pha, 1, Seg, Time, n_bins)  
        soft_masks_expanded = soft_masks.unsqueeze(3)
        
        # Compute weighted amplitude means per bin
        # Weight amplitudes by soft bin assignments
        weighted_amps = soft_masks_expanded * amp_expanded
        
        # Sum over time dimension (dim 5)
        amp_sums_per_bin = weighted_amps.sum(dim=5, keepdim=True)
        weight_sums_per_bin = soft_masks_expanded.sum(dim=5, keepdim=True)
        
        # Mean amplitude per bin (weighted by soft assignments)
        amp_means_per_bin = amp_sums_per_bin / (weight_sums_per_bin + epsilon)
        
        # Normalize to get probability distribution across bins (dim -1)
        amp_probs = amp_means_per_bin / (
            amp_means_per_bin.sum(dim=-1, keepdim=True) + epsilon
        )
        
        # Compute differentiable Modulation Index
        n_bins_tensor = torch.tensor(
            self.n_bins, device=device, dtype=compute_dtype
        )
        entropy_uniform = torch.log(n_bins_tensor)
        
        # Compute entropy of observed distribution
        entropy_observed = -(amp_probs * torch.log(amp_probs + epsilon)).sum(dim=-1)
        
        # MI = (H_uniform - H_observed) / H_uniform
        # This is equivalent to KL divergence normalized by uniform entropy
        mi_result = (entropy_uniform - entropy_observed) / entropy_uniform
        
        # Remove singleton dimensions and average over segments
        mi_result = mi_result.squeeze(5).mean(dim=4)  # Average over segments
        amp_prob_result = amp_probs.squeeze(5).mean(dim=4)
        
        # Handle potential numerical issues
        if torch.isnan(mi_result).any():
            warnings.warn(
                "NaN values detected in differentiable MI. Replacing with 0.0."
            )
            mi_result = torch.nan_to_num(mi_result, nan=0.0)
        
        # Clamp MI to valid range [0, 1]
        mi_result = torch.clamp(mi_result, min=0.0, max=1.0)
        
        return {
            "mi": mi_result.to(output_dtype),
            "amp_prob": amp_prob_result.to(output_dtype), 
            "pha_bin_centers": self.pha_bin_centers,
        }


# EOF