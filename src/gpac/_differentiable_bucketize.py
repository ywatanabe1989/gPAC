#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_differentiable_bucketize.py

"""
Differentiable versions of torch.bucketize for gradient-based optimization.

This module provides soft binning alternatives to the non-differentiable
torch.bucketize function, enabling gradient flow through discretization operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional


def differentiable_bucketize(
    input: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    temperature: float = 1.0,
    method: Literal["softmax", "sigmoid", "gaussian"] = "softmax",
    circular: bool = False,
    right: bool = False,
) -> torch.Tensor:
    """
    Differentiable version of torch.bucketize using soft binning.
    
    Instead of hard assignment to bins, this function returns soft weights
    indicating the degree of membership to each bin, enabling gradient flow.
    
    Parameters
    ----------
    input : torch.Tensor
        Input tensor of any shape
    boundaries : torch.Tensor
        1D tensor of sorted bin boundaries
    temperature : float, optional
        Temperature parameter controlling softness of binning.
        Lower values → harder binning, higher values → softer binning
    method : str, optional
        Soft binning method: "softmax", "sigmoid", or "gaussian"
    circular : bool, optional
        Whether to treat the data as circular (e.g., phase data)
    right : bool, optional
        If False, bins include left boundary. If True, include right boundary.
        
    Returns
    -------
    torch.Tensor
        Soft bin assignments of shape (*input.shape, n_bins) where n_bins = len(boundaries) - 1
        Each element in the last dimension represents the weight/probability
        of belonging to that bin.
        
    Examples
    --------
    >>> x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
    >>> boundaries = torch.tensor([0., 1., 2., 3., 4.])
    >>> soft_bins = differentiable_bucketize(x, boundaries, temperature=0.1)
    >>> soft_bins.shape
    torch.Size([4, 4])  # 4 inputs, 4 bins
    >>> soft_bins.sum(dim=-1)  # Weights sum to 1
    tensor([1., 1., 1., 1.])
    """
    n_bins = len(boundaries) - 1
    if n_bins <= 0:
        raise ValueError("boundaries must have at least 2 elements")
    
    # Compute bin centers
    bin_centers = (boundaries[:-1] + boundaries[1:]) / 2
    bin_widths = boundaries[1:] - boundaries[:-1]
    
    # Reshape for broadcasting: input (...) -> (..., 1)
    input_expanded = input.unsqueeze(-1)
    
    # Reshape bin centers: (n_bins,) -> (1, ..., 1, n_bins)
    centers_shape = [1] * input.ndim + [n_bins]
    centers_expanded = bin_centers.view(*centers_shape)
    widths_expanded = bin_widths.view(*centers_shape)
    
    if method == "softmax":
        if circular:
            # For circular data (e.g., phases from -π to π)
            # Compute circular distance
            diff = input_expanded - centers_expanded
            # Assume the circular range is from boundaries[0] to boundaries[-1]
            period = boundaries[-1] - boundaries[0]
            
            # Compute three possible distances (direct, +period, -period)
            dist1 = torch.abs(diff)
            dist2 = torch.abs(diff - period)
            dist3 = torch.abs(diff + period)
            
            # Use minimum circular distance
            distances = torch.minimum(torch.minimum(dist1, dist2), dist3)
        else:
            # Regular Euclidean distance
            distances = torch.abs(input_expanded - centers_expanded)
        
        # Convert distances to logits (negative distances)
        logits = -distances / temperature
        
        # Apply softmax to get soft bin assignments
        soft_bins = torch.softmax(logits, dim=-1)
        
    elif method == "sigmoid":
        # Sigmoid-based soft binning
        # Compute "probability" of being greater than each boundary
        # Then take differences to get bin probabilities
        
        # Expand boundaries for broadcasting
        boundaries_expanded = boundaries.view(*centers_shape[:-1], n_bins + 1)
        
        # Compute sigmoid activations for each boundary
        if right:
            # For right=True, use x > boundary
            activations = torch.sigmoid((input_expanded - boundaries_expanded) / temperature)
        else:
            # For right=False, use x >= boundary (approximated as x > boundary - epsilon)
            activations = torch.sigmoid((input_expanded - boundaries_expanded + 1e-7) / temperature)
        
        # Compute bin probabilities as differences
        # P(bin_i) = P(x > boundary_i) - P(x > boundary_{i+1})
        soft_bins = activations[..., :-1] - activations[..., 1:]
        
    elif method == "gaussian":
        # Gaussian kernel soft binning
        if circular:
            # Similar to softmax case but with Gaussian kernel
            diff = input_expanded - centers_expanded
            period = boundaries[-1] - boundaries[0]
            
            # Compute squared distances for three cases
            dist1_sq = diff ** 2
            dist2_sq = (diff - period) ** 2
            dist3_sq = (diff + period) ** 2
            
            # Use minimum squared distance
            min_dist_sq = torch.minimum(torch.minimum(dist1_sq, dist2_sq), dist3_sq)
            distances_sq = min_dist_sq
        else:
            distances_sq = (input_expanded - centers_expanded) ** 2
        
        # Normalize by bin width for scale invariance
        normalized_dist_sq = distances_sq / (widths_expanded ** 2)
        
        # Gaussian weights
        weights = torch.exp(-normalized_dist_sq / (2 * temperature ** 2))
        
        # Normalize to get probabilities
        soft_bins = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return soft_bins


def differentiable_bucketize_indices(
    input: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    temperature: float = 1.0,
    method: Literal["softmax", "sigmoid", "gaussian"] = "softmax",
    circular: bool = False,
    right: bool = False,
) -> torch.Tensor:
    """
    Differentiable version of torch.bucketize that returns weighted indices.
    
    This function returns a weighted average of bin indices instead of
    soft bin assignments, making it a drop-in replacement for torch.bucketize
    in differentiable contexts.
    
    Parameters
    ----------
    input : torch.Tensor
        Input tensor of any shape
    boundaries : torch.Tensor
        1D tensor of sorted bin boundaries
    temperature : float, optional
        Temperature parameter controlling softness of binning
    method : str, optional
        Soft binning method: "softmax", "sigmoid", or "gaussian"
    circular : bool, optional
        Whether to treat the data as circular
    right : bool, optional
        If False, bins include left boundary. If True, include right boundary.
        
    Returns
    -------
    torch.Tensor
        Weighted bin indices of the same shape as input.
        Values are continuous (not integers) to maintain differentiability.
        
    Examples
    --------
    >>> x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
    >>> boundaries = torch.tensor([0., 1., 2., 3., 4.])
    >>> soft_indices = differentiable_bucketize_indices(x, boundaries)
    >>> soft_indices  # Close to [0, 1, 2, 3] but continuous
    tensor([0.0000, 1.0000, 2.0000, 3.0000], grad_fn=<SumBackward1>)
    """
    # Get soft bin assignments
    soft_bins = differentiable_bucketize(
        input, boundaries, 
        temperature=temperature,
        method=method,
        circular=circular,
        right=right
    )
    
    # Create bin indices
    n_bins = soft_bins.shape[-1]
    indices = torch.arange(n_bins, dtype=input.dtype, device=input.device)
    
    # Reshape indices for broadcasting
    indices_shape = [1] * input.ndim + [n_bins]
    indices_expanded = indices.view(*indices_shape)
    
    # Compute weighted average of indices
    weighted_indices = (soft_bins * indices_expanded).sum(dim=-1)
    
    return weighted_indices


class DifferentiableBucketize(torch.nn.Module):
    """
    Module version of differentiable_bucketize for use in nn.Sequential.
    
    Parameters
    ----------
    boundaries : torch.Tensor or list
        Bin boundaries (will be converted to a buffer)
    temperature : float, optional
        Temperature parameter for soft binning
    method : str, optional
        Soft binning method
    circular : bool, optional
        Whether to treat data as circular
    right : bool, optional
        Bin boundary inclusion rule
    return_indices : bool, optional
        If True, return weighted indices. If False, return soft bin assignments.
    """
    
    def __init__(
        self,
        boundaries: torch.Tensor,
        temperature: float = 1.0,
        method: Literal["softmax", "sigmoid", "gaussian"] = "softmax",
        circular: bool = False,
        right: bool = False,
        return_indices: bool = False,
    ):
        super().__init__()
        
        if isinstance(boundaries, list):
            boundaries = torch.tensor(boundaries, dtype=torch.float32)
        
        self.register_buffer("boundaries", boundaries)
        self.temperature = temperature
        self.method = method
        self.circular = circular
        self.right = right
        self.return_indices = return_indices
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            return differentiable_bucketize_indices(
                input, self.boundaries,
                temperature=self.temperature,
                method=self.method,
                circular=self.circular,
                right=self.right,
            )
        else:
            return differentiable_bucketize(
                input, self.boundaries,
                temperature=self.temperature,
                method=self.method,
                circular=self.circular,
                right=self.right,
            )


# Convenience function for phase binning (circular data from -π to π)
def differentiable_phase_binning(
    phases: torch.Tensor,
    n_bins: int = 18,
    temperature: float = 1.0,
    method: Literal["softmax", "sigmoid", "gaussian"] = "softmax",
) -> torch.Tensor:
    """
    Specialized differentiable binning for phase data.
    
    Parameters
    ----------
    phases : torch.Tensor
        Phase values (assumed to be in [-π, π])
    n_bins : int
        Number of phase bins
    temperature : float
        Temperature for soft binning
    method : str
        Soft binning method
        
    Returns
    -------
    torch.Tensor
        Soft bin assignments with shape (*phases.shape, n_bins)
    """
    boundaries = torch.linspace(-np.pi, np.pi, n_bins + 1, 
                               device=phases.device, dtype=phases.dtype)
    
    return differentiable_bucketize(
        phases, boundaries,
        temperature=temperature,
        method=method,
        circular=True,
        right=False,
    )


# EOF