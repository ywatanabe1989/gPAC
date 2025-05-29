#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 19:00:00 (ywatanabe)"
# Author: Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)
# File: /home/ywatanabe/proj/gPAC/src/gpac/_ModulationIndex.py

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationIndex(nn.Module):
    """
    Optimized modulation index computation with reduced loops.
    
    The modulation index quantifies the non-uniformity of amplitude 
    distribution across phase bins using Shannon entropy.
    
    Uses differentiable soft binning with softmax for gradient-based optimization.
    """

    def __init__(
        self, 
        n_bins: int = 18,
        temperature: float = 0.1
    ) -> None:
        """
        Initialize ModulationIndex calculator.
        
        Parameters
        ----------
        n_bins : int
            Number of phase bins (default: 18, i.e., 20 degrees per bin)
        temperature : float
            Temperature parameter for soft binning (default: 0.1)
            Lower values approach hard binning
        """
        super().__init__()
        
        # Input validation
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        self.n_bins = n_bins
        self.epsilon = 1e-10  # To avoid log(0)
        self.temperature = temperature
        
        # Pre-calculate phase bins and Shannon entropy normalization
        self.register_buffer(
            "phase_bins", 
            torch.linspace(-np.pi, np.pi, n_bins + 1)
        )
        self.register_buffer(
            "phase_bin_centers",
            (self.phase_bins[1:] + self.phase_bins[:-1]) / 2
        )
        self.register_buffer(
            "uniform_entropy", 
            torch.tensor(np.log(n_bins))
        )

    def forward(self, phase: torch.Tensor, amplitude: torch.Tensor, compute_distributions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Calculate modulation index from phase and amplitude signals.
        
        Optimized implementation with reduced loops using broadcasting.
        
        Parameters
        ----------
        phase : torch.Tensor
            Phase values in radians with shape:
            (batch, channels, freqs_phase, segments, time)
        amplitude : torch.Tensor  
            Amplitude values with shape:
            (batch, channels, freqs_amplitude, segments, time)
        compute_distributions : bool
            If True, compute and return amplitude distributions (slower).
            Default is False for efficiency.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'mi': Modulation index values, shape (batch, channels, freqs_phase, freqs_amplitude)
            - 'mi_per_segment': MI values per segment before averaging (None if compute_distributions=False)
            - 'amplitude_distributions': Amplitude probability distributions across phase bins (None if compute_distributions=False)
            - 'phase_bin_centers': Center values of phase bins in radians
            - 'phase_bin_edges': Edge values of phase bins in radians
        """
        # Get dimensions
        batch, channels, freqs_phase, segments, time = phase.shape
        batch_amp, channels_amp, freqs_amplitude, segments_amp, time_amp = amplitude.shape
        
        # Validate shape compatibility
        if batch != batch_amp:
            raise ValueError(f"Batch size mismatch: phase={batch}, amplitude={batch_amp}")
        if channels != channels_amp:
            raise ValueError(f"Channel size mismatch: phase={channels}, amplitude={channels_amp}")
        if time != time_amp:
            raise ValueError(f"Time dimension mismatch: phase={time}, amplitude={time_amp}")
        if phase.shape[3] != amplitude.shape[3]:
            raise ValueError(f"Segment dimension mismatch: phase={phase.shape[3]}, amplitude={amplitude.shape[3]}")
        
        # Compute phase binning for all phase data at once
        phase_flat = phase.reshape(-1)
        weights = self._phase_binning(phase_flat)
        weights_shaped = weights.reshape(batch, channels, freqs_phase, segments, time, self.n_bins)
        
        # Initialize output lists
        mi_list = []
        mi_per_segment_list = [] if compute_distributions else None
        amp_dist_list = [] if compute_distributions else None
        
        # Loop over batch and channel only (major optimization)
        for b in range(batch):
            for c in range(channels):
                # Get weights and amplitude for this batch/channel
                weights_bc = weights_shaped[b, c]  # (freqs_phase, segments, time, n_bins)
                amp_bc = amplitude[b, c]  # (freqs_amplitude, segments, time)
                
                # Compute MI for all frequency pairs using broadcasting
                mi_bc, mi_per_seg_bc, amp_dist_bc = self._compute_mi_broadcast(
                    weights_bc, amp_bc, compute_distributions
                )
                
                mi_list.append(mi_bc)
                if compute_distributions:
                    mi_per_segment_list.append(mi_per_seg_bc)
                    amp_dist_list.append(amp_dist_bc)
        
        # Stack results
        mi_tensor = torch.stack(mi_list).reshape(batch, channels, freqs_phase, freqs_amplitude)
        
        if compute_distributions:
            mi_per_segment_tensor = torch.stack(mi_per_segment_list).reshape(
                batch, channels, freqs_phase, freqs_amplitude, segments
            )
            amp_dists_tensor = torch.stack(amp_dist_list).reshape(
                batch, channels, freqs_phase, freqs_amplitude, self.n_bins
            )
            
            return {
                "mi": mi_tensor,
                "mi_per_segment": mi_per_segment_tensor,
                "amplitude_distributions": amp_dists_tensor,
                "phase_bin_centers": self.phase_bin_centers,
                "phase_bin_edges": self.phase_bins,
            }
        else:
            return {
                "mi": mi_tensor,
                "mi_per_segment": None,
                "amplitude_distributions": None,
                "phase_bin_centers": self.phase_bin_centers,
                "phase_bin_edges": self.phase_bins,
            }

    def _phase_binning(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Assign phases to bins using differentiable soft binning with softmax.
        Memory-optimized version that remains differentiable.
        
        Parameters
        ----------
        phase : torch.Tensor
            Phase values in radians, shape (n_phases,)
            
        Returns
        -------
        torch.Tensor
            Soft bin assignments, shape (n_phases, n_bins)
        """
        # Use lower temperature for large inputs to approximate hard binning
        # while maintaining differentiability
        n_phases = phase.shape[0]
        effective_temperature = self.temperature
        if n_phases > 10_000_000:  # 10M threshold
            # Use very low temperature to approximate hard binning
            # This remains differentiable but approaches one-hot
            effective_temperature = 0.01
        
        # Expand dimensions for broadcasting
        phase_expanded = phase.unsqueeze(-1)  # (n_phases, 1)
        centers_expanded = self.phase_bin_centers.unsqueeze(0)  # (1, n_bins)
        
        # Compute circular distance efficiently
        diff = phase_expanded - centers_expanded
        
        # Wrap to [-pi, pi] using differentiable operations
        # Using fmod is more memory efficient than round
        diff = torch.fmod(diff + np.pi, 2 * np.pi) - np.pi
        
        # Convert distance to similarity (closer = higher)
        # Use in-place operations where possible
        similarity = -torch.abs(diff) / effective_temperature
        
        # Apply softmax to get weights
        # For memory efficiency with large tensors, we could use log_softmax + exp
        # but softmax should be fine with proper temperature
        weights = F.softmax(similarity, dim=-1)
        
        return weights
    
    def _compute_mi_broadcast(self, weights, amplitude, compute_distributions):
        """
        Compute MI for all frequency pairs using broadcasting.
        
        Parameters
        ----------
        weights : torch.Tensor
            Shape: (freqs_phase, segments, time, n_bins)
        amplitude : torch.Tensor
            Shape: (freqs_amplitude, segments, time)
        compute_distributions : bool
            Whether to compute distributions
            
        Returns
        -------
        tuple
            (mi, mi_per_segment, amp_distributions) or (mi, None, None)
        """
        freqs_phase, segments, time, n_bins = weights.shape
        freqs_amplitude = amplitude.shape[0]
        
        # Expand dimensions for broadcasting
        weights_exp = weights.unsqueeze(1)  # (freqs_phase, 1, segments, time, n_bins)
        amp_exp = amplitude.unsqueeze(0)    # (1, freqs_amplitude, segments, time)
        
        # Compute MI per segment
        mi_per_segment = []
        
        for seg in range(segments):
            # Get segment data
            seg_weights = weights_exp[:, :, seg, :, :]  # (freqs_p, 1, time, n_bins)
            seg_amp = amp_exp[:, :, seg, :]  # (1, freqs_a, time)
            
            # Transpose weights for multiplication
            seg_weights_t = seg_weights.transpose(-2, -1)  # (freqs_p, 1, n_bins, time)
            seg_amp_exp = seg_amp.unsqueeze(2)  # (1, freqs_a, 1, time)
            
            # Weighted amplitude
            weighted_amp = seg_weights_t * seg_amp_exp  # (freqs_p, freqs_a, n_bins, time)
            
            # Sum over time
            amp_sum = weighted_amp.sum(dim=-1)  # (freqs_p, freqs_a, n_bins)
            weight_sum = seg_weights.sum(dim=-2) + self.epsilon  # (freqs_p, 1, n_bins)
            
            # Normalize
            amp_dist = amp_sum / weight_sum
            amp_dist = amp_dist / (amp_dist.sum(dim=-1, keepdim=True) + self.epsilon)
            amp_dist = torch.clamp(amp_dist, min=self.epsilon)
            
            # Compute entropy
            entropy = -torch.sum(amp_dist * torch.log(amp_dist), dim=-1)  # (freqs_p, freqs_a)
            
            # Compute MI
            seg_mi = (self.uniform_entropy - entropy) / self.uniform_entropy
            mi_per_segment.append(seg_mi)
        
        # Stack and average
        mi_per_segment_stack = torch.stack(mi_per_segment, dim=-1)  # (freqs_p, freqs_a, segments)
        mi = mi_per_segment_stack.mean(dim=-1)  # (freqs_p, freqs_a)
        
        if compute_distributions:
            # Compute overall distributions
            weights_flat = weights.reshape(freqs_phase, -1, n_bins)  # (freqs_p, S*T, n_bins)
            weights_flat_t = weights_flat.transpose(1, 2).unsqueeze(1)  # (freqs_p, 1, n_bins, S*T)
            
            amp_flat = amplitude.reshape(freqs_amplitude, -1)  # (freqs_a, S*T)
            amp_flat_exp = amp_flat.unsqueeze(0).unsqueeze(2)  # (1, freqs_a, 1, S*T)
            
            # Weighted amplitude for all time
            weighted_amp_all = weights_flat_t * amp_flat_exp  # (freqs_p, freqs_a, n_bins, S*T)
            amp_sum_all = weighted_amp_all.sum(dim=-1)  # (freqs_p, freqs_a, n_bins)
            
            weight_sum_all = weights_flat.sum(dim=1, keepdim=True) + self.epsilon  # (freqs_p, 1, n_bins)
            amp_dist_all = amp_sum_all / weight_sum_all
            amp_dist_all = amp_dist_all / (amp_dist_all.sum(dim=-1, keepdim=True) + self.epsilon)
            amp_dist_all = torch.clamp(amp_dist_all, min=self.epsilon)
            
            return mi, mi_per_segment_stack, amp_dist_all
        else:
            return mi, None, None


# EOF