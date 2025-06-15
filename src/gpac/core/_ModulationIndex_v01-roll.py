#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:25:08 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/core/_ModulationIndex.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/core/_ModulationIndex.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationIndex(nn.Module):
    """
    Optimized modulation index computation with reduced loops.
    The modulation index quantifies the non-uniformity of amplitude
    distribution across phase bins using Shannon entropy.
    Formula: MI = 1 + sum(p * log(p))/log(N) where sum(p * log(p)) = -H
    This exactly matches TensorPAC's implementation (Tort et al. 2010).
    Uses differentiable soft binning with softmax for gradient-based optimization.
    """

    def __init__(
        self, n_bins: int = 18, temperature: float = 0.01, fp16: bool = False
    ) -> None:
        """
        Initialize ModulationIndex calculator with soft binning for differentiability.
        Parameters
        ----------
        n_bins : int
            Number of phase bins (default: 18, i.e., 20 degrees per bin)
        temperature : float
            Temperature parameter for soft binning (default: 0.01)
            Lower values approach hard binning while maintaining differentiability
        """
        super().__init__()
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.n_bins = n_bins
        self.epsilon = 1e-5 if fp16 else 1e-10
        self.temperature = temperature
        self.fp16 = fp16
        self._chunk_cache = {}

        phase_bins = torch.linspace(-np.pi, np.pi, n_bins + 1)
        phase_bin_centers = (phase_bins[1:] + phase_bins[:-1]) / 2
        uniform_entropy = torch.tensor(np.log(n_bins))

        if fp16:
            phase_bins = phase_bins.half()
            phase_bin_centers = phase_bin_centers.half()
            uniform_entropy = uniform_entropy.half()

        self.register_buffer("phase_bins", phase_bins)
        self.register_buffer("phase_bin_centers", phase_bin_centers)
        self.register_buffer("uniform_entropy", uniform_entropy)

    def _get_chunk_size(
        self, batch_channels, freqs_phase, segments, time, freqs_amplitude
    ):
        cache_key = (batch_channels, freqs_phase, segments, time)

        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]

        available_memory_gb = 75
        bytes_per_element = 2 if self.fp16 else 4
        estimated_memory_per_freq = (
            batch_channels
            * freqs_phase
            * segments
            * time
            * self.n_bins
            * bytes_per_element
        ) / 1e9
        max_freqs_per_chunk = min(
            freqs_amplitude,
            max(1, int(available_memory_gb / estimated_memory_per_freq)),
        )
        chunk_size = max(min(max_freqs_per_chunk, freqs_amplitude), 16)

        self._chunk_cache[cache_key] = chunk_size
        return chunk_size

    def forward(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor,
        compute_distributions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate modulation index from phase and amplitude signals.

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

        Returns
        -------
        dict
            Dictionary containing:
            - 'mi': MI values per segment, shape (batch, channels, segments, freqs_phase, freqs_amplitude)
            - 'amplitude_distributions': Amplitude distributions (None if compute_distributions=False)
            - 'phase_bin_centers': Center values of phase bins in radians
            - 'phase_bin_edges': Edge values of phase bins in radians
        """
        batch, channels, freqs_phase, segments, time = phase.shape
        batch_amp, channels_amp, freqs_amplitude, segments_amp, time_amp = (
            amplitude.shape
        )

        self._validate_shapes(phase.shape, amplitude.shape)

        phase_flat = phase.reshape(-1)
        weights = self._phase_binning(phase_flat)
        weights_shaped = weights.reshape(
            batch, channels, freqs_phase, segments, time, self.n_bins
        )

        weights_vectorized = weights_shaped.view(
            batch * channels, freqs_phase, segments, time, self.n_bins
        )
        amp_vectorized = amplitude.view(
            batch * channels, freqs_amplitude, segments, time
        )

        mi_per_seg_vectorized, amp_dist_vectorized = self._compute_mi_vectorized(
            weights_vectorized,
            amp_vectorized,
            compute_distributions,
        )

        mi_per_segment_tensor = mi_per_seg_vectorized.view(
            batch, channels, segments, freqs_phase, freqs_amplitude
        )

        amp_dists_tensor = None
        if compute_distributions and amp_dist_vectorized is not None:
            amp_dists_tensor = amp_dist_vectorized.view(
                batch,
                channels,
                segments,
                freqs_phase,
                freqs_amplitude,
                self.n_bins,
            )

        return {
            "mi": mi_per_segment_tensor,
            "amplitude_distributions": amp_dists_tensor,
            "phase_bin_centers": self.phase_bin_centers.to(
                mi_per_segment_tensor.device
            ),
            "phase_bin_edges": self.phase_bins.to(mi_per_segment_tensor.device),
        }

    def _validate_shapes(self, phase_shape, amplitude_shape):
        """Validate shape compatibility between phase and amplitude."""
        batch, channels, freqs_phase, segments, time = phase_shape
        batch_amp, channels_amp, freqs_amplitude, segments_amp, time_amp = (
            amplitude_shape
        )

        if batch != batch_amp:
            raise ValueError(
                f"Batch size mismatch: phase={batch}, amplitude={batch_amp}"
            )
        if channels != channels_amp:
            raise ValueError(
                f"Channel size mismatch: phase={channels}, amplitude={channels_amp}"
            )
        if time != time_amp:
            raise ValueError(
                f"Time dimension mismatch: phase={time}, amplitude={time_amp}"
            )
        if segments != segments_amp:
            raise ValueError(
                f"Segment dimension mismatch: phase={segments}, amplitude={segments_amp}"
            )

    def _phase_binning(self, phase: torch.Tensor) -> torch.Tensor:
        """Assign phases to bins using differentiable soft binning."""
        n_phases = phase.shape[0]
        effective_temperature = self.temperature

        if n_phases > 10_000_000:
            effective_temperature = 0.01

        phase_expanded = phase.unsqueeze(-1)
        centers_expanded = self.phase_bin_centers.to(phase.device).unsqueeze(0)

        diff = phase_expanded - centers_expanded
        diff = torch.fmod(diff + np.pi, 2 * np.pi) - np.pi

        similarity = -torch.abs(diff) / effective_temperature
        weights = F.softmax(similarity, dim=-1)

        return weights

    def _compute_mi_vectorized(
        self,
        weights: torch.Tensor,
        amplitude: torch.Tensor,
        compute_distributions: bool = False,
    ) -> tuple:
        """
        Memory-efficient MI computation following TensorPAC's approach.

        Parameters
        ----------
        weights : torch.Tensor
            Phase bin weights, shape (batch*channels, freqs_phase, segments, time, n_bins)
        amplitude : torch.Tensor
            Amplitude values, shape (batch*channels, freqs_amplitude, segments, time)

        Returns
        -------
        tuple
            (mi_per_segment, amplitude_distributions)
        """
        batch_channels, freqs_phase, segments, time, n_bins = weights.shape
        _, freqs_amplitude, _, _ = amplitude.shape

        dtype = torch.float16 if self.fp16 else torch.float32

        mi_per_segment = torch.zeros(
            (batch_channels, segments, freqs_phase, freqs_amplitude),
            device=weights.device,
            dtype=dtype,
        )

        if compute_distributions:
            amp_distributions = torch.zeros(
                (
                    batch_channels,
                    segments,
                    freqs_phase,
                    freqs_amplitude,
                    n_bins,
                ),
                device=weights.device,
                dtype=dtype,
            )
        else:
            amp_distributions = None

        chunk_size = self._get_chunk_size(
            batch_channels, freqs_phase, segments, time, freqs_amplitude
        )
        bytes_per_element = 2 if self.fp16 else 4

        for amp_start in range(0, freqs_amplitude, chunk_size):
            amp_end = min(amp_start + chunk_size, freqs_amplitude)
            amp_chunk = amplitude[:, amp_start:amp_end]

            amp_dist_per_seg = torch.zeros(
                (
                    batch_channels,
                    freqs_phase,
                    amp_chunk.shape[1],
                    segments,
                    n_bins,
                ),
                device=weights.device,
                dtype=dtype,
            )

            chunk_memory_gb = (
                batch_channels
                * freqs_phase
                * amp_chunk.shape[1]
                * segments
                * n_bins
                * bytes_per_element
            ) / 1e9

            if chunk_memory_gb < 30:
                weights_exp = weights.unsqueeze(2)
                amp_exp = amp_chunk.unsqueeze(1).unsqueeze(-1)
                weighted_distributions = weights_exp * amp_exp
                amp_dist_per_seg = weighted_distributions.sum(dim=4)
            else:
                for p_idx in range(freqs_phase):
                    pha_weights = weights[:, p_idx]
                    amp_dist_per_seg[:, p_idx] = torch.einsum(
                        "bast,bstn->basn", amp_chunk, pha_weights
                    )

            amp_sum_per_seg = amp_dist_per_seg.sum(dim=-1, keepdim=True)
            amp_dist_per_seg_norm = amp_dist_per_seg / (amp_sum_per_seg + self.epsilon)
            amp_dist_per_seg_norm = torch.clamp(amp_dist_per_seg_norm, min=self.epsilon)

            log_p_per_seg = torch.log(amp_dist_per_seg_norm)
            neg_entropy_per_seg = (amp_dist_per_seg_norm * log_p_per_seg).sum(dim=-1)
            mi_per_seg_vals = 1 + neg_entropy_per_seg / self.uniform_entropy

            # Correct assignment: [batch_channels, segments, freqs_phase, freqs_amplitude]
            mi_per_segment[:, :, :, amp_start:amp_end] = mi_per_seg_vals.permute(
                0, 3, 1, 2
            )

            if compute_distributions:
                # Correct assignment: [batch_channels, segments, freqs_phase, freqs_amplitude, n_bins]
                amp_distributions[:, :, :, amp_start:amp_end, :] = (
                    amp_dist_per_seg_norm.permute(0, 3, 1, 2, 4)
                )

        return (mi_per_segment, amp_distributions)

    def compute_surrogates(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor,
        n_perm: int,
        chunk_size: int = 20,
        pac_values: torch.Tensor = None,
        return_surrogates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute surrogate statistics for significance testing.
        Parameters
        ----------
        phase : torch.Tensor
            Phase values, shape (batch, channels, freqs_phase, segments, time)
        amplitude : torch.Tensor
            Amplitude values, shape (batch, channels, freqs_amplitude, segments, time)
        n_perm : int
            Number of permutations
        chunk_size : int
            Process permutations in chunks for memory efficiency
        pac_values : torch.Tensor, optional
            Original PAC values for z-score computation
        return_surrogates : bool
            If True, return all surrogate values (memory intensive)
        Returns
        -------
        dict
            Dictionary containing:
            - 'surrogate_mean': Mean of surrogates
            - 'surrogate_std': Standard deviation of surrogates
            - 'pac_z': Z-scores (if pac_values provided)
            - 'surrogates': All surrogate MI values (if return_surrogates=True)
        """
        batch, channels, freqs_phase, segments, time = phase.shape
        _, _, freqs_amplitude, _, _ = amplitude.shape

        surrogate_sum = torch.zeros(
            batch,
            channels,
            segments,
            freqs_phase,
            freqs_amplitude,
            device=phase.device,
            dtype=phase.dtype,
        )
        surrogate_sum_sq = torch.zeros_like(surrogate_sum)

        if return_surrogates:
            surrogates = torch.zeros(
                batch,
                channels,
                segments,
                freqs_phase,
                freqs_amplitude,
                n_perm,
                device=phase.device,
                dtype=phase.dtype,
            )
        else:
            surrogates = None

        # Fix surrogate assignment to match new shape
        for start_idx in range(0, n_perm, chunk_size):
            end_idx = min(start_idx + chunk_size, n_perm)
            current_chunk = end_idx - start_idx

            # Use full range of shifts (1 to time-1) for unbiased surrogate generation
            # Avoid shift=0 (no change) and shift=time (same as no change due to circular shift)
            shifts = torch.randint(1, time, (current_chunk,), device=phase.device)

            for perm_idx, shift in enumerate(shifts):
                phase_shifted = torch.roll(phase, shifts=shift.item(), dims=-1)
                mi_result = self.forward(
                    phase_shifted,
                    amplitude,
                    compute_distributions=False,
                )
                surrogate_mi = mi_result["mi"]

                surrogate_sum += surrogate_mi
                surrogate_sum_sq += surrogate_mi**2

                if return_surrogates:
                    # Fix: add dimension 5 for n_perm
                    if surrogates is None:
                        surrogates = torch.zeros(
                            *surrogate_mi.shape,
                            n_perm,
                            device=phase.device,
                            dtype=phase.dtype,
                        )
                    surrogates[..., start_idx + perm_idx] = surrogate_mi

        surrogate_mean = surrogate_sum / n_perm
        surrogate_var = (surrogate_sum_sq / n_perm) - (surrogate_mean**2)
        surrogate_std = torch.sqrt(surrogate_var + 1e-10)

        pac_z = None
        if pac_values is not None:
            pac_z = (pac_values - surrogate_mean) / (surrogate_std + 1e-10)

        return {
            "surrogate_mean": surrogate_mean,
            "surrogate_std": surrogate_std,
            "pac_z": pac_z,
            "surrogates": surrogates,
        }


# EOF
