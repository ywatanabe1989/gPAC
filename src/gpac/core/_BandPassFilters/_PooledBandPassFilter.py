#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:50:23 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/core/_BandPassFilters/_PooledBandPassFilter.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/core/_BandPassFilters/_PooledBandPassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledBandPassFilter(nn.Module):
    """
    Learnable bandpass filter with Gumbel Softmax selection.

    Creates comprehensive filter pool across frequency range, then learns
    to select optimal subset using differentiable top-k selection.

    Supports two modes:
    1. Automatic generation with systematic filter bank covering frequency range
    2. Manual band specification for precise control

    Field-standard bandwidths (when auto-generating):
    - Phase bands: [f - f/4, f + f/4] (bandwidth = f/2)
    - Amplitude bands: [f - f/8, f + f/8] (bandwidth = f/4)

    Selection mechanism:
    - Pool ratio controls diversity (e.g., 2.0x creates 20 filters to select 10)
    - Gumbel Softmax enables differentiable discrete selection
    - Hard selection available for inference

    WARNING: Manual bands take precedence over range parameters.
    If both are provided, manual bands will be used and range parameters ignored.
    """

    def __init__(
        self,
        fs: float,
        pha_range_hz: Optional[Tuple[float, float]] = (2, 20),
        amp_range_hz: Optional[Tuple[float, float]] = (60, 160),
        pha_n_bands: Optional[int] = 10,
        amp_n_bands: Optional[int] = 10,
        pha_bands_hz: Optional[List[List[float]]] = None,
        amp_bands_hz: Optional[List[List[float]]] = None,
        pha_n_pool_ratio: float = 2.0,
        amp_n_pool_ratio: float = 2.0,
        n_cycles: int = 4,
        temperature: float = 1.0,
        hard_selection: bool = False,
        spacing: str = "linear",
        fp16: bool = False,
    ):
        """
        Initialize pooled bandpass filter with learnable selection.
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        pha_range_hz : tuple, optional
            Phase frequency range (min, max) in Hz. Ignored if pha_bands_hz provided.
        amp_range_hz : tuple, optional
            Amplitude frequency range (min, max) in Hz. Ignored if amp_bands_hz provided.
        pha_n_bands : int, optional
            Number of phase filters to select from pool
        amp_n_bands : int, optional
            Number of amplitude filters to select from pool
        pha_bands_hz : list of lists, optional
            Manual phase bands [[low1, high1], [low2, high2], ...].
            Takes precedence over range parameters.
        amp_bands_hz : list of lists, optional
            Manual amplitude bands [[low1, high1], [low2, high2], ...].
            Takes precedence over range parameters.
        pha_n_pool_ratio : float
            Ratio of pool size to number of bands for phase filters
        amp_n_pool_ratio : float
            Ratio of pool size to number of bands for amplitude filters
        n_cycles : int
            Filter order (number of cycles for bandwidth calculation)
        temperature : float
            Gumbel softmax temperature for selection
        hard_selection : bool
            Use hard selection in forward pass
        spacing : str
            'log' for logarithmic spacing, 'linear' for linear spacing
        fp16 : bool
            Use half precision for memory efficiency
        Raises
        ------
        ValueError
            If band frequencies exceed Nyquist frequency or are invalid
        """
        super().__init__()
        self.fs = float(fs)
        self.n_cycles = int(n_cycles)
        self.pha_n_bands = pha_n_bands
        self.amp_n_bands = amp_n_bands
        self.temperature = temperature
        self.hard_selection = hard_selection
        self.spacing = spacing
        self.fp16 = fp16
        self.nyq = self.fs / 2

        # Set default values if None
        self.pha_n_pool_ratio = (
            pha_n_pool_ratio if pha_n_pool_ratio is not None else 2.0
        )
        self.amp_n_pool_ratio = (
            amp_n_pool_ratio if amp_n_pool_ratio is not None else 2.0
        )

        self._check_parameter_conflicts(
            pha_range_hz,
            pha_n_bands,
            pha_bands_hz,
            amp_range_hz,
            amp_n_bands,
            amp_bands_hz,
        )

        # Device management
        if self.fp16:
            self.register_parameter(
                "_dummy", nn.Parameter(torch.zeros(1, dtype=torch.half))
            )
        else:
            self.register_parameter("_dummy", nn.Parameter(torch.zeros(1)))

        # Handle direct band specification vs range generation
        if pha_bands_hz is not None:
            self.pha_bands_hz = torch.tensor(pha_bands_hz, dtype=torch.float32)
            self.pha_n_bands = len(pha_bands_hz)  # Always override
            self.pha_n_pool = int(len(pha_bands_hz) * self.pha_n_pool_ratio)
            self.pha_bands_hz, self.pha_center_freqs, self.pha_bandwidths = (
                self._create_hybrid_pool(self.pha_bands_hz, self.pha_n_pool, 0.5)
            )
        else:
            self.pha_n_pool = int(pha_n_bands * self.pha_n_pool_ratio)
            self.pha_center_freqs = self._create_frequency_grid(
                pha_range_hz[0], pha_range_hz[1], self.pha_n_pool
            )
            self.pha_bandwidths = self.pha_center_freqs / 2.0
            self.pha_bands_hz = self._create_bands_from_centers(
                self.pha_center_freqs, self.pha_bandwidths
            )

        if amp_bands_hz is not None:
            self.amp_bands_hz = torch.tensor(amp_bands_hz, dtype=torch.float32)
            self.amp_n_bands = len(amp_bands_hz)  # Always override
            self.amp_n_pool = int(len(amp_bands_hz) * self.amp_n_pool_ratio)
            self.amp_bands_hz, self.amp_center_freqs, self.amp_bandwidths = (
                self._create_hybrid_pool(self.amp_bands_hz, self.amp_n_pool, 0.25)
            )
        else:
            self.amp_n_pool = int(amp_n_bands * self.amp_n_pool_ratio)
            self.amp_center_freqs = self._create_frequency_grid(
                amp_range_hz[0], amp_range_hz[1], self.amp_n_pool
            )
            self.amp_bandwidths = self.amp_center_freqs / 4.0
            self.amp_bands_hz = self._create_bands_from_centers(
                self.amp_center_freqs, self.amp_bandwidths
            )

        # Validate all bands against Nyquist
        self._validate_nyquist(self.pha_bands_hz)
        self._validate_nyquist(self.amp_bands_hz)

        # Initialize filter banks
        self._init_filter_banks()

        # Initialize selection parameters
        self.pha_logits = nn.Parameter(torch.randn(self.pha_n_pool))
        self.amp_logits = nn.Parameter(torch.randn(self.amp_n_pool))

        # For caching in eval mode
        self._cached_pha_selection = None
        self._cached_amp_selection = None

    def _check_parameter_conflicts(
        self,
        pha_range_hz,
        pha_n_bands,
        pha_bands_hz,
        amp_range_hz,
        amp_n_bands,
        amp_bands_hz,
    ):
        """Check for parameter conflicts and issue warnings."""
        if pha_bands_hz is not None and (
            pha_range_hz is not None or pha_n_bands is not None
        ):
            warnings.warn(
                "Both pha_bands_hz and range parameters provided. "
                "Manual pha_bands_hz takes precedence - range parameters ignored.",
                UserWarning,
            )
        if amp_bands_hz is not None and (
            amp_range_hz is not None or amp_n_bands is not None
        ):
            warnings.warn(
                "Both amp_bands_hz and range parameters provided. "
                "Manual amp_bands_hz takes precedence - range parameters ignored.",
                UserWarning,
            )

    def _create_frequency_grid(
        self, f_min: float, f_max: float, n_filters: int
    ) -> torch.Tensor:
        """Create frequency grid with specified spacing."""
        if self.spacing == "log":
            log_freqs = torch.logspace(
                np.log10(f_min),
                np.log10(f_max),
                n_filters,
                dtype=torch.float32,
            )
        else:
            log_freqs = torch.linspace(f_min, f_max, n_filters, dtype=torch.float32)
        return log_freqs

    def _create_bands_from_centers(self, center_freqs, bandwidths):
        """Create band edges from center frequencies and bandwidths."""
        low_freqs = center_freqs - bandwidths / 2
        high_freqs = center_freqs + bandwidths / 2
        # Ensure valid frequencies
        low_freqs = torch.maximum(low_freqs, torch.tensor(0.5))
        high_freqs = torch.minimum(high_freqs, torch.tensor(self.nyq - 0.5))
        return torch.stack([low_freqs, high_freqs], dim=1)

    def _validate_nyquist(self, bands):
        """Validate all bands against Nyquist frequency."""
        if bands.max() >= self.nyq:
            invalid_bands = torch.where(bands >= self.nyq)
            raise ValueError(
                f"Band frequencies {bands[invalid_bands]} exceed Nyquist frequency {self.nyq} Hz"
            )

    def _init_filter_banks(self):
        """Initialize the systematic filter banks."""
        min_freq = min(
            self.pha_center_freqs.min().item(),
            self.amp_center_freqs.min().item(),
        )
        max_filter_len = int(self.n_cycles * self.fs / min_freq)
        max_filter_len = (
            max_filter_len if max_filter_len % 2 == 1 else max_filter_len + 1
        )

        # Phase filters
        pha_filters = []
        for cf, bw in zip(self.pha_center_freqs, self.pha_bandwidths):
            filt = self._design_bandpass_filter(cf.item(), bw.item(), max_filter_len)
            pha_filters.append(filt)
        self.register_buffer("pha_filter_bank", torch.stack(pha_filters))

        # Amplitude filters
        amp_filters = []
        for cf, bw in zip(self.amp_center_freqs, self.amp_bandwidths):
            filt = self._design_bandpass_filter(cf.item(), bw.item(), max_filter_len)
            amp_filters.append(filt)
        self.register_buffer("amp_filter_bank", torch.stack(amp_filters))
        self.max_filter_len = max_filter_len

    def _design_bandpass_filter(
        self, center_freq: float, bandwidth: float, target_len: int = None
    ) -> torch.Tensor:
        """Design bandpass filter using scipy Butterworth."""
        low_freq = center_freq - bandwidth / 2
        high_freq = center_freq + bandwidth / 2
        # Ensure frequencies are valid
        low_freq = max(0.5, low_freq)
        high_freq = min(self.fs / 2 - 0.5, high_freq)

        if target_len is None:
            filter_len = int(self.n_cycles * self.fs / center_freq)
            filter_len = filter_len if filter_len % 2 == 1 else filter_len + 1
        else:
            filter_len = target_len

        # Use scipy Butterworth design like StaticBandPassFilter
        from scipy.signal import butter, lfilter

        b, a = butter(
            self.n_cycles,
            [low_freq / (self.fs / 2), high_freq / (self.fs / 2)],
            btype="band",
        )

        impulse = np.zeros(filter_len)
        impulse[filter_len // 2] = 1.0
        fir_kernel = lfilter(b, a, impulse)

        # Normalize
        fir_kernel = fir_kernel / np.max(np.abs(fir_kernel))

        return torch.from_numpy(fir_kernel).float().type_as(self._dummy)

    def _gumbel_softmax_topk(
        self, logits: torch.Tensor, k: int, hard: bool = False
    ) -> torch.Tensor:
        """Gumbel softmax with top-k selection."""
        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
            )
            logits = logits + gumbel_noise

        topk_vals, topk_idx = torch.topk(logits, k)
        mask = torch.zeros_like(logits)
        mask[topk_idx] = 1.0

        exp_logits = torch.exp(logits / self.temperature) * mask
        weights = exp_logits / (exp_logits.sum() + 1e-10)

        if hard or self.hard_selection:
            weights_hard = torch.zeros_like(weights)
            weights_hard[topk_idx] = 1.0 / k
            weights = weights_hard - weights.detach() + weights

        return weights

    @torch._dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply zero-phase bandpass filtering to input signal.
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (..., time)
        Returns
        -------
        torch.Tensor
            Filtered signals of shape (..., n_selected_filters, time)
            where n_selected_filters = pha_n_bands + amp_n_bands
        """
        original_shape = x.shape
        seq_len = original_shape[-1]
        x_flat = x.view(-1, seq_len)

        if self.fp16:
            x_flat = x_flat.half()

        pha_selection = self._get_selection(self.pha_logits, self.pha_n_bands)
        amp_selection = self._get_selection(self.amp_logits, self.amp_n_bands)

        pha_filtered = self._apply_selected_filters_filtfilt(
            x_flat, self.pha_filter_bank, pha_selection, self.pha_n_bands
        )
        amp_filtered = self._apply_selected_filters_filtfilt(
            x_flat, self.amp_filter_bank, amp_selection, self.amp_n_bands
        )

        output = torch.cat([pha_filtered, amp_filtered], dim=1)
        n_selected = self.pha_n_bands + self.amp_n_bands
        new_shape = original_shape[:-1] + (n_selected, seq_len)
        return output.reshape(new_shape)

    def get_selected_frequencies(self) -> Tuple[List[float], List[float]]:
        """
        Get frequencies of currently selected filters.
        Returns
        -------
        tuple
            (phase_frequencies, amplitude_frequencies)
        """
        with torch.no_grad():
            pha_selection = self._gumbel_softmax_topk(
                self.pha_logits, self.pha_n_bands, hard=True
            )
            amp_selection = self._gumbel_softmax_topk(
                self.amp_logits, self.amp_n_bands, hard=True
            )

            pha_selected_idx = torch.where(pha_selection > 0)[0]
            amp_selected_idx = torch.where(amp_selection > 0)[0]

            pha_freqs = [self.pha_center_freqs[idx].item() for idx in pha_selected_idx]
            amp_freqs = [self.amp_center_freqs[idx].item() for idx in amp_selected_idx]

            return pha_freqs, amp_freqs

    def _get_selection(self, logits: torch.Tensor, n_bands: int) -> torch.Tensor:
        """Get filter selection weights with caching in eval mode."""
        cache_attr = (
            "_cached_pha_selection"
            if logits is self.pha_logits
            else "_cached_amp_selection"
        )

        if not self.training and getattr(self, cache_attr) is not None:
            return getattr(self, cache_attr)

        selection = self._gumbel_softmax_topk(logits, n_bands, not self.training)

        if not self.training:
            setattr(self, cache_attr, selection)

        return selection

    @torch._dynamo.disable
    def _apply_selected_filters_filtfilt(
        self,
        x: torch.Tensor,
        filter_bank: torch.Tensor,
        selection: torch.Tensor,
        n_bands: int,
    ) -> torch.Tensor:
        if self.training:
            selected_filters = (
                selection.unsqueeze(0).unsqueeze(-1) * filter_bank.unsqueeze(0)
            ).sum(dim=1, keepdim=True)
            selected_filters = selected_filters.expand(n_bands, -1, -1)
        else:
            _, topk_idx = torch.topk(selection, n_bands)
            selected_filters = filter_bank[topk_idx].unsqueeze(1)

        x = x.to(selected_filters.device)

        # Ensure dtype compatibility using _dummy
        if x.dtype != selected_filters.dtype:
            x = x.type_as(self._dummy)
        selected_filters = selected_filters.type_as(self._dummy)

        padding = filter_bank.shape[-1] // 2
        x_expanded = x.unsqueeze(1).expand(-1, n_bands, -1)

        filtered = F.conv1d(
            x_expanded,
            selected_filters,
            groups=n_bands,
            padding=padding,
        )

        filtered_backward = F.conv1d(
            filtered.flip(-1),
            selected_filters,
            groups=n_bands,
            padding=padding,
        ).flip(-1)

        return filtered_backward

    def clear_cache(self):
        """Clear cached selections when model parameters change."""
        self._cached_pha_selection = None
        self._cached_amp_selection = None

    def train(self, mode: bool = True):
        """Override train to clear cache when switching modes."""
        if mode != self.training:
            self.clear_cache()
        return super().train(mode)

    def _create_hybrid_pool(self, manual_bands_hz, n_pool, bandwidth_ratio):
        """Create hybrid pool combining manual bands with systematic filters."""
        # Get range from manual bands
        manual_centers = (manual_bands_hz[:, 0] + manual_bands_hz[:, 1]) / 2
        f_min = manual_centers.min().item()
        f_max = manual_centers.max().item()

        # Generate systematic pool across range
        systematic_centers = self._create_frequency_grid(f_min, f_max, n_pool)

        # Replace closest systematic filters with manual bands
        for manual_center in manual_centers:
            distances = torch.abs(systematic_centers - manual_center)
            closest_idx = torch.argmin(distances)
            systematic_centers[closest_idx] = manual_center

        # Create bands from final centers
        final_bandwidths = systematic_centers * bandwidth_ratio
        return (
            self._create_bands_from_centers(systematic_centers, final_bandwidths),
            systematic_centers,
            final_bandwidths,
        )

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current selection weights for post-analysis.

        Returns
        -------
        tuple
            (pha_weights, amp_weights) where weights sum to 1 for selected filters
        """
        with torch.no_grad():
            pha_weights = self._gumbel_softmax_topk(
                self.pha_logits, self.pha_n_bands, hard=True
            )
            amp_weights = self._gumbel_softmax_topk(
                self.amp_logits, self.amp_n_bands, hard=True
            )
            return pha_weights, amp_weights

    def get_selection_summary(self) -> dict:
        """
        Get comprehensive selection summary for analysis.

        Returns
        -------
        dict
            Selection summary with frequencies, weights, and indices
        """
        pha_weights, amp_weights = self.get_weights()
        pha_freqs, amp_freqs = self.get_selected_frequencies()

        pha_selected_idx = torch.where(pha_weights > 0)[0]
        amp_selected_idx = torch.where(amp_weights > 0)[0]

        return {
            "pha_selected_frequencies": pha_freqs,
            "amp_selected_frequencies": amp_freqs,
            "pha_weights": pha_weights[pha_selected_idx],
            "amp_weights": amp_weights[amp_selected_idx],
            "pha_selected_indices": pha_selected_idx.tolist(),
            "amp_selected_indices": amp_selected_idx.tolist(),
            "pha_logits": self.pha_logits.detach(),
            "amp_logits": self.amp_logits.detach(),
        }

    @property
    def info(self):
        """
        Return comprehensive filter configuration including current selections.
        Returns
        -------
        dict
            Filter configuration dictionary with selection weights and frequencies
        """
        with torch.no_grad():
            pha_weights = self._gumbel_softmax_topk(
                self.pha_logits, self.pha_n_bands, hard=True
            )
            amp_weights = self._gumbel_softmax_topk(
                self.amp_logits, self.amp_n_bands, hard=True
            )

            pha_selected_idx = torch.where(pha_weights > 0)[0]
            amp_selected_idx = torch.where(amp_weights > 0)[0]

            pha_freqs = [self.pha_center_freqs[idx].item() for idx in pha_selected_idx]
            amp_freqs = [self.amp_center_freqs[idx].item() for idx in amp_selected_idx]

        return {
            "pha_bands_hz": self.pha_bands_hz,
            "amp_bands_hz": self.amp_bands_hz,
            "pha_center_freqs": self.pha_center_freqs,
            "amp_center_freqs": self.amp_center_freqs,
            "pha_bandwidths": self.pha_bandwidths,
            "amp_bandwidths": self.amp_bandwidths,
            "fs": self.fs,
            "n_cycles": self.n_cycles,
            "pha_n_bands": self.pha_n_bands,
            "amp_n_bands": self.amp_n_bands,
            "pha_n_pool": self.pha_n_pool,
            "amp_n_pool": self.amp_n_pool,
            "spacing": self.spacing,
            "pha_selected_frequencies": pha_freqs,
            "amp_selected_frequencies": amp_freqs,
            "pha_weights": pha_weights,
            "amp_weights": amp_weights,
            "pha_logits": self.pha_logits.detach(),
            "amp_logits": self.amp_logits.detach(),
        }

    # @property
    # def info(self):
    #     """
    #     Return comprehensive filter configuration.
    #     Returns
    #     -------
    #     dict
    #         Filter configuration dictionary
    #     """
    #     return {
    #         "pha_bands_hz": self.pha_bands_hz,
    #         "amp_bands_hz": self.amp_bands_hz,
    #         "pha_center_freqs": self.pha_center_freqs,
    #         "amp_center_freqs": self.amp_center_freqs,
    #         "pha_bandwidths": self.pha_bandwidths,
    #         "amp_bandwidths": self.amp_bandwidths,
    #         "fs": self.fs,
    #         "n_cycles": self.n_cycles,
    #         "pha_n_bands": self.pha_n_bands,
    #         "amp_n_bands": self.amp_n_bands,
    #         "pha_n_pool": self.pha_n_pool,
    #         "amp_n_pool": self.amp_n_pool,
    #         "spacing": self.spacing,
    #     }


# EOF
