#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 00:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/src/gpac/_PAC.py

"""
PyTorch Module for calculating Phase-Amplitude Coupling (PAC).

This implementation uses TensorPAC-compatible filter design for
better comparability with existing neuroscience tools.
Includes differentiable MI calculation when trainable=True.
"""

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

from ._BandPassFilter import BandPassFilter
from ._OptimizedBandPassFilter import OptimizedBandPassFilter
from ._Hilbert import Hilbert
from ._ModulationIndex import ModulationIndex
from ._DifferentiableModulationIndex import DifferentiableModulationIndex
from ._utils import ensure_4d_input

# Try to import trainable filter
try:
    from ._Filters._TrainableBandPassFilter import TrainableBandPassFilter
    _TRAINABLE_AVAILABLE = True
except ImportError:
    _TRAINABLE_AVAILABLE = False


class PAC(nn.Module):
    """
    PyTorch Module for calculating Phase-Amplitude Coupling (PAC).
    
    When trainable=True:
    - Uses trainable bandpass filters
    - Uses differentiable soft-binning MI calculation
    - Enables end-to-end gradient optimization
    
    When trainable=False:
    - Uses static bandpass filters  
    - Uses standard histogram-based MI calculation
    - Maintains TensorPAC compatibility
    """

    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz: float = 2.0,
        pha_end_hz: float = 20.0,
        pha_n_bands: int = 50,
        amp_start_hz: float = 60.0,
        amp_end_hz: float = 160.0,
        amp_n_bands: int = 30,
        n_perm: Optional[int] = None,
        trainable: bool = False,
        fp16: bool = False,
        mi_n_bins: int = 18,
        filter_cycle_pha: int = 3,
        filter_cycle_amp: int = 6,
        filtfilt_mode: bool = False,
        edge_mode: Optional[str] = None,
        differentiable_mi_temperature: float = 1.0,
        differentiable_mi_method: str = 'softmax',
        use_optimized_filter: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.seq_len = seq_len
        self.fs = fs
        self.fp16 = fp16
        self.trainable = trainable
        self.filter_cycle_pha = filter_cycle_pha
        self.filter_cycle_amp = filter_cycle_amp
        self.mi_n_bins = mi_n_bins
        self.filtfilt_mode = filtfilt_mode
        self.edge_mode = edge_mode
        self.use_optimized_filter = use_optimized_filter

        # Validate and store permutation setting
        self.n_perm = None
        if n_perm is not None:
            if not isinstance(n_perm, int) or n_perm < 1:
                raise ValueError("n_perm must be a positive integer or None.")
            self.n_perm = n_perm

        # Initialize core components
        self.bandpass_filter = self._init_bandpass(
            seq_len,
            fs,
            pha_start_hz,
            pha_end_hz,
            pha_n_bands,
            amp_start_hz,
            amp_end_hz,
            amp_n_bands,
            trainable,
            fp16,
        )
        self.hilbert = Hilbert(seq_len=seq_len, dim=-1, fp16=fp16)
        
        # Initialize MI module based on trainable mode
        if trainable:
            self.modulation_index = DifferentiableModulationIndex(
                n_bins=mi_n_bins,
                temperature=differentiable_mi_temperature,
                binning_method=differentiable_mi_method,
                fp16=fp16,
            )
        else:
            self.modulation_index = ModulationIndex(
                n_bins=mi_n_bins,
                fp16=fp16,
            )

        # Store frequency information
        self.PHA_MIDS_HZ: torch.Tensor
        self.AMP_MIDS_HZ: torch.Tensor
        # Frequency info is set within _init_bandpass

        # Store band counts
        self._pha_n_bands = pha_n_bands
        self._amp_n_bands = amp_n_bands

    def _init_bandpass(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz: float,
        pha_end_hz: float,
        pha_n_bands: int,
        amp_start_hz: float,
        amp_end_hz: float,
        amp_n_bands: int,
        trainable: bool,
        fp16: bool,
    ) -> nn.Module:
        """Initialize bandpass filters with TensorPAC compatibility."""
        # Create frequency bands
        pha_bands = torch.stack(
            [
                torch.linspace(pha_start_hz, pha_end_hz, pha_n_bands + 1)[:-1],
                torch.linspace(pha_start_hz, pha_end_hz, pha_n_bands + 1)[1:],
            ],
            dim=1,
        )
        amp_bands = torch.stack(
            [
                torch.linspace(amp_start_hz, amp_end_hz, amp_n_bands + 1)[:-1],
                torch.linspace(amp_start_hz, amp_end_hz, amp_n_bands + 1)[1:],
            ],
            dim=1,
        )

        # Store frequency centers
        pha_mids = pha_bands.mean(dim=1)
        amp_mids = amp_bands.mean(dim=1)
        self.register_buffer("PHA_MIDS_HZ", pha_mids, persistent=False)
        self.register_buffer("AMP_MIDS_HZ", amp_mids, persistent=False)

        if trainable:
            if not _TRAINABLE_AVAILABLE:
                raise ImportError(
                    "Trainable mode requires TrainableBandPassFilter but import failed."
                )
            return TrainableBandPassFilter(
                seq_len=seq_len,
                fs=fs,
                pha_low_hz=pha_start_hz,
                pha_high_hz=pha_end_hz,
                pha_n_bands=pha_n_bands,
                amp_low_hz=amp_start_hz,
                amp_high_hz=amp_end_hz,
                amp_n_bands=amp_n_bands,
                cycle_pha=self.filter_cycle_pha,
                cycle_amp=self.filter_cycle_amp,
                fp16=fp16,
                filtfilt_mode=self.filtfilt_mode,
                edge_mode=self.edge_mode,
            )
        else:
            FilterClass = OptimizedBandPassFilter if self.use_optimized_filter else BandPassFilter
            kwargs = dict(
                pha_bands=pha_bands,
                amp_bands=amp_bands,
                fs=fs,
                seq_len=seq_len,
                fp16=fp16,
                cycle_pha=self.filter_cycle_pha,
                cycle_amp=self.filter_cycle_amp,
                filtfilt_mode=self.filtfilt_mode,
                edge_mode=self.edge_mode,
            )
            return FilterClass(**kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the full PAC calculation pipeline.

        Args:
            x: Input signal tensor with shape (B, C, Seg, Time)

        Returns:
            dict: Dictionary containing:
                - 'mi': Modulation Index values (B, C, F_pha, F_amp)
                - 'amp_prob': Amplitude probability distribution (B, C, F_pha, F_amp, n_bins)
                - 'pha_bin_centers': Phase bin center values
                - 'pha_freqs_hz': Phase frequency centers
                - 'amp_freqs_hz': Amplitude frequency centers
                - 'mi_z': Z-scored MI values (only if n_perm is not None)
                - 'surrogate_mis': Surrogate MI distributions (only if n_perm is not None)
        """

        # Input preparation
        x = ensure_4d_input(x)
        batch_size, n_chs, n_segments, current_seq_len = x.shape
        device = x.device
        target_dtype = torch.float16 if self.fp16 else torch.float32

        if current_seq_len != self.seq_len:
            warnings.warn(
                f"Input length {current_seq_len} != init length {self.seq_len}. Results may be suboptimal."
            )

        x = x.to(target_dtype)

        # Set gradient tracking context
        grad_context = (
            torch.enable_grad() if self.trainable else torch.no_grad()
        )
        with grad_context:

            # Bandpass filtering
            # Reshape: (B, C, Seg, Time) -> (B * C * Seg, 1, Time)
            x_flat = x.reshape(-1, 1, current_seq_len)
            # Apply filter: (B * C * Seg, 1, N_filters, Time)
            x_filt_flat = self.bandpass_filter(x_flat)
            # Reshape back: (B, C, Seg, N_filters, Time)
            x_filt = x_filt_flat.view(
                batch_size, n_chs, n_segments, -1, current_seq_len
            )

            # Hilbert transform
            # Output: (B, C, Seg, N_filters, Time, 2=[Phase, Amp])
            x_analytic = self.hilbert(x_filt)

            # Extract phase and amplitude bands
            # Phase: (B, C, Seg, n_pha_bands, Time)
            pha = x_analytic[..., : self._pha_n_bands, :, 0]
            # Amplitude: (B, C, Seg, n_amp_bands, Time)
            amp = x_analytic[..., self._pha_n_bands :, :, 1]

            # Permute for Modulation Index: (B, C, Freq, Seg, Time)
            pha = pha.permute(0, 1, 3, 2, 4)
            amp = amp.permute(0, 1, 3, 2, 4)

            # Calculate observed Modulation index
            pac_result = self.modulation_index(pha, amp)

            # Build result dictionary
            result = pac_result
            result.update(
                {
                    "pha_freqs_hz": self.PHA_MIDS_HZ,
                    "amp_freqs_hz": self.AMP_MIDS_HZ,
                }
            )

            # Permutation test
            if self.n_perm is not None:
                observed_mi = pac_result["mi"]
                surrogate_mis = self._generate_surrogates_with_grad(
                    pha, amp, device, target_dtype
                )
                mean_surr = surrogate_mis.mean(dim=0)
                std_surr = surrogate_mis.std(dim=0)
                mi_z = (observed_mi - mean_surr) / (std_surr + 1e-9)
                mask = torch.isfinite(mi_z)
                mi_z = torch.where(mask, mi_z, torch.zeros_like(mi_z))
                result["mi_z"] = mi_z
                result["surrogate_mis"] = surrogate_mis

            return result

    def _generate_surrogates_with_grad(
        self, pha: torch.Tensor, amp: torch.Tensor, device, dtype
    ) -> torch.Tensor:
        batch_size, n_chs, n_amp_bands, n_segments, time_core = amp.shape

        if time_core <= 1:
            warnings.warn("Cannot generate surrogates: sequence length <= 1.")
            dummy_shape = pha.shape[:2] + (pha.shape[2], amp.shape[2])
            return torch.zeros(
                (self.n_perm,) + dummy_shape, dtype=dtype, device=device
            )

        surrogate_results = []
        indices = torch.arange(time_core, device=device)

        for _ in range(self.n_perm):
            shift_shape = (batch_size, n_chs, n_amp_bands, n_segments)
            shifts = torch.randint(
                1, time_core, size=shift_shape, device=device
            )
            shifted_indices = (
                indices.view(1, 1, 1, 1, -1) - shifts.unsqueeze(-1)
            ) % time_core
            amp_shifted = torch.gather(amp, dim=-1, index=shifted_indices)
            surrogate_pac = self.modulation_index(pha, amp_shifted)["mi"]
            surrogate_results.append(surrogate_pac)

        return torch.stack(surrogate_results, dim=0)


# EOF