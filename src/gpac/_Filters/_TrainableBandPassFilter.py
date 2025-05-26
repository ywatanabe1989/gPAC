#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ._BaseBandPassFilter import BaseBandPassFilter
from ._DifferentiableBandPassFilterDesign import (
    build_differentiable_bandpass_filters,
    TORCHAUDIO_SINC_AVAILABLE
)


class TrainableBandPassFilter(BaseBandPassFilter):
    """
    Trainable bandpass filter with learnable frequency parameters.
    Frequency bands are updated during training via gradient descent.
    """

    def __init__(
        self,
        fs,
        seq_len,
        pha_low_hz=2,
        pha_high_hz=20,
        pha_n_bands=10,
        amp_low_hz=80,
        amp_high_hz=160,
        amp_n_bands=10,
        cycle_pha=3,
        cycle_amp=6,
        fp16=False,
        filtfilt_mode=False,
        edge_mode=None,
    ):
        """
        Initialize trainable bandpass filter.
        
        Args:
            fs: Sampling frequency
            seq_len: Sequence length
            pha_low_hz: Phase band lower bound
            pha_high_hz: Phase band upper bound
            pha_n_bands: Number of phase bands
            amp_low_hz: Amplitude band lower bound
            amp_high_hz: Amplitude band upper bound
            amp_n_bands: Number of amplitude bands
            cycle_pha: Cycles for phase filters
            cycle_amp: Cycles for amplitude filters
            fp16: Use half precision
            filtfilt_mode: Use zero-phase filtering
            edge_mode: Edge padding mode
        """
        super().__init__(fs, seq_len, fp16, filtfilt_mode, edge_mode)
        
        # Store fixed parameters
        self.pha_n_bands = pha_n_bands
        self.amp_n_bands = amp_n_bands
        self.cycle_pha = cycle_pha
        self.cycle_amp = cycle_amp
        
        # Nyquist frequency constraint
        self.nyq = fs / 2.0
        self.min_freq = 0.1
        
        # Create learnable center frequency parameters (like mngs approach)
        if TORCHAUDIO_SINC_AVAILABLE:
            self.pha_mids = nn.Parameter(
                torch.linspace(pha_low_hz, pha_high_hz, pha_n_bands)
            )
            self.amp_mids = nn.Parameter(
                torch.linspace(amp_low_hz, amp_high_hz, amp_n_bands) 
            )
            self._use_differentiable = True
        else:
            # Fallback to boundary parameters if torchaudio not available
            self.pha_low_hz = nn.Parameter(torch.tensor(float(pha_low_hz)))
            self.pha_high_hz = nn.Parameter(torch.tensor(float(pha_high_hz)))
            self.amp_low_hz = nn.Parameter(torch.tensor(float(amp_low_hz)))
            self.amp_high_hz = nn.Parameter(torch.tensor(float(amp_high_hz)))
            self._use_differentiable = False
            print("Warning: torchaudio.prototype not available. Using non-differentiable filter design.")

    def _constrain_mids(self):
        """Constrain center frequency parameters to valid ranges."""
        if self._use_differentiable:
            with torch.no_grad():
                self.pha_mids.clamp_(self.min_freq, self.nyq - 1)
                self.amp_mids.clamp_(self.min_freq, self.nyq - 1)
    
    def _constrain_parameters(self):
        """Constrain frequency parameters to valid ranges (fallback mode)."""
        if not self._use_differentiable:
            with torch.no_grad():
                # Constrain phase parameters
                self.pha_low_hz.clamp_(self.min_freq, self.nyq - 2)
                self.pha_high_hz.clamp_(self.pha_low_hz + 1, self.nyq - 1)
                
                # Constrain amplitude parameters
                self.amp_low_hz.clamp_(self.min_freq, self.nyq - 2)
                self.amp_high_hz.clamp_(self.amp_low_hz + 1, self.nyq - 1)

    def get_bands(self):
        """
        Get frequency bands for filtering.
        Returns (pha_bands, amp_bands, cycle_pha, cycle_amp).
        """
        # Constrain parameters to valid ranges
        self._constrain_parameters()
        
        # Generate phase bands
        pha_bands = []
        if self.pha_n_bands > 0:
            pha_freqs = torch.linspace(
                self.pha_low_hz, self.pha_high_hz, self.pha_n_bands + 1
            )
            for i in range(self.pha_n_bands):
                pha_bands.append([pha_freqs[i].item(), pha_freqs[i + 1].item()])
        
        # Generate amplitude bands
        amp_bands = []
        if self.amp_n_bands > 0:
            amp_freqs = torch.linspace(
                self.amp_low_hz, self.amp_high_hz, self.amp_n_bands + 1
            )
            for i in range(self.amp_n_bands):
                amp_bands.append([amp_freqs[i].item(), amp_freqs[i + 1].item()])
        
        return pha_bands, amp_bands, self.cycle_pha, self.cycle_amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply trainable bandpass filtering with differentiable filter design.
        """
        if self._use_differentiable:
            # Use differentiable filter design
            self._constrain_mids()
            
            # Rebuild kernels dynamically with gradient flow
            kernels = build_differentiable_bandpass_filters(
                self.seq_len, self.fs, self.pha_mids, self.amp_mids, self.cycle_pha
            )
            
            if self.fp16:
                kernels = kernels.half()
            
            # Store kernels temporarily (they'll be rebuilt next forward pass)
            self.kernels = kernels
            
            # Apply filtering using parent logic
            return super().forward(x)
        else:
            # Fallback to static approach
            self._constrain_parameters()
            if not hasattr(self, 'kernels'):
                self._initialize_kernels()
            return super().forward(x)

    def get_current_frequencies(self):
        """Get current frequency parameters."""
        self._constrain_parameters()
        return {
            'pha_low_hz': self.pha_low_hz.item(),
            'pha_high_hz': self.pha_high_hz.item(),
            'amp_low_hz': self.amp_low_hz.item(),
            'amp_high_hz': self.amp_high_hz.item(),
        }

    def get_filter_info(self):
        """Get comprehensive filter information."""
        pha_bands, amp_bands, cycle_pha, cycle_amp = self.get_bands()
        freq_params = self.get_current_frequencies()
        
        return {
            'fs': self.fs,
            'seq_len': self.seq_len,
            'fp16': self.fp16,
            'filtfilt_mode': self.filtfilt_mode,
            'edge_mode': self.edge_mode,
            'pha_n_bands': self.pha_n_bands,
            'amp_n_bands': self.amp_n_bands,
            'cycle_pha': self.cycle_pha,
            'cycle_amp': self.cycle_amp,
            'pha_bands': pha_bands,
            'amp_bands': amp_bands,
            'current_frequencies': freq_params,
            'trainable': True,
        }