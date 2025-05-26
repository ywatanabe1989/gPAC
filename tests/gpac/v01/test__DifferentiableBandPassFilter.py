import pytest
import torch
import sys
import os

# Add path for gpac imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from gpac.v01._DifferentiableBandPassFilter import DifferentiableBandPassFilter


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/v01/_DifferentiableBandPassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 
# import torch
# import torch.nn as nn
# from ._CombinedBandPassFilter_v01_working import CombinedBandPassFilter
# 
# 
# class DifferentiableBandPassFilter(nn.Module):
#     """
#     Differentiable bandpass filter with learnable frequency parameters.
#     Uses CombinedBandPassFilter as the underlying filtering mechanism.
#     """
# 
#     def __init__(
#         self,
#         seq_len,
#         fs,
#         pha_low_hz=2,
#         pha_high_hz=20,
#         pha_n_bands=10,
#         amp_low_hz=80,
#         amp_high_hz=160,
#         amp_n_bands=10,
#         cycle_pha=3,
#         cycle_amp=6,
#         fp16=False,
#         filtfilt_mode=False,
#         edge_mode=None,
#     ):
#         """
#         Initialize differentiable bandpass filter.
#         
#         Args:
#             seq_len: Sequence length
#             fs: Sampling frequency
#             pha_low_hz: Phase band lower bound
#             pha_high_hz: Phase band upper bound
#             pha_n_bands: Number of phase bands
#             amp_low_hz: Amplitude band lower bound
#             amp_high_hz: Amplitude band upper bound
#             amp_n_bands: Number of amplitude bands
#             cycle_pha: Cycles for phase filters
#             cycle_amp: Cycles for amplitude filters
#             fp16: Use half precision
#             filtfilt_mode: Use zero-phase filtering
#             edge_mode: Edge padding mode
#         """
#         super().__init__()
#         
#         self.seq_len = seq_len
#         self.fs = fs
#         self.fp16 = fp16
#         self.cycle_pha = cycle_pha
#         self.cycle_amp = cycle_amp
#         self.filtfilt_mode = filtfilt_mode
#         self.edge_mode = edge_mode
#         
#         # Nyquist frequency constraint
#         nyq = fs / 2.0
#         
#         # Learnable frequency parameters
#         self.pha_low_hz = nn.Parameter(torch.tensor(float(pha_low_hz)))
#         self.pha_high_hz = nn.Parameter(torch.tensor(float(pha_high_hz)))
#         self.amp_low_hz = nn.Parameter(torch.tensor(float(amp_low_hz)))
#         self.amp_high_hz = nn.Parameter(torch.tensor(float(amp_high_hz)))
#         
#         # Fixed number of bands
#         self.pha_n_bands = pha_n_bands
#         self.amp_n_bands = amp_n_bands
#         
#         # Constraints
#         self.nyq = nyq
#         self.min_freq = 0.1
#         
#         # Initialize filter (will be rebuilt in forward)
#         self._current_filter = None
# 
#     def _constrain_parameters(self):
#         """Constrain frequency parameters to valid ranges."""
#         with torch.no_grad():
#             self.pha_low_hz.clamp_(self.min_freq, self.nyq - 2)
#             self.pha_high_hz.clamp_(self.pha_low_hz + 1, self.nyq - 1)
#             self.amp_low_hz.clamp_(self.min_freq, self.nyq - 2)
#             self.amp_high_hz.clamp_(self.amp_low_hz + 1, self.nyq - 1)
# 
#     def _generate_bands(self):
#         """Generate frequency bands from learnable parameters."""
#         # Generate phase bands
#         pha_freqs = torch.linspace(
#             self.pha_low_hz, self.pha_high_hz, self.pha_n_bands + 1
#         )
#         pha_bands = []
#         for i in range(self.pha_n_bands):
#             pha_bands.append([pha_freqs[i].item(), pha_freqs[i + 1].item()])
#         
#         # Generate amplitude bands
#         amp_freqs = torch.linspace(
#             self.amp_low_hz, self.amp_high_hz, self.amp_n_bands + 1
#         )
#         amp_bands = []
#         for i in range(self.amp_n_bands):
#             amp_bands.append([amp_freqs[i].item(), amp_freqs[i + 1].item()])
#         
#         return pha_bands, amp_bands
# 
#     def _rebuild_filter(self):
#         """Rebuild the filter with current parameters."""
#         self._constrain_parameters()
#         pha_bands, amp_bands = self._generate_bands()
#         
#         self._current_filter = CombinedBandPassFilter(
#             pha_bands=pha_bands,
#             amp_bands=amp_bands,
#             fs=self.fs,
#             seq_len=self.seq_len,
#             fp16=self.fp16,
#             cycle_pha=self.cycle_pha,
#             cycle_amp=self.cycle_amp,
#             filtfilt_mode=self.filtfilt_mode,
#             edge_mode=self.edge_mode,
#         )
# 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply differentiable bandpass filtering.
#         
#         Args:
#             x: Input tensor (batch, channels, time) or (batch*channel*segment, 1, time)
#             
#         Returns:
#             Filtered tensor (batch*channel*segment, 1, n_bands, time)
#         """
#         # Rebuild filter with current parameters
#         self._rebuild_filter()
#         
#         # Ensure input is 3D
#         if x.ndim == 2:
#             x = x.unsqueeze(1)
#         
#         # Apply filtering
#         filtered = self._current_filter(x)
#         
#         return filtered
# 
#     def get_current_bands(self):
#         """Get current frequency bands."""
#         self._constrain_parameters()
#         return self._generate_bands()
# 
#     @property
#     def kernels(self):
#         """Access to current filter kernels."""
#         if self._current_filter is None:
#             self._rebuild_filter()
#         return self._current_filter.kernels
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/v01/_DifferentiableBandPassFilter.py
# --------------------------------------------------------------------------------
