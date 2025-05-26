import pytest
import torch
import sys
import os

# Add path for gpac imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from gpac.v01._StaticBandPassFilter import StaticBandPassFilter


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/v01/_StaticBandPassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 
# import torch
# import torch.nn as nn
# from ._CombinedBandPassFilter_v01_working import CombinedBandPassFilter
# 
# 
# class StaticBandPassFilter(nn.Module):
#     """
#     Static bandpass filter with fixed bands using CombinedBandPassFilter.
#     Simplified interface for basic bandpass filtering operations.
#     """
# 
#     def __init__(
#         self,
#         bands,
#         fs,
#         seq_len,
#         fp16=False,
#         cycle=3,
#         filtfilt_mode=False,
#         edge_mode=None,
#     ):
#         """
#         Initialize static bandpass filter.
#         
#         Args:
#             bands: List of [low_hz, high_hz] frequency pairs
#             fs: Sampling frequency
#             seq_len: Sequence length
#             fp16: Use half precision
#             cycle: Number of cycles for filter design
#             filtfilt_mode: Use zero-phase filtering
#             edge_mode: Edge padding mode ('reflect', 'constant', etc.)
#         """
#         super().__init__()
#         
#         # Convert bands to phase bands (use same bands for simplicity)
#         # For static filtering, we treat all bands the same way
#         self.bands = torch.tensor(bands)
#         
#         # Use CombinedBandPassFilter internally
#         # Pass the same bands for both phase and amplitude
#         self.filter = CombinedBandPassFilter(
#             pha_bands=bands,
#             amp_bands=[],  # Empty amp_bands for static filtering
#             fs=fs,
#             seq_len=seq_len,
#             fp16=fp16,
#             cycle_pha=cycle,
#             cycle_amp=cycle,
#             filtfilt_mode=filtfilt_mode,
#             edge_mode=edge_mode,
#         )
#         
#         self.n_bands = len(bands)
# 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply static bandpass filtering.
#         
#         Args:
#             x: Input tensor (batch, channels, time) or (batch*channel*segment, 1, time)
#             
#         Returns:
#             Filtered tensor (batch*channel*segment, 1, n_bands, time)
#         """
#         # Ensure input is 3D
#         if x.ndim == 2:
#             x = x.unsqueeze(1)
#         
#         # Apply filtering
#         filtered = self.filter(x)
#         
#         return filtered
# 
#     @property
#     def kernels(self):
#         """Access to filter kernels."""
#         return self.filter.kernels
# 
#     def get_band_frequencies(self):
#         """Get the frequency bands."""
#         return self.bands.clone()
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/v01/_StaticBandPassFilter.py
# --------------------------------------------------------------------------------
