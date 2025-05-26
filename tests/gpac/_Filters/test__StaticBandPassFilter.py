import pytest
import torch
import sys
import os

# Add path for gpac imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from gpac._Filters._StaticBandPassFilter import StaticBandPassFilter


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_StaticBandPassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 
# import torch
# from ._BaseBandPassFilter import BaseBandPassFilter
# 
# 
# class StaticBandPassFilter(BaseBandPassFilter):
#     """
#     Static bandpass filter with fixed frequency bands.
#     No trainable parameters - purely static filtering.
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
#         band_type="phase",  # "phase", "amplitude", or "both"
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
#             edge_mode: Edge padding mode
#             band_type: Type of bands ("phase", "amplitude", or "both")
#         """
#         super().__init__(fs, seq_len, fp16, filtfilt_mode, edge_mode)
#         
#         # Validate and store bands
#         self.bands = self._validate_bands(bands)
#         self.cycle = cycle
#         self.band_type = band_type
#         
#         # Initialize kernels immediately since bands are static
#         self._initialize_kernels()
# 
#     def _validate_bands(self, bands):
#         """Validate frequency bands."""
#         if not bands:
#             raise ValueError("At least one frequency band must be provided")
#         
#         bands = torch.tensor(bands) if not isinstance(bands, torch.Tensor) else bands
#         nyq = self.fs / 2.0
#         
#         # Clip to valid range
#         bands = torch.clip(bands, 0.1, nyq - 1)
#         
#         # Validate band pairs
#         for ll, hh in bands:
#             if not (0 < ll < hh < nyq):
#                 raise ValueError(f"Invalid band [{ll}, {hh}]. Must satisfy 0 < low < high < {nyq}")
#         
#         return bands.tolist()
# 
#     def get_bands(self):
#         """
#         Get frequency bands for filtering.
#         Returns (pha_bands, amp_bands, cycle_pha, cycle_amp).
#         """
#         if self.band_type == "phase":
#             return self.bands, [], self.cycle, self.cycle
#         elif self.band_type == "amplitude":
#             return [], self.bands, self.cycle, self.cycle
#         elif self.band_type == "both":
#             # Split bands equally between phase and amplitude
#             mid_idx = len(self.bands) // 2
#             pha_bands = self.bands[:mid_idx] if mid_idx > 0 else []
#             amp_bands = self.bands[mid_idx:] if mid_idx < len(self.bands) else []
#             return pha_bands, amp_bands, self.cycle, self.cycle
#         else:
#             raise ValueError(f"Invalid band_type: {self.band_type}. Must be 'phase', 'amplitude', or 'both'")
# 
#     def get_filter_info(self):
#         """Get comprehensive filter information."""
#         pha_bands, amp_bands, cycle_pha, cycle_amp = self.get_bands()
#         return {
#             'fs': self.fs,
#             'seq_len': self.seq_len,
#             'fp16': self.fp16,
#             'filtfilt_mode': self.filtfilt_mode,
#             'edge_mode': self.edge_mode,
#             'band_type': self.band_type,
#             'cycle': self.cycle,
#             'pha_bands': pha_bands,
#             'amp_bands': amp_bands,
#             'n_pha_bands': len(pha_bands),
#             'n_amp_bands': len(amp_bands),
#             'total_bands': len(pha_bands) + len(amp_bands),
#         }
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_StaticBandPassFilter.py
# --------------------------------------------------------------------------------
