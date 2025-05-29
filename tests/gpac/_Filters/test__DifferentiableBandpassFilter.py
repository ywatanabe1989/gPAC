# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_DifferentiableBandpassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-28 18:49:51 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_filter/_DifferentiableBandpassFilter.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_filter/_DifferentiableBandpassFilter.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import torch
# import torch.nn as nn
# from torchaudio.prototype.functional import sinc_impulse_response
#
# from ._BaseFilter1D import BaseFilter1D
#
#
# class DifferentiableBandPassFilter(BaseFilter1D):
#     def __init__(
#         self,
#         sig_len,
#         fs,
#         pha_low_hz=2,
#         pha_high_hz=20,
#         pha_n_bands=30,
#         amp_low_hz=80,
#         amp_high_hz=160,
#         amp_n_bands=50,
#         cycle=3,
#         fp16=False,
#     ):
#         super().__init__(fp16=fp16)
#
#         # Attributes
#         self.pha_low_hz = pha_low_hz
#         self.pha_high_hz = pha_high_hz
#         self.amp_low_hz = amp_low_hz
#         self.amp_high_hz = amp_high_hz
#         self.sig_len = sig_len
#         self.fs = fs
#         self.cycle = cycle
#         self.fp16 = fp16
#
#         # Validate frequency bounds
#         nyq = fs / 2.0
#         assert pha_low_hz < pha_high_hz < nyq
#         assert amp_low_hz < amp_high_hz < nyq
#
#         # Initialize learnable parameters and kernels
#         kernels = self.init_kernels(
#             sig_len=sig_len,
#             fs=fs,
#             pha_low_hz=pha_low_hz,
#             pha_high_hz=pha_high_hz,
#             pha_n_bands=pha_n_bands,
#             amp_low_hz=amp_low_hz,
#             amp_high_hz=amp_high_hz,
#             amp_n_bands=amp_n_bands,
#             cycle=cycle,
#         )
#
#         self.register_buffer(
#             "kernels",
#             kernels,
#         )
#
#         if fp16:
#             self.kernels = self.kernels.half()
#
#     def init_kernels(
#         self,
#         sig_len,
#         fs,
#         pha_low_hz,
#         pha_high_hz,
#         pha_n_bands,
#         amp_low_hz,
#         amp_high_hz,
#         amp_n_bands,
#         cycle,
#     ):
#         self._pha_mids_raw = nn.Parameter(torch.zeros(pha_n_bands))
#         self._amp_mids_raw = nn.Parameter(torch.zeros(amp_n_bands))
#
#         filters = self.rebuild_differentiable_bandpass_filters(
#             sig_len, fs, self.pha_mids, self.amp_mids, cycle
#         )
#         return filters
#
#     @staticmethod
#     def rebuild_differentiable_bandpass_filters(
#         sig_len, fs, pha_mids, amp_mids, cycle
#     ):
#
#         def _to_even(n):
#             return int(n) - (int(n) % 2)
#
#         def _to_odd(n):
#             return int(n) - ((int(n) + 1) % 2)
#
#         def _freqs_to_freq_bands(mids, factor):
#             lows = mids - mids / factor
#             highs = mids + mids / factor
#             return lows, highs
#
#         def _define_order(low_hz, fs, sig_len, cycle):
#             order = cycle * int((fs // low_hz))
#             order = order if 3 * order >= sig_len else (sig_len - 1) // 3
#             order = _to_even(order)
#             return order
#
#         def _create_differentiable_bandpass_filters(
#             lows_hz, highs_hz, fs, order
#         ):
#             nyq = fs / 2.0
#             order = _to_odd(order)
#             irs_ll = sinc_impulse_response(lows_hz / nyq, window_size=order)
#             irs_hh = sinc_impulse_response(highs_hz / nyq, window_size=order)
#             irs = irs_ll - irs_hh
#             return irs
#
#         # Main
#         pha_lows, pha_highs = _freqs_to_freq_bands(pha_mids, factor=4.0)
#         amp_lows, amp_highs = _freqs_to_freq_bands(amp_mids, factor=8.0)
#
#         # Order
#         lowest = min(pha_lows.min().item(), amp_lows.min().item())
#         order = _define_order(lowest, fs, sig_len, cycle)
#
#         # Order
#         pha_bp_filters = _create_differentiable_bandpass_filters(
#             pha_lows, pha_highs, fs, order
#         )
#         amp_bp_filters = _create_differentiable_bandpass_filters(
#             amp_lows, amp_highs, fs, order
#         )
#         return torch.vstack([pha_bp_filters, amp_bp_filters])
#
#     def forward(self, x, t=None, edge_len=0):
#         # Rebuild kernels with current parameters
#         self.kernels = self.rebuild_differentiable_bandpass_filters(
#             self.sig_len, self.fs, self.pha_mids, self.amp_mids, self.cycle
#         )
#         return super().forward(x=x, t=t, edge_len=edge_len)
#
#     @property
#     def pha_mids(self):
#         # Constrains the parameter spaces
#         return self.pha_low_hz + torch.sigmoid(self._pha_mids_raw) * (
#             self.pha_high_hz - self.pha_low_hz
#         )
#
#     @property
#     def amp_mids(self):
#         # Constrains the parameter spaces
#         return self.amp_low_hz + torch.sigmoid(self._amp_mids_raw) * (
#             self.amp_high_hz - self.amp_low_hz
#         )
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_DifferentiableBandpassFilter.py
# --------------------------------------------------------------------------------
