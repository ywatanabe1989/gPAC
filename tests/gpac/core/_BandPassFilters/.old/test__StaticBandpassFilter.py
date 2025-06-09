# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_StaticBandpassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-28 18:27:19 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_filter/_StaticBandpassFilter.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_filter/_StaticBandpassFilter.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import torch
# import torch.nn.functional as F
# from scipy.signal import firwin
#
# from ._BaseFilter1D import BaseFilter1D
#
#
# class StaticBandPassFilter(BaseFilter1D):
#     def __init__(self, bands, fs, seq_len, fp16=False):
#         super().__init__(fp16=fp16)
#         self.fp16 = fp16
#         # Ensures bands shape
#         assert bands.ndim == 2
#         # Check bands definitions
#         nyq = fs / 2.0
#         bands = torch.clip(bands, 0.1, nyq - 1)
#         for ll, hh in bands:
#             assert 0 < ll
#             assert ll < hh
#             assert hh < nyq
#         # Prepare kernels
#         kernels = self.init_kernels(seq_len, fs, bands)
#         if fp16:
#             kernels = kernels.half()
#         self.register_buffer(
#             "kernels",
#             kernels,
#         )
#
#     @staticmethod
#     def zero_pad_1d(x, target_length):
#         padding_needed = target_length - len(x)
#         padding_left = padding_needed // 2
#         padding_right = padding_needed - padding_left
#         return F.pad(x, (padding_left, padding_right), "constant", 0)
#
#     @staticmethod
#     def zero_pad(xs, dim=0):
#         max_len = max([len(x) for x in xs])
#         return torch.stack(
#             [StaticBandPassFilter.zero_pad_1d(x, max_len) for x in xs], dim=dim
#         )
#
#     @staticmethod
#     def ensure_even_len(x):
#         if x.shape[-1] % 2 == 0:
#             return x
#         else:
#             return x[..., :-1]
#
#     @staticmethod
#     def init_kernels(seq_len, fs, bands):
#         filters = [
#             StaticBandPassFilter._design_fir_filter(
#                 seq_len,
#                 fs,
#                 low_hz=ll,
#                 high_hz=hh,
#                 is_bandstop=False,
#             )
#             for ll, hh in bands
#         ]
#         kernels = StaticBandPassFilter.zero_pad(filters)
#         kernels = StaticBandPassFilter.ensure_even_len(kernels)
#         kernels = torch.tensor(kernels).clone().detach()
#         return kernels
#
#     @staticmethod
#     def _design_fir_filter(
#         sig_len, fs, low_hz=None, high_hz=None, cycle=3, is_bandstop=False
#     ):
#         """
#         Designs a Finite Impulse Response (FIR) filter based on the specified parameters.
#
#         Arguments:
#         - sig_len (int): Length of the signal for which the filter is being designed.
#         - fs (int): Sampling frequency of the signal.
#         - low_hz (float, optional): Low cutoff frequency for the filter. Required for lowpass and bandpass filters.
#         - high_hz (float, optional): High cutoff frequency for the filter. Required for highpass and bandpass filters.
#         - cycle (int, optional): Number of cycles to use in determining the filter order. Defaults to 3.
#         - is_bandstop (bool, optional): Specifies if the filter should be a bandstop filter. Defaults to False.
#
#         Returns:
#         - The coefficients of the designed FIR filter.
#
#         Raises:
#         - FilterParameterError: If the provided parameters are invalid.
#         """
#
#         class FilterParameterError(Exception):
#             """Custom Exception for invalid filter parameters."""
#
#             pass
#
#         def estimate_filter_type(low_hz=None, high_hz=None, is_bandstop=False):
#             """
#             Estimates the filter type based on the provided low and high cutoff frequencies,
#             and whether a bandstop filter is desired. Raises an Exception for invalid configurations.
#             """
#             if low_hz is not None and low_hz < 0:
#                 raise FilterParameterError("low_hz must be non-negative.")
#             if high_hz is not None and high_hz < 0:
#                 raise FilterParameterError("high_hz must be non-negative.")
#             if (
#                 low_hz is not None
#                 and high_hz is not None
#                 and low_hz >= high_hz
#             ):
#                 raise FilterParameterError(
#                     "low_hz must be less than high_hz for valid configurations."
#                 )
#
#             if low_hz is not None and high_hz is not None:
#                 return "bandstop" if is_bandstop else "bandpass"
#             elif low_hz is not None:
#                 return "lowpass"
#             elif high_hz is not None:
#                 return "highpass"
#             else:
#                 raise FilterParameterError(
#                     "At least one of low_hz or high_hz must be provided."
#                 )
#
#         def determine_cutoff_frequencies(filter_mode, low_hz, high_hz):
#             if filter_mode in ["lowpass", "highpass"]:
#                 cutoff = low_hz if filter_mode == "lowpass" else high_hz
#             else:  # 'bandpass' or 'bandstop'
#                 cutoff = [low_hz, high_hz]
#             return cutoff
#
#         def determine_low_freq(filter_mode, low_hz, high_hz):
#             if filter_mode in ["lowpass", "bandstop"]:
#                 low_freq = low_hz
#             else:  # 'highpass' or 'bandpass'
#                 low_freq = (
#                     high_hz
#                     if filter_mode == "highpass"
#                     else min(low_hz, high_hz)
#                 )
#             return low_freq
#
#         def determine_order(filter_mode, fs, low_freq, sig_len, cycle):
#             order = cycle * int((fs // low_freq))
#             if 3 * order < sig_len:
#                 order = (sig_len - 1) // 3
#             order = _to_even(order)
#             return order
#
#         def _to_even(n):
#             return int(n) - (int(n) % 2)
#
#         fs = int(fs)
#         low_hz = float(low_hz) if low_hz is not None else low_hz
#         high_hz = float(high_hz) if high_hz is not None else high_hz
#         filter_mode = estimate_filter_type(low_hz, high_hz, is_bandstop)
#         cutoff = determine_cutoff_frequencies(filter_mode, low_hz, high_hz)
#         low_freq = determine_low_freq(filter_mode, low_hz, high_hz)
#         order = determine_order(filter_mode, fs, low_freq, sig_len, cycle)
#         numtaps = order + 1
#
#         return firwin(
#             numtaps=numtaps,
#             cutoff=cutoff,
#             pass_zero=(filter_mode in ["highpass", "bandstop"]),
#             window="hamming",
#             fs=fs,
#             scale=True,
#         )
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_StaticBandpassFilter.py
# --------------------------------------------------------------------------------
