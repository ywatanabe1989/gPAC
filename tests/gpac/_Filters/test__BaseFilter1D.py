# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_BaseFilter1D.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-28 19:02:47 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_Filters/_BaseFilter1D.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_Filters/_BaseFilter1D.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Implements various neural network filter layers:
#     - BaseFilter1D: Abstract base class for 1D filters
#     - BandPassFilter: WRITE DESCRIPTION HERE
#     - DifferentiableBandPassFilter: Implements learnable bandpass filtering
# """
#
# # Imports
# from abc import abstractmethod
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BaseFilter1D(nn.Module):
#     def __init__(self, fp16=False, in_place=False):
#         super().__init__()
#         self.fp16 = fp16
#         self.in_place = in_place
#
#     @abstractmethod
#     def init_kernels(
#         self,
#     ):
#         """
#         Abstract method to initialize filter kernels.
#         Must be implemented by subclasses.
#         """
#         pass
#
#     def forward(self, x, t=None, edge_len=0):
#         """Apply the filter to input signal x with shape: (batch_size, n_chs, seq_len)"""
#
#         # Shape check
#         if self.fp16:
#             x = x.half()
#
#         x = self.ensure_3d(x)
#         batch_size, n_chs, seq_len = x.shape
#
#         # Kernel Check
#         if self.kernels is None:
#             raise ValueError("Filter kernels has not been initialized.")
#
#         # Filtering
#         x = self.flip_extend(x, self.kernel_size // 2)
#         x = self.batch_conv(x, self.kernels, padding=0)
#         x = x[..., :seq_len]
#
#         assert x.shape == (
#             batch_size,
#             n_chs,
#             len(self.kernels),
#             seq_len,
#         ), f"The shape of the filtered signal ({x.shape}) does not match the expected shape: ({batch_size}, {n_chs}, {len(self.kernels)}, {seq_len})."
#
#         # Edge remove
#         x = self.remove_edges(x, edge_len)
#
#         if t is None:
#             return x
#         else:
#             t = self.remove_edges(t, edge_len)
#             return x, t
#
#     @staticmethod
#     def ensure_3d(x):
#         """
#         Ensure input tensor is 3D with shape (batch_size, n_chs, seq_len).
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor
#
#         Returns
#         -------
#         torch.Tensor
#             3D tensor with shape (batch_size, n_chs, seq_len)
#         """
#         if x.ndim == 2:
#             # Assume shape is (batch_size, seq_len), add channel dimension
#             x = x.unsqueeze(1)
#         elif x.ndim == 1:
#             # Assume shape is (seq_len,), add batch and channel dimensions
#             x = x.unsqueeze(0).unsqueeze(0)
#         elif x.ndim > 3:
#             raise ValueError(
#                 f"Input tensor has too many dimensions: {x.shape}"
#             )
#
#         return x
#
#     @property
#     def kernel_size(
#         self,
#     ):
#         ks = self.kernels.shape[-1]
#         return ks
#
#     @staticmethod
#     def flip_extend(x, extension_length):
#         first_segment = x[:, :, :extension_length].flip(dims=[-1])
#         last_segment = x[:, :, -extension_length:].flip(dims=[-1])
#         return torch.cat([first_segment, x, last_segment], dim=-1)
#
#     @staticmethod
#     def batch_conv(x, kernels, padding="same"):
#         """
#         x: (batch_size, n_chs, seq_len)
#         kernels: (n_kernels, seq_len_filt)
#         """
#         assert x.ndim == 3
#         assert kernels.ndim == 2
#         batch_size, n_chs, n_time = x.shape
#         x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
#         kernels = kernels.unsqueeze(1)  # add the channel dimension
#         n_kernels = len(kernels)
#         filted = F.conv1d(x, kernels.type_as(x), padding=padding)
#         return filted.reshape(batch_size, n_chs, n_kernels, -1)
#
#     @staticmethod
#     def remove_edges(x, edge_len):
#         edge_len = x.shape[-1] // 8 if edge_len == "auto" else edge_len
#
#         if 0 < edge_len:
#             return x[..., edge_len:-edge_len]
#         else:
#             return x
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Filters/_BaseFilter1D.py
# --------------------------------------------------------------------------------
