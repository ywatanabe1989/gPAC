#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 18:12:11 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_BaseFilter1D.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_BaseFilter1D.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import ensure_3d

class BaseFilter1D(nn.Module):
    def __init__(self, fp16=False, in_place=False):
        super().__init__()
        self.fp16 = fp16
        self.in_place = in_place
        # self.kernels = None

    @abstractmethod
    def init_kernels(
        self,
    ):
        """
        Abstract method to initialize filter kernels.
        Must be implemented by subclasses.
        """
        pass

    def forward(self, x, t=None, edge_len=0):
        """Apply the filter to input signal x with shape: (batch_size, n_chs, seq_len)"""

        # Shape check
        if self.fp16:
            x = x.half()

        x = ensure_3d(x)
        batch_size, n_chs, seq_len = x.shape

        # Kernel Check
        if self.kernels is None:
            raise ValueError("Filter kernels has not been initialized.")

        # Filtering
        x = self.flip_extend(x, self.kernel_size // 2)
        x = self.batch_conv(x, self.kernels, padding=0)
        x = x[..., :seq_len]

        assert x.shape == (
            batch_size,
            n_chs,
            len(self.kernels),
            seq_len,
        ), f"The shape of the filtered signal ({x.shape}) does not match the expected shape: ({batch_size}, {n_chs}, {len(self.kernels)}, {seq_len})."

        # Edge remove
        x = self.remove_edges(x, edge_len)

        if t is None:
            return x
        else:
            t = self.remove_edges(t, edge_len)
            return x, t

    @property
    def kernel_size(
        self,
    ):
        ks = self.kernels.shape[-1]
        # if not ks % 2 == 0:
        #     raise ValueError("Kernel size should be an even number.")
        return ks

    @staticmethod
    def flip_extend(x, extension_length):
        first_segment = x[:, :, :extension_length].flip(dims=[-1])
        last_segment = x[:, :, -extension_length:].flip(dims=[-1])
        return torch.cat([first_segment, x, last_segment], dim=-1)

    @staticmethod
    def batch_conv(x, kernels, padding="same"):
        """
        x: (batch_size, n_chs, seq_len)
        kernels: (n_kernels, seq_len_filt)
        """
        assert x.ndim == 3
        assert kernels.ndim == 2
        batch_size, n_chs, n_time = x.shape
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        kernels = kernels.unsqueeze(1)  # add the channel dimension
        n_kernels = len(kernels)
        filted = F.conv1d(x, kernels.type_as(x), padding=padding)
        return filted.reshape(batch_size, n_chs, n_kernels, -1)

    @staticmethod
    def remove_edges(x, edge_len):
        edge_len = x.shape[-1] // 8 if edge_len == "auto" else edge_len

        if 0 < edge_len:
            return x[..., edge_len:-edge_len]
        else:
            return x


class BandPassFilter(BaseFilter1D):
    def __init__(self, bands, fs, seq_len, fp16=False):
        super().__init__(fp16=fp16)

        self.fp16 = fp16

        # Ensures bands shape
        assert bands.ndim == 2

        # Check bands definitions
        nyq = fs / 2.0
        bands = torch.clip(bands, 0.1, nyq - 1)
        for ll, hh in bands:
            assert 0 < ll
            assert ll < hh
            assert hh < nyq

        # Prepare kernels
        kernels = self.init_kernels(seq_len, fs, bands)
        if fp16:
            kernels = kernels.half()
        self.register_buffer(
            "kernels",
            kernels,
        )

    @staticmethod
    def init_kernels(seq_len, fs, bands):
        filters = [
            design_filter(
                seq_len,
                fs,
                low_hz=ll,
                high_hz=hh,
                is_bandstop=False,
            )
            for ll, hh in bands
        ]

        kernels = zero_pad(filters)
        kernels = ensure_even_len(kernels)
        kernels = torch.tensor(kernels).clone().detach()
        # kernels = kernels.clone().detach().requires_grad_(True)
        return kernels

# EOF