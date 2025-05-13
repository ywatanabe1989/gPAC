#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 19:04:07 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_BandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py

import torch

from ._BaseFilter1D import BaseFilter1D
from ._utils import design_filter, ensure_even_len, zero_pad


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
        # Use clone().detach() directly instead of torch.tensor() to avoid warning
        kernels = kernels.clone().detach()
        return kernels

# EOF