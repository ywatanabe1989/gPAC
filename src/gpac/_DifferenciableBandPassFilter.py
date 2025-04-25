#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 18:12:27 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_DifferenciableBandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_DifferenciableBandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import torch
import torch.nn as nn

from ._utils import (
    TORCHAUDIO_SINC_AVAILABLE,
    build_bandpass_filters,
    init_bandpass_filters,
)
from ._BaseFilter1D import BaseFilter1D

class DifferentiableBandPassFilter(BaseFilter1D):
    def __init__(
        self,
        sig_len,
        fs,
        pha_low_hz=2,
        pha_high_hz=20,
        pha_n_bands=30,
        amp_low_hz=80,
        amp_high_hz=160,
        amp_n_bands=50,
        cycle=3,
        fp16=False,
    ):
        super().__init__(fp16=fp16)

        # Attributes
        self.pha_low_hz = pha_low_hz
        self.pha_high_hz = pha_high_hz
        self.amp_low_hz = amp_low_hz
        self.amp_high_hz = amp_high_hz
        self.sig_len = sig_len
        self.fs = fs
        self.cycle = cycle
        self.fp16 = fp16

        # Check bands definitions
        nyq = fs / 2.0
        pha_high_hz = torch.tensor(pha_high_hz).clip(0.1, nyq - 1)
        pha_low_hz = torch.tensor(pha_low_hz).clip(0.1, pha_high_hz - 1)
        amp_high_hz = torch.tensor(amp_high_hz).clip(0.1, nyq - 1)
        amp_low_hz = torch.tensor(amp_low_hz).clip(0.1, amp_high_hz - 1)

        assert pha_low_hz < pha_high_hz < nyq
        assert amp_low_hz < amp_high_hz < nyq

        # Prepare kernels
        self.init_kernels = init_bandpass_filters
        self.build_bandpass_filters = build_bandpass_filters
        kernels, self.pha_mids, self.amp_mids = self.init_kernels(
            sig_len=sig_len,
            fs=fs,
            pha_low_hz=pha_low_hz,
            pha_high_hz=pha_high_hz,
            pha_n_bands=pha_n_bands,
            amp_low_hz=amp_low_hz,
            amp_high_hz=amp_high_hz,
            amp_n_bands=amp_n_bands,
            cycle=cycle,
        )

        self.register_buffer(
            "kernels",
            kernels,
        )
        # self.register_buffer("pha_mids", pha_mids)
        # self.register_buffer("amp_mids", amp_mids)
        # self.pha_mids = nn.Parameter(pha_mids.detach())
        # self.amp_mids = nn.Parameter(amp_mids.detach())

        if fp16:
            self.kernels = self.kernels.half()
            # self.pha_mids = self.pha_mids.half()
            # self.amp_mids = self.amp_mids.half()

    def forward(self, x, t=None, edge_len=0):
        # Constrains the parameter spaces
        torch.clip(self.pha_mids, self.pha_low_hz, self.pha_high_hz)
        torch.clip(self.amp_mids, self.amp_low_hz, self.amp_high_hz)

        self.kernels = self.build_bandpass_filters(
            self.sig_len, self.fs, self.pha_mids, self.amp_mids, self.cycle
        )
        return super().forward(x=x, t=t, edge_len=edge_len)

# EOF