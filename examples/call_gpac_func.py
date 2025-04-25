#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 19:24:52 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/call_gpac_func.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/call_gpac_func.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import gpac
import numpy as np
import torch

# Parameters
FS = 1024
SEQ_LEN = FS * 5
BATCH_SIZE = 4
N_CHS = 16
N_SEGMENTS = 1

# Demo Signal
signal_np = np.random.randn(BATCH_SIZE, N_CHS, N_SEGMENTS, SEQ_LEN).astype(
    np.float32
)
signal_gpu = torch.from_numpy(signal_np).cuda()


# Calculate PAC
try:
    pac_values, freqs_pha, freqs_amp = gpac.calculate_pac(
        signal=signal_gpu,
        fs=FS,
        pha_n_bands=50,
        amp_n_bands=30,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        device="cuda",
        fp16=True,
        n_perm=200,
        # trainable=False,    # Use static filters (default)
        # chunk_size=16       # Process in chunks (optional, if memory is limited)
    )

    print("PAC calculation successful.")
    print(f"Input Signal Shape: {signal_gpu.shape}")  # [4, 16, 50, 30]
    print(f"Output PAC Tensor Shape: {pac_values.shape}")  # [4, 16, 50, 30]
    print(f"Phase Frequencies (Num): {len(freqs_pha)}")
    print(f"Amplitude Frequencies (Num): {len(freqs_amp)}")


except Exception as e:
    print(f"An error occurred during PAC calculation: {e}")

# EOF