#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/conftest.py
"""Shared fixtures for integration tests.

All fixtures return real PyTorch tensors / real ``gpac.PAC`` instances
(no mocks).
"""

import math

import pytest
import torch

from gpac import PAC


@pytest.fixture
def fs():
    """Sampling rate used across the synthetic PAC integration suite."""
    return 512.0


@pytest.fixture
def seq_len():
    """Sequence length used across the synthetic PAC integration suite."""
    return 2048


@pytest.fixture
def coupled_synthetic_signal(fs, seq_len):
    """Synthetic 1-channel signal with strong 6 Hz → 80 Hz PAC.

    Shape ``(batch=1, channels=1, time=seq_len)``.
    """
    t_vals = torch.arange(seq_len, dtype=torch.float32) / fs
    phase_freq = 6.0
    amp_freq = 80.0
    phase = 2 * math.pi * phase_freq * t_vals
    modulation = 1.0 + 0.9 * torch.cos(phase)
    carrier = torch.sin(2 * math.pi * amp_freq * t_vals)
    signal = 0.5 * torch.sin(phase) + modulation * carrier
    return signal.view(1, 1, -1)


@pytest.fixture
def pac_for_synthetic(seq_len, fs):
    """PAC instance configured for the synthetic-signal test (no perms)."""
    return PAC(
        seq_len=seq_len,
        fs=fs,
        pha_range_hz=(2, 20),
        amp_range_hz=(40, 120),
        pha_n_bands=10,
        amp_n_bands=10,
        n_perm=None,
        device_ids=[],
        compile_mode=False,
        random_seed=42,
    )


# EOF
