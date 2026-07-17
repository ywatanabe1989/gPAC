#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_reproducibility.py
"""Integration: same inputs + same seed produces identical outputs.

The neurovista manuscript (PR #52) treats ``random_seed=42`` as the
reproducibility contract for gPAC. These tests pin that contract end-
to-end across two independent PAC invocations on the same synthetic
signal.
"""

import math

import pytest
import torch

from gpac import PAC


pytestmark = pytest.mark.integration


@pytest.fixture
def reproducibility_kwargs():
    """Constructor kwargs used by the reproducibility tests."""
    return {
        "seq_len": 1024,
        "fs": 512.0,
        "pha_range_hz": (4, 12),
        "amp_range_hz": (60, 100),
        "pha_n_bands": 4,
        "amp_n_bands": 4,
        "n_perm": 8,
        "device_ids": [],
        "compile_mode": False,
        "random_seed": 42,
    }


@pytest.fixture
def reproducibility_signal():
    """Coupled (6 Hz → 80 Hz) signal used for the reproducibility tests."""
    fs = 512.0
    seq_len = 1024
    t_vals = torch.arange(seq_len, dtype=torch.float32) / fs
    phase = 2 * math.pi * 6.0 * t_vals
    modulation = 1.0 + 0.7 * torch.cos(phase)
    carrier = torch.sin(2 * math.pi * 80.0 * t_vals)
    signal = 0.5 * torch.sin(phase) + modulation * carrier
    return signal.view(1, 1, -1)


def test_two_pac_invocations_with_same_seed_match_raw_pac(
    reproducibility_kwargs, reproducibility_signal
):
    # Arrange
    pac_a = PAC(**reproducibility_kwargs)
    pac_b = PAC(**reproducibility_kwargs)
    # Act
    with torch.no_grad():
        out_a = pac_a(reproducibility_signal)["pac"]
        out_b = pac_b(reproducibility_signal)["pac"]
    # Assert
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_two_pac_invocations_with_same_seed_match_z_scores(
    reproducibility_kwargs, reproducibility_signal
):
    # Arrange
    pac_a = PAC(**reproducibility_kwargs)
    pac_b = PAC(**reproducibility_kwargs)
    # Act
    with torch.no_grad():
        z_a = pac_a(reproducibility_signal)["pac_z"]
        z_b = pac_b(reproducibility_signal)["pac_z"]
    # Assert
    assert torch.allclose(z_a, z_b, atol=1e-6)


# EOF
