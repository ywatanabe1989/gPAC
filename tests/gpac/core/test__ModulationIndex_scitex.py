#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/gpac/core/test__ModulationIndex_scitex.py
"""scitex-grade tests for ``gpac.core._ModulationIndex.ModulationIndex``.

Covers construction-time validation, the per-segment MI shape contract,
amplitude-distribution normalisation, and the seeded surrogate
generator. See ``tests/gpac/test__PAC_scitex.py`` for the rule reference.
"""

import math

import numpy as np
import pytest
import torch

from gpac.core._ModulationIndex import ModulationIndex


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mi_calc():
    """ModulationIndex with default 18 bins."""
    return ModulationIndex(n_bins=18, temperature=0.01, fp16=False)


@pytest.fixture
def coupled_phase_amplitude():
    """Phase / amplitude pair with strong coupling.

    Shape: (batch, channels, freqs, segments, time) = (1, 1, 1, 1, 1024).
    """
    fs = 512.0
    seq_len = 1024
    t_vals = torch.arange(seq_len, dtype=torch.float32) / fs
    phase_signal = 2 * math.pi * 6.0 * t_vals
    # amplitude is modulated by the phase
    amplitude = 1.0 + 0.9 * torch.cos(phase_signal)
    phase_5d = phase_signal.view(1, 1, 1, 1, -1)
    amp_5d = amplitude.view(1, 1, 1, 1, -1)
    return phase_5d, amp_5d


@pytest.fixture
def uncoupled_phase_amplitude():
    """Phase / amplitude with no systematic coupling.

    Phase increases linearly while amplitude is white noise from a fixed
    seed — the resulting MI should be small.
    """
    fs = 512.0
    seq_len = 1024
    t_vals = torch.arange(seq_len, dtype=torch.float32) / fs
    phase_signal = 2 * math.pi * 6.0 * t_vals
    generator = torch.Generator().manual_seed(0)
    amplitude = 1.0 + 0.1 * torch.randn(seq_len, generator=generator)
    phase_5d = phase_signal.view(1, 1, 1, 1, -1)
    amp_5d = amplitude.view(1, 1, 1, 1, -1)
    return phase_5d, amp_5d


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_modulation_index_rejects_non_positive_n_bins():
    # Arrange
    bad_n_bins = 0
    # Act
    ctx = pytest.raises(ValueError, match="n_bins must be positive")
    # Assert
    with ctx:
        ModulationIndex(n_bins=bad_n_bins)


def test_modulation_index_rejects_non_positive_temperature():
    # Arrange
    bad_temperature = -1.0
    # Act
    ctx = pytest.raises(ValueError, match="temperature must be positive")
    # Assert
    with ctx:
        ModulationIndex(n_bins=18, temperature=bad_temperature)


def test_modulation_index_registers_18_phase_bins(mi_calc):
    # Arrange
    mi = mi_calc
    # Act
    n_centers = int(mi.phase_bin_centers.shape[0])
    # Assert
    assert n_centers == 18


def test_modulation_index_phase_bin_edges_span_full_circle(mi_calc):
    # Arrange
    mi = mi_calc
    # Act
    span = float(mi.phase_bins[-1].item() - mi.phase_bins[0].item())
    # Assert
    assert math.isclose(span, 2 * math.pi, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Forward / MI contract
# ---------------------------------------------------------------------------


def test_modulation_index_forward_returns_expected_mi_shape(
    mi_calc, coupled_phase_amplitude
):
    # Arrange
    phase, amplitude = coupled_phase_amplitude
    # Act
    with torch.no_grad():
        out = mi_calc(phase, amplitude)
    # Assert
    assert out["mi"].shape == (1, 1, 1, 1, 1)


def test_modulation_index_is_higher_for_coupled_than_uncoupled(
    mi_calc, coupled_phase_amplitude, uncoupled_phase_amplitude
):
    # Arrange
    phase_c, amp_c = coupled_phase_amplitude
    phase_u, amp_u = uncoupled_phase_amplitude
    # Act
    with torch.no_grad():
        mi_coupled = float(mi_calc(phase_c, amp_c)["mi"].item())
        mi_uncoupled = float(mi_calc(phase_u, amp_u)["mi"].item())
    # Assert
    assert mi_coupled > mi_uncoupled


def test_modulation_index_distributions_sum_to_one(
    mi_calc, coupled_phase_amplitude
):
    # Arrange
    phase, amplitude = coupled_phase_amplitude
    # Act
    with torch.no_grad():
        dist = mi_calc(phase, amplitude, compute_distributions=True)[
            "amplitude_distributions"
        ]
        sums = dist.sum(dim=-1)
    # Assert
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_modulation_index_rejects_mismatched_batch():
    # Arrange
    mi = ModulationIndex(n_bins=18)
    phase = torch.zeros(1, 1, 1, 1, 64)
    amplitude = torch.zeros(2, 1, 1, 1, 64)
    # Act
    ctx = pytest.raises(ValueError, match="Batch size mismatch")
    # Assert
    with ctx:
        mi(phase, amplitude)


def test_modulation_index_rejects_mismatched_time():
    # Arrange
    mi = ModulationIndex(n_bins=18)
    phase = torch.zeros(1, 1, 1, 1, 64)
    amplitude = torch.zeros(1, 1, 1, 1, 32)
    # Act
    ctx = pytest.raises(ValueError, match="Time dimension mismatch")
    # Assert
    with ctx:
        mi(phase, amplitude)


# ---------------------------------------------------------------------------
# Seeded surrogate generation
# ---------------------------------------------------------------------------


def test_compute_surrogates_with_seed_42_is_reproducible(
    mi_calc, coupled_phase_amplitude
):
    # Arrange
    phase, amplitude = coupled_phase_amplitude
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    # Act
    with torch.no_grad():
        out_a = mi_calc.compute_surrogates(
            phase, amplitude, n_perm=4, chunk_size=4, generator=gen_a
        )
        out_b = mi_calc.compute_surrogates(
            phase, amplitude, n_perm=4, chunk_size=4, generator=gen_b
        )
    # Assert
    assert torch.allclose(out_a["surrogate_mean"], out_b["surrogate_mean"], atol=1e-6)


def test_compute_surrogates_different_seeds_diverge(
    mi_calc, coupled_phase_amplitude
):
    # Arrange
    phase, amplitude = coupled_phase_amplitude
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(7)
    # Act
    with torch.no_grad():
        out_a = mi_calc.compute_surrogates(
            phase, amplitude, n_perm=16, chunk_size=4, generator=gen_a
        )
        out_b = mi_calc.compute_surrogates(
            phase, amplitude, n_perm=16, chunk_size=4, generator=gen_b
        )
    # Assert
    assert not torch.allclose(
        out_a["surrogate_mean"], out_b["surrogate_mean"], atol=1e-6
    )


# EOF
