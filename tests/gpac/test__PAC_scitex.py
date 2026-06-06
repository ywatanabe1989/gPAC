#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/gpac/test__PAC_scitex.py
"""scitex-grade tests for ``gpac.PAC``.

Conventions (STX-TQ001-007, STX-NM001-003):

- Each test follows the Arrange / Act / Assert pattern with marker
  comments on their own lines.
- Each test exercises a single behaviour and contains exactly one
  assertion (``assert`` statement or ``pytest.raises`` block).
- No ``unittest.mock``, ``pytest-mock``, or ``monkeypatch`` is used —
  every collaborator is the real PyTorch object the production code
  uses.
"""

import numpy as np
import pytest
import torch

from gpac import PAC


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_pac_kwargs():
    """Minimal PAC constructor kwargs for fast CPU tests."""
    return {
        "seq_len": 512,
        "fs": 512.0,
        "pha_range_hz": (4, 12),
        "amp_range_hz": (60, 100),
        "pha_n_bands": 3,
        "amp_n_bands": 3,
        "n_perm": None,
        "device_ids": [],
        "compile_mode": False,
        "random_seed": 42,
    }


@pytest.fixture
def small_pac(small_pac_kwargs):
    """Fresh PAC instance with no surrogates and no compile."""
    return PAC(**small_pac_kwargs)


@pytest.fixture
def small_pac_with_perms(small_pac_kwargs):
    """PAC instance with a small number of permutations (seeded)."""
    kwargs = dict(small_pac_kwargs)
    kwargs["n_perm"] = 8
    return PAC(**kwargs)


@pytest.fixture
def random_signal_3d():
    """Deterministic 3-D input (batch, channels, time)."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 512, generator=generator)


@pytest.fixture
def random_signal_4d():
    """Deterministic 4-D input (batch, channels, segments, time)."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 2, 512, generator=generator)


# ---------------------------------------------------------------------------
# Construction and parameter validation
# ---------------------------------------------------------------------------


def test_pac_constructor_accepts_minimal_kwargs(small_pac_kwargs):
    # Arrange
    kwargs = small_pac_kwargs
    # Act
    pac = PAC(**kwargs)
    # Assert
    assert isinstance(pac, PAC)


def test_pac_constructor_rejects_non_positive_seq_len(small_pac_kwargs):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["seq_len"] = 0
    # Act
    ctx = pytest.raises(ValueError, match="seq_len must be positive")
    # Assert
    with ctx:
        PAC(**kwargs)


def test_pac_constructor_rejects_non_positive_sampling_rate(small_pac_kwargs):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["fs"] = -1.0
    # Act
    ctx = pytest.raises(ValueError, match="fs must be positive")
    # Assert
    with ctx:
        PAC(**kwargs)


def test_pac_constructor_rejects_non_integer_random_seed(small_pac_kwargs):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["random_seed"] = "not-an-int"
    # Act
    ctx = pytest.raises(ValueError, match="random_seed must be an integer or None")
    # Assert
    with ctx:
        PAC(**kwargs)


def test_pac_default_random_seed_is_42():
    # Arrange
    seq_len, fs = 512, 512.0
    # Act
    pac = PAC(seq_len=seq_len, fs=fs, device_ids=[], compile_mode=False)
    # Assert
    assert pac.random_seed == 42


def test_pac_random_seed_none_yields_no_generator():
    # Arrange
    seq_len, fs = 512, 512.0
    # Act
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        random_seed=None,
        device_ids=[],
        compile_mode=False,
    )
    # Assert
    assert pac.generator is None


# ---------------------------------------------------------------------------
# Forward / output-shape contract
# ---------------------------------------------------------------------------


def test_pac_forward_with_3d_input_returns_4d_pac(small_pac, random_signal_3d):
    # Arrange
    pac = small_pac
    signal = random_signal_3d
    # Act
    with torch.no_grad():
        results = pac(signal)
    # Assert
    assert results["pac"].shape == (2, 3, 3, 3)


def test_pac_forward_with_4d_input_keeps_segment_dim(small_pac, random_signal_4d):
    # Arrange
    pac = small_pac
    signal = random_signal_4d
    # Act
    with torch.no_grad():
        results = pac(signal)
    # Assert
    assert results["pac"].shape == (2, 3, 2, 3, 3)


def test_pac_forward_with_2d_input_raises_value_error(small_pac):
    # Arrange
    bad_signal = torch.randn(3, 512)
    # Act
    ctx = pytest.raises(ValueError, match="Input must be 3D or 4D")
    # Assert
    with ctx:
        small_pac(bad_signal)


def test_pac_returns_phase_band_array_with_expected_length(small_pac):
    # Arrange
    pac = small_pac
    # Act
    bands = pac.pha_bands_hz
    # Assert
    assert bands.shape[0] == 3


def test_pac_returns_amplitude_band_array_with_expected_length(small_pac):
    # Arrange
    pac = small_pac
    # Act
    bands = pac.amp_bands_hz
    # Assert
    assert bands.shape[0] == 3


def test_pac_values_are_non_negative(small_pac, random_signal_3d):
    # Arrange
    pac = small_pac
    signal = random_signal_3d
    # Act
    with torch.no_grad():
        pac_values = pac(signal)["pac"]
    # Assert
    assert float(pac_values.min().item()) >= 0.0


# ---------------------------------------------------------------------------
# Surrogate generation / reproducibility
# ---------------------------------------------------------------------------


def test_pac_with_seed_42_produces_finite_z_scores(
    small_pac_with_perms, random_signal_3d
):
    # Arrange
    pac = small_pac_with_perms
    signal = random_signal_3d
    # Act
    with torch.no_grad():
        z = pac(signal)["pac_z"]
    # Assert
    assert torch.isfinite(z).all().item()


def test_pac_same_seed_produces_identical_z_scores(
    small_pac_kwargs, random_signal_3d
):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["n_perm"] = 8
    pac1 = PAC(**kwargs)
    pac2 = PAC(**kwargs)
    # Act
    with torch.no_grad():
        z1 = pac1(random_signal_3d)["pac_z"]
        z2 = pac2(random_signal_3d)["pac_z"]
    # Assert
    assert torch.allclose(z1, z2, atol=1e-6)


def test_pac_same_seed_produces_identical_raw_pac(
    small_pac_kwargs, random_signal_3d
):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["n_perm"] = 8
    pac1 = PAC(**kwargs)
    pac2 = PAC(**kwargs)
    # Act
    with torch.no_grad():
        pac_a = pac1(random_signal_3d)["pac"]
        pac_b = pac2(random_signal_3d)["pac"]
    # Assert
    assert torch.allclose(pac_a, pac_b, atol=1e-6)


def test_pac_distinct_seeds_yield_distinct_z_scores(
    small_pac_kwargs, random_signal_3d
):
    # Arrange
    kwargs1 = dict(small_pac_kwargs, n_perm=8, random_seed=42)
    kwargs2 = dict(small_pac_kwargs, n_perm=8, random_seed=123)
    pac1 = PAC(**kwargs1)
    pac2 = PAC(**kwargs2)
    # Act
    with torch.no_grad():
        z1 = pac1(random_signal_3d)["pac_z"]
        z2 = pac2(random_signal_3d)["pac_z"]
    # Assert
    assert not torch.allclose(z1, z2, atol=1e-3)


def test_pac_distinct_seeds_preserve_raw_pac(
    small_pac_kwargs, random_signal_3d
):
    # Arrange
    kwargs1 = dict(small_pac_kwargs, n_perm=8, random_seed=42)
    kwargs2 = dict(small_pac_kwargs, n_perm=8, random_seed=123)
    pac1 = PAC(**kwargs1)
    pac2 = PAC(**kwargs2)
    # Act
    with torch.no_grad():
        pac_a = pac1(random_signal_3d)["pac"]
        pac_b = pac2(random_signal_3d)["pac"]
    # Assert
    assert torch.allclose(pac_a, pac_b, atol=1e-6)


def test_pac_without_permutations_returns_none_z(small_pac, random_signal_3d):
    # Arrange
    pac = small_pac  # n_perm=None
    signal = random_signal_3d
    # Act
    with torch.no_grad():
        results = pac(signal)
    # Assert
    assert results["pac_z"] is None


# ---------------------------------------------------------------------------
# Batched / single-channel parity
# ---------------------------------------------------------------------------


def test_pac_batched_matches_single_channel(small_pac_kwargs):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    pac = PAC(**kwargs)
    generator = torch.Generator().manual_seed(7)
    signal = torch.randn(1, 4, 512, generator=generator)
    # Act
    with torch.no_grad():
        batched = pac(signal)["pac"]
        single = pac(signal[:, :1])["pac"]
    # Assert
    assert torch.allclose(batched[:, :1], single, atol=1e-5)


def test_pac_custom_phase_bands_take_precedence(small_pac_kwargs):
    # Arrange
    kwargs = dict(small_pac_kwargs)
    kwargs["pha_bands_hz"] = [[4.0, 8.0], [8.0, 12.0]]
    kwargs["amp_bands_hz"] = [[60.0, 80.0], [80.0, 100.0]]
    kwargs["pha_n_bands"] = None
    kwargs["amp_n_bands"] = None
    pac = PAC(**kwargs)
    # Act
    n_phase_bands = int(pac.pha_bands_hz.shape[0])
    # Assert
    assert n_phase_bands == 2


# EOF
