#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/gpac/core/test__BandPassFilter_scitex.py
"""scitex-grade tests for ``gpac.core._BandPassFilter.BandPassFilter``.

Covers static-mode filterbank construction for both phase and amplitude
bands. See ``tests/gpac/test__PAC_scitex.py`` for the rule reference.
"""

import pytest
import torch

from gpac.core._BandPassFilter import BandPassFilter


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_filter():
    """BandPassFilter with default phase / amplitude ranges."""
    return BandPassFilter(
        fs=512.0,
        pha_range_hz=(4, 12),
        amp_range_hz=(60, 100),
        pha_n_bands=3,
        amp_n_bands=4,
        trainable=False,
    )


@pytest.fixture
def manual_band_filter():
    """BandPassFilter with explicit phase / amplitude bands."""
    return BandPassFilter(
        fs=512.0,
        pha_bands_hz=[[4.0, 8.0], [8.0, 12.0]],
        amp_bands_hz=[[60.0, 80.0], [80.0, 100.0], [100.0, 120.0]],
        trainable=False,
    )


# ---------------------------------------------------------------------------
# Phase filterbank construction
# ---------------------------------------------------------------------------


def test_bandpass_phase_band_count_matches_pha_n_bands(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.pha_bands_hz
    # Assert
    assert bands.shape[0] == 3


def test_bandpass_phase_band_lows_below_highs(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.pha_bands_hz
    # Assert
    assert bool((bands[:, 0] < bands[:, 1]).all().item())


def test_bandpass_phase_band_range_respects_pha_range_hz(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.pha_bands_hz
    centers = bands.mean(dim=1)
    # Assert
    assert bool(((centers >= 4.0) & (centers <= 12.0)).all().item())


# ---------------------------------------------------------------------------
# Amplitude filterbank construction
# ---------------------------------------------------------------------------


def test_bandpass_amplitude_band_count_matches_amp_n_bands(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.amp_bands_hz
    # Assert
    assert bands.shape[0] == 4


def test_bandpass_amplitude_band_lows_below_highs(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.amp_bands_hz
    # Assert
    assert bool((bands[:, 0] < bands[:, 1]).all().item())


def test_bandpass_amplitude_band_range_respects_amp_range_hz(default_filter):
    # Arrange
    bandpass = default_filter
    # Act
    bands = bandpass.amp_bands_hz
    centers = bands.mean(dim=1)
    # Assert
    assert bool(((centers >= 60.0) & (centers <= 100.0)).all().item())


# ---------------------------------------------------------------------------
# Manual-band override
# ---------------------------------------------------------------------------


def test_bandpass_manual_phase_bands_override_count(manual_band_filter):
    # Arrange
    bandpass = manual_band_filter
    # Act
    bands = bandpass.pha_bands_hz
    # Assert
    assert bands.shape[0] == 2


def test_bandpass_manual_amplitude_bands_override_count(manual_band_filter):
    # Arrange
    bandpass = manual_band_filter
    # Act
    bands = bandpass.amp_bands_hz
    # Assert
    assert bands.shape[0] == 3


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_bandpass_rejects_non_positive_sampling_rate():
    # Arrange
    bad_fs = -1.0
    # Act
    ctx = pytest.raises(ValueError, match="Sampling frequency must be positive")
    # Assert
    with ctx:
        BandPassFilter(fs=bad_fs)


def test_bandpass_rejects_phase_range_above_nyquist():
    # Arrange
    fs = 100.0  # Nyquist = 50 Hz
    # Act
    ctx = pytest.raises(ValueError, match="exceeds Nyquist frequency")
    # Assert
    with ctx:
        BandPassFilter(fs=fs, pha_range_hz=(4, 80), pha_n_bands=3)


def test_bandpass_rejects_amplitude_range_above_nyquist():
    # Arrange
    fs = 200.0  # Nyquist = 100 Hz
    # Act
    ctx = pytest.raises(ValueError, match="exceeds Nyquist frequency")
    # Assert
    with ctx:
        BandPassFilter(
            fs=fs,
            pha_range_hz=(4, 12),
            amp_range_hz=(60, 200),
            amp_n_bands=3,
        )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_bandpass_forward_returns_stacked_band_axis(default_filter):
    # Arrange
    bandpass = default_filter
    signal = torch.randn(2, 512)
    # Act
    with torch.no_grad():
        out = bandpass(signal)
    # Assert
    assert out.shape[-2] == 3 + 4


def test_bandpass_forward_preserves_time_axis(default_filter):
    # Arrange
    bandpass = default_filter
    signal = torch.randn(2, 512)
    # Act
    with torch.no_grad():
        out = bandpass(signal)
    # Assert
    assert out.shape[-1] == 512


# EOF
