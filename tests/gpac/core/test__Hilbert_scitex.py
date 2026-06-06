#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/gpac/core/test__Hilbert_scitex.py
"""scitex-grade tests for ``gpac.core._Hilbert.Hilbert``.

See ``tests/gpac/test__PAC_scitex.py`` for the rule reference.
"""

import numpy as np
import pytest
import torch
from scipy.signal import hilbert as scipy_hilbert

from gpac.core._Hilbert import Hilbert


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hilbert_512():
    """Hilbert wrapper sized for a 512-sample signal."""
    return Hilbert(seq_len=512, dim=-1, fp16=False)


@pytest.fixture
def sine_wave_10hz():
    """10 Hz sine wave, 1 s @ 512 Hz."""
    t_vals = torch.linspace(0, 1, 512)
    return torch.sin(2 * np.pi * 10 * t_vals)


@pytest.fixture
def multi_freq_signal():
    """Sum of three sinusoids."""
    t_vals = torch.linspace(0, 1, 512)
    return (
        torch.sin(2 * np.pi * 5 * t_vals)
        + 0.5 * torch.sin(2 * np.pi * 15 * t_vals)
        + 0.3 * torch.sin(2 * np.pi * 30 * t_vals)
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_hilbert_forward_appends_phase_amplitude_axis(hilbert_512, sine_wave_10hz):
    # Arrange
    hilbert = hilbert_512
    signal = sine_wave_10hz
    # Act
    result = hilbert(signal)
    # Assert
    assert result.shape == (512, 2)


def test_hilbert_forward_preserves_batch_dims(hilbert_512):
    # Arrange
    signal = torch.randn(4, 3, 512)
    # Act
    result = hilbert_512(signal)
    # Assert
    assert result.shape == (4, 3, 512, 2)


# ---------------------------------------------------------------------------
# Scipy-compatibility
# ---------------------------------------------------------------------------


def test_hilbert_real_part_matches_scipy_analytic(hilbert_512, sine_wave_10hz):
    # Arrange
    hilbert = hilbert_512
    signal = sine_wave_10hz
    scipy_complex = scipy_hilbert(signal.numpy())
    # Act
    torch_complex = hilbert.get_analytic_signal(signal).numpy()
    # Assert
    assert np.allclose(torch_complex.real, scipy_complex.real, atol=1e-5)


def test_hilbert_imag_part_correlates_with_scipy(hilbert_512, multi_freq_signal):
    # Arrange
    hilbert = hilbert_512
    signal = multi_freq_signal
    scipy_complex = scipy_hilbert(signal.numpy())
    # Act
    torch_complex = hilbert.get_analytic_signal(signal).numpy()
    correlation = float(np.corrcoef(torch_complex.imag, scipy_complex.imag)[0, 1])
    # Assert
    assert correlation > 0.999


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


def test_hilbert_amplitude_is_non_negative(hilbert_512, multi_freq_signal):
    # Arrange
    hilbert = hilbert_512
    signal = multi_freq_signal
    # Act
    amplitude = hilbert(signal)[..., 1]
    # Assert
    assert (amplitude >= 0).all().item()


def test_hilbert_phase_lies_in_pi_interval(hilbert_512, multi_freq_signal):
    # Arrange
    hilbert = hilbert_512
    signal = multi_freq_signal
    # Act
    phase = hilbert(signal)[..., 0]
    # Assert
    assert bool(((phase >= -np.pi) & (phase <= np.pi)).all().item())


def test_hilbert_constant_signal_has_zero_phase(hilbert_512):
    # Arrange
    hilbert = hilbert_512
    constant = torch.ones(512)
    # Act
    phase = hilbert(constant)[..., 0]
    # Assert
    assert torch.allclose(phase, torch.zeros_like(phase), atol=1e-3)


def test_hilbert_constant_signal_has_unit_amplitude(hilbert_512):
    # Arrange
    hilbert = hilbert_512
    constant = torch.ones(512)
    # Act
    amplitude = hilbert(constant)[..., 1]
    # Assert
    assert torch.allclose(amplitude, torch.ones_like(amplitude), atol=1e-3)


# ---------------------------------------------------------------------------
# Forward / extract parity
# ---------------------------------------------------------------------------


def test_hilbert_forward_and_extract_phase_agree(hilbert_512, sine_wave_10hz):
    # Arrange
    hilbert = hilbert_512
    signal = sine_wave_10hz
    # Act
    phase_forward = hilbert(signal)[..., 0]
    phase_extract, _ = hilbert.extract_phase_amplitude(signal)
    # Assert
    assert torch.allclose(phase_forward, phase_extract, atol=1e-6)


def test_hilbert_forward_and_extract_amplitude_agree(hilbert_512, sine_wave_10hz):
    # Arrange
    hilbert = hilbert_512
    signal = sine_wave_10hz
    # Act
    amp_forward = hilbert(signal)[..., 1]
    _, amp_extract = hilbert.extract_phase_amplitude(signal)
    # Assert
    assert torch.allclose(amp_forward, amp_extract, atol=1e-6)


def test_hilbert_rejects_complex_input(hilbert_512):
    # Arrange
    hilbert = hilbert_512
    bad = torch.complex(torch.zeros(512), torch.ones(512))
    # Act
    ctx = pytest.raises(ValueError, match="Input must be real-valued")
    # Assert
    with ctx:
        hilbert.get_analytic_signal(bad)


# EOF
