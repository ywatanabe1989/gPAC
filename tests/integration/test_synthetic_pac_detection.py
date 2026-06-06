#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_synthetic_pac_detection.py
"""End-to-end integration: gPAC detects coupling at the right band pair.

A synthetic signal is built where the amplitude of an 80 Hz carrier is
modulated by the phase of a 6 Hz oscillation. After running ``gpac.PAC``
on it, the peak MI in the (phase, amplitude) grid must fall on the
band that contains 6 Hz / 80 Hz respectively.
"""

import pytest
import torch

from gpac import PAC


pytestmark = pytest.mark.integration


def _band_containing(bands_hz: torch.Tensor, target_hz: float) -> int:
    """Return the row index of the band whose centre is closest to target_hz."""
    centres = bands_hz.mean(dim=1)
    diffs = (centres - target_hz).abs()
    return int(diffs.argmin().item())


def test_synthetic_pac_peak_falls_on_phase_band_containing_6hz(
    pac_for_synthetic, coupled_synthetic_signal
):
    # Arrange
    pac = pac_for_synthetic
    signal = coupled_synthetic_signal
    expected_phase_idx = _band_containing(pac.pha_bands_hz, 6.0)
    # Act
    with torch.no_grad():
        pac_values = pac(signal)["pac"]
        argmax = int(pac_values.view(-1).argmax().item())
        n_amp = int(pac.amp_bands_hz.shape[0])
        observed_phase_idx = argmax // n_amp
    # Assert
    assert observed_phase_idx == expected_phase_idx


def test_synthetic_pac_peak_falls_on_amplitude_band_containing_80hz(
    pac_for_synthetic, coupled_synthetic_signal
):
    # Arrange
    pac = pac_for_synthetic
    signal = coupled_synthetic_signal
    expected_amp_idx = _band_containing(pac.amp_bands_hz, 80.0)
    # Act
    with torch.no_grad():
        pac_values = pac(signal)["pac"]
        argmax = int(pac_values.view(-1).argmax().item())
        n_amp = int(pac.amp_bands_hz.shape[0])
        observed_amp_idx = argmax % n_amp
    # Assert
    assert observed_amp_idx == expected_amp_idx


def test_synthetic_pac_peak_value_exceeds_noise_floor_mean(
    pac_for_synthetic, coupled_synthetic_signal
):
    # Arrange
    pac = pac_for_synthetic
    signal = coupled_synthetic_signal
    # Act
    with torch.no_grad():
        pac_values = pac(signal)["pac"].squeeze()
        peak = float(pac_values.max().item())
        mean = float(pac_values.mean().item())
    # Assert
    assert peak > 5.0 * mean


# EOF
