#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/integration/test_synthetic_pac_detection.py
"""End-to-end integration: gPAC detects coupling near the right band pair.

A synthetic signal is built where the amplitude of an 80 Hz carrier is
modulated by the phase of a 6 Hz oscillation. After running ``gpac.PAC``
on it, the peak MI in the (phase, amplitude) grid must fall *close* to
the band containing 6 Hz / 80 Hz respectively. We allow a tolerance of
``BAND_TOLERANCE`` neighbour bands either side because gpac's bandpass
side-lobes can pull the argmax one or two bands away from the analytic
target, especially when the wide amplitude carrier (80 Hz) sits next to
the upper edge of the amp grid.
"""

import pytest
import torch

from gpac import PAC


pytestmark = pytest.mark.integration

# Tolerance: how many bands either side of the analytic target band we accept
# as "near". gpac default bands are linspace(f_min, f_max, n_bands) so two
# neighbours either side is roughly one harmonic step.
BAND_TOLERANCE = 2


def _band_containing(bands_hz: torch.Tensor, target_hz: float) -> int:
    """Return the row index of the band whose centre is closest to target_hz."""
    centres = bands_hz.mean(dim=1)
    diffs = (centres - target_hz).abs()
    return int(diffs.argmin().item())


def _decode_peak_indices(pac_values: torch.Tensor, n_amp: int) -> tuple[int, int]:
    """Return (phase_idx, amp_idx) of the global peak in the (n_pha, n_amp) grid.

    ``pac_values`` is shaped ``(batch, channels, segments, n_pha, n_amp)``. We
    flatten the trailing pair, take argmax, and decode against the known
    ``n_amp`` width.
    """
    argmax = int(pac_values.view(-1).argmax().item())
    return argmax // n_amp, argmax % n_amp


def test_synthetic_pac_peak_falls_near_phase_band_containing_6hz(
    pac_for_synthetic, coupled_synthetic_signal
):
    # Arrange
    pac = pac_for_synthetic
    signal = coupled_synthetic_signal
    expected_phase_idx = _band_containing(pac.pha_bands_hz, 6.0)
    n_amp = int(pac.amp_bands_hz.shape[0])
    # Act
    with torch.no_grad():
        observed_phase_idx, _ = _decode_peak_indices(pac(signal)["pac"], n_amp)
    # Assert
    assert abs(observed_phase_idx - expected_phase_idx) <= BAND_TOLERANCE


def test_synthetic_pac_peak_falls_near_amplitude_band_containing_80hz(
    pac_for_synthetic, coupled_synthetic_signal
):
    # Arrange
    pac = pac_for_synthetic
    signal = coupled_synthetic_signal
    expected_amp_idx = _band_containing(pac.amp_bands_hz, 80.0)
    n_amp = int(pac.amp_bands_hz.shape[0])
    # Act
    with torch.no_grad():
        _, observed_amp_idx = _decode_peak_indices(pac(signal)["pac"], n_amp)
    # Assert
    assert abs(observed_amp_idx - expected_amp_idx) <= BAND_TOLERANCE


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
