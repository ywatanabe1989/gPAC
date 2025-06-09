#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:26:24 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/core/test__ModulationIndex.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/core/test__ModulationIndex.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pytest
import torch
from gpac.core._ModulationIndex import ModulationIndex


def test_modulation_index_basic():
    """Test basic ModulationIndex functionality."""
    mi_calc = ModulationIndex(n_bins=18, temperature=0.01)

    batch_size = 2
    channels = 3
    freqs_phase = 4
    freqs_amplitude = 5
    segments = 2
    time_points = 1000

    phase = torch.randn(
        batch_size, channels, freqs_phase, segments, time_points
    )
    amplitude = torch.abs(
        torch.randn(
            batch_size, channels, freqs_amplitude, segments, time_points
        )
    )

    result = mi_calc(phase, amplitude)

    assert "mi" in result
    assert result["mi"].shape == (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
    )
    assert torch.all(result["mi"] >= 0)
    assert torch.all(result["mi"] <= 1)


def test_modulation_index_shape():
    """Test ModulationIndex output shapes."""
    mi_calc = ModulationIndex(n_bins=18)

    batch_size = 2
    channels = 3
    freqs_phase = 4
    freqs_amplitude = 5
    segments = 6
    time_points = 1000

    phase = torch.randn(
        batch_size, channels, freqs_phase, segments, time_points
    )
    amplitude = torch.abs(
        torch.randn(
            batch_size, channels, freqs_amplitude, segments, time_points
        )
    )

    result = mi_calc(phase, amplitude, compute_distributions=True)

    expected_mi_shape = (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
    )
    expected_dist_shape = (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
        18,
    )

    assert result["mi"].shape == expected_mi_shape
    assert result["amplitude_distributions"].shape == expected_dist_shape
    assert result["phase_bin_centers"].shape == (18,)
    assert result["phase_bin_edges"].shape == (19,)


def test_modulation_index_with_distributions():
    """Test MI computation with amplitude distributions."""
    mi_calc = ModulationIndex(n_bins=18)

    batch_size = 1
    channels = 1
    freqs_phase = 2
    freqs_amplitude = 2
    segments = 1
    time_points = 500

    phase = torch.randn(
        batch_size, channels, freqs_phase, segments, time_points
    )
    amplitude = torch.abs(
        torch.randn(
            batch_size, channels, freqs_amplitude, segments, time_points
        )
    )

    result = mi_calc(phase, amplitude, compute_distributions=True)

    assert "amplitude_distributions" in result
    assert result["amplitude_distributions"] is not None
    assert result["amplitude_distributions"].shape == (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
        18,
    )


def test_surrogate_computation():
    """Test surrogate statistics computation."""
    mi_calc = ModulationIndex(n_bins=18)

    batch_size = 1
    channels = 1
    freqs_phase = 2
    freqs_amplitude = 2
    segments = 1
    time_points = 500
    n_perm = 10

    phase = torch.randn(
        batch_size, channels, freqs_phase, segments, time_points
    )
    amplitude = torch.abs(
        torch.randn(
            batch_size, channels, freqs_amplitude, segments, time_points
        )
    )

    # Compute original MI
    original_result = mi_calc(phase, amplitude)
    pac_values = original_result["mi"]

    # Compute surrogates with return_surrogates=True
    surrogate_result = mi_calc.compute_surrogates(
        phase,
        amplitude,
        n_perm=n_perm,
        pac_values=pac_values,
        return_surrogates=True,
    )

    assert "surrogates" in surrogate_result
    assert "surrogate_mean" in surrogate_result
    assert "surrogate_std" in surrogate_result
    assert "pac_z" in surrogate_result

    assert surrogate_result["surrogates"].shape == (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
        n_perm,
    )
    assert surrogate_result["surrogate_mean"].shape == (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
    )
    assert surrogate_result["pac_z"].shape == (
        batch_size,
        channels,
        segments,
        freqs_phase,
        freqs_amplitude,
    )


def test_phase_binning():
    """Test phase binning functionality."""
    mi_calc = ModulationIndex(n_bins=18, temperature=0.01)

    # Test with uniform phase distribution
    phase_uniform = torch.linspace(-np.pi, np.pi, 1000)
    weights = mi_calc._phase_binning(phase_uniform)

    assert weights.shape == (1000, 18)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1000), atol=1e-5)


def test_device_compatibility():
    """Test GPU/CPU compatibility."""
    mi_calc = ModulationIndex(n_bins=18)

    phase = torch.randn(1, 1, 2, 1, 500)
    amplitude = torch.abs(torch.randn(1, 1, 2, 1, 500))

    # CPU computation
    result_cpu = mi_calc(phase, amplitude)

    if torch.cuda.is_available():
        # GPU computation
        mi_calc_gpu = mi_calc.cuda()
        phase_gpu = phase.cuda()
        amplitude_gpu = amplitude.cuda()

        result_gpu = mi_calc_gpu(phase_gpu, amplitude_gpu)

        assert result_gpu["mi"].device.type == "cuda"
        # Results should be similar (allowing for numerical differences)
        assert torch.allclose(
            result_cpu["mi"], result_gpu["mi"].cpu(), atol=1e-4
        )


def test_edge_cases():
    """Test edge cases and error conditions."""
    mi_calc = ModulationIndex(n_bins=18)

    # Test shape mismatch
    phase = torch.randn(1, 1, 2, 1, 500)
    amplitude = torch.abs(torch.randn(1, 2, 2, 1, 500))  # Different channels

    with pytest.raises(ValueError):
        mi_calc(phase, amplitude)

    # Test with very small amplitude
    phase = torch.randn(1, 1, 1, 1, 100)
    amplitude = torch.ones(1, 1, 1, 1, 100) * 1e-10

    result = mi_calc(phase, amplitude)
    assert torch.all(torch.isfinite(result["mi"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
