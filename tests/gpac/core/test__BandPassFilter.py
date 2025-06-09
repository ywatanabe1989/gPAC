#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:56:05 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/core/test__BandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/core/test__BandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

import pytest
import torch
from gpac.core._BandPassFilter import BandPassFilter


def test_static_mode():
    """Test static bandpass filter mode."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=2,
        trainable=False,
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 5, 1000)
    assert not filt.trainable
    assert filt.get_regularization_loss() == 0.0


def test_trainable_mode():
    """Test trainable bandpass filter mode."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        trainable=True,
        pha_n_pool_ratio=4.0,
        amp_n_pool_ratio=4.0,
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 4, 1000)
    assert filt.trainable
    # Check that regularization loss is available for trainable mode
    reg_loss = filt.get_regularization_loss()
    assert isinstance(reg_loss, torch.Tensor)


def test_manual_bands_static():
    """Test manual band specification in static mode."""
    pha_bands = [[4, 8], [8, 12]]
    amp_bands = [[60, 80]]
    filt = BandPassFilter(
        fs=500,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        trainable=False,
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 3, 1000)
    assert filt.pha_n_bands == 2
    assert filt.amp_n_bands == 1


def test_manual_bands_trainable():
    """Test manual band specification in trainable mode."""
    pha_bands = [[4, 8], [8, 12], [12, 20]]
    amp_bands = [[60, 80], [80, 120]]
    filt = BandPassFilter(
        fs=500,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        trainable=True,
        pha_n_bands=2,
        amp_n_bands=1,
        pha_n_pool_ratio=1.5,
        amp_n_pool_ratio=2.0,
    )
    x = torch.randn(2, 1000)
    output = filt(x)

    actual_pha_n_bands = len(pha_bands)
    actual_amp_n_bands = len(amp_bands)
    expected_n_bands = actual_pha_n_bands + actual_amp_n_bands

    assert output.shape == (2, expected_n_bands, 1000)
    assert filt.trainable


def test_info_delegation():
    """Test info property delegation."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=3,
        trainable=False,
    )
    info = filt.info
    assert "trainable" in info
    assert info["trainable"] == False
    assert "pha_center_freqs" in info
    assert "amp_center_freqs" in info
    assert "fs" in info
    assert "n_cycles" in info


def test_property_access():
    """Test center frequency property access."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=2,
        trainable=False,
    )
    pha_mids = filt.pha_mids
    amp_mids = filt.amp_mids
    assert len(pha_mids) == 3
    assert len(amp_mids) == 2
    assert all(4 <= f <= 30 for f in pha_mids)
    assert all(60 <= f <= 150 for f in amp_mids)


def test_parameter_conflicts():
    """Test parameter conflict warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BandPassFilter(
            fs=500,
            pha_range_hz=(4, 30),
            pha_n_bands=3,
            pha_bands_hz=[[4, 8], [8, 12]],
            amp_range_hz=(60, 150),
            amp_n_bands=2,
            trainable=False,
        )
        # Should have at least one warning about pha_bands_hz
        assert len(w) >= 1
        assert any(
            "Manual pha_bands_hz takes precedence" in str(warning.message)
            for warning in w
        )


def test_nyquist_validation():
    """Test that frequencies exceeding Nyquist are rejected."""
    with pytest.raises(ValueError, match="exceeds Nyquist frequency"):
        BandPassFilter(
            fs=100,
            pha_range_hz=(4, 30),
            amp_range_hz=(60, 80),  # Exceeds Nyquist of 50
            pha_n_bands=2,
            amp_n_bands=2,
            trainable=False,
        )


def test_sampling_frequency_validation():
    """Test that invalid sampling frequencies are rejected."""
    with pytest.raises(
        ValueError, match="Sampling frequency must be positive"
    ):
        BandPassFilter(
            fs=-100,
            pha_range_hz=(4, 30),
            amp_range_hz=(60, 150),
            pha_n_bands=2,
            amp_n_bands=2,
            trainable=False,
        )


def test_spacing_modes():
    """Test both linear and logarithmic spacing modes."""
    # Test log spacing
    filt_log = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=3,
        spacing="log",
        trainable=False,
    )
    # Test linear spacing
    filt_linear = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=3,
        spacing="linear",
        trainable=False,
    )
    x = torch.randn(2, 1000)
    output_log = filt_log(x)
    output_linear = filt_linear(x)
    assert output_log.shape == output_linear.shape == (2, 6, 1000)
    assert filt_log.spacing == "log"
    assert filt_linear.spacing == "linear"


def test_fp16_mode():
    """Test FP16 precision mode."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        fp16=True,
        trainable=False,
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 4, 1000)


def test_trainable_regularization():
    """Test regularization loss for trainable filters."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=1,
        trainable=True,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
    )
    # Regularization loss should be available for trainable filters
    reg_loss = filt.get_regularization_loss()
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # scalar


def test_temperature_control():
    """Test Gumbel softmax temperature control."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        trainable=True,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        temperature=0.1,  # Low temperature for sharper selection
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 4, 1000)


def test_hard_selection():
    """Test hard selection mode for trainable filters."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        trainable=True,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        hard_selection=True,
    )
    x = torch.randn(2, 1000)
    output = filt(x)
    assert output.shape == (2, 4, 1000)


def test_batch_processing():
    """Test processing of different batch sizes."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        trainable=False,
    )
    # Test different batch dimensions
    for batch_shape in [(1000,), (5, 1000), (3, 4, 1000)]:
        x = torch.randn(*batch_shape)
        output = filt(x)
        expected_shape = batch_shape[:-1] + (4, 1000)
        assert output.shape == expected_shape


def test_device_compatibility():
    """Test device transfer compatibility."""
    filt = BandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        trainable=False,
    )
    x = torch.randn(2, 1000)
    if torch.cuda.is_available():
        filt_cuda = filt.cuda()
        x_cuda = x.cuda()
        output = filt_cuda(x_cuda)
        assert output.is_cuda
        assert output.device == x_cuda.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
