#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:29:24 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/core/_BandPassFilters/test__StaticBandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/core/_BandPassFilters/test__StaticBandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
import torch
from gpac.core._BandPassFilters._StaticBandPassFilter import \
    StaticBandPassFilter


def test_manual_bands():
    """Test manual band specification."""
    pha_bands = [[4, 8], [8, 12]]
    amp_bands = [[60, 80], [80, 120]]

    filt = StaticBandPassFilter(
        fs=500,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
    )

    x = torch.randn(2, 3, 1000)
    output = filt(x)

    assert output.shape == (2, 3, 4, 1000)
    assert filt.pha_n_bands == 2
    assert filt.amp_n_bands == 2


def test_auto_generation():
    """Test automatic band generation."""
    filt = StaticBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=5,
    )

    x = torch.randn(2, 1000)
    output = filt(x)

    assert output.shape == (2, 8, 1000)
    assert len(filt.pha_center_freqs) == 3
    assert len(filt.amp_center_freqs) == 5


def test_info_property():
    """Test info property completeness."""
    filt = StaticBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=3,
    )

    info = filt.info
    required_keys = [
        "pha_bands_hz",
        "amp_bands_hz",
        "pha_center_freqs",
        "amp_center_freqs",
        "fs",
        "n_cycles",
        "spacing",
    ]

    for key in required_keys:
        assert key in info


def test_nyquist_validation():
    """Test Nyquist frequency validation."""
    with pytest.raises(ValueError):
        StaticBandPassFilter(
            fs=100,
            pha_range_hz=(4, 60),  # Exceeds Nyquist of 50
            amp_range_hz=(60, 80),
            pha_n_bands=2,
            amp_n_bands=2,
        )


def test_fp16_mode():
    """Test FP16 precision mode."""
    filt = StaticBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        fp16=True,
    )

    x = torch.randn(2, 1000)
    output = filt(x)

    assert output.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
