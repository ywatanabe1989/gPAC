#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 15:54:52 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/core/_BandPassFilters/test__PooledBandPassFilter.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/core/_BandPassFilters/test__PooledBandPassFilter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
import torch
from gpac.core._BandPassFilters._PooledBandPassFilter import \
    PooledBandPassFilter


def test_learnable_selection():
    """Test learnable filter selection."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=2,
        pha_n_pool_ratio=3.0,
        amp_n_pool_ratio=3.0,
    )
    xx_input = torch.randn(2, 1000)
    output = filt(xx_input)

    assert output.shape == (2, 5, 1000)
    assert filt.pha_n_bands == 3
    assert filt.amp_n_bands == 2
    assert filt.pha_n_pool == 9
    assert filt.amp_n_pool == 6


def test_manual_bands_pooled():
    """Test manual bands with pooled filter."""
    pha_bands = [[4, 8], [8, 12], [12, 20]]
    amp_bands = [[60, 80], [80, 120]]

    filt = PooledBandPassFilter(
        fs=500,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        pha_n_bands=2,
        amp_n_bands=1,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
    )

    xx_input = torch.randn(2, 1000)
    output = filt(xx_input)

    actual_pha_n_bands = len(pha_bands)
    actual_amp_n_bands = len(amp_bands)
    expected_n_bands = actual_pha_n_bands + actual_amp_n_bands

    assert output.shape == (
        2,
        expected_n_bands,
        1000,
    )
    assert filt.pha_n_pool == 6
    assert filt.amp_n_pool == 4


def test_gumbel_softmax():
    """Test Gumbel softmax selection mechanism."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.5,
        amp_n_pool_ratio=2.5,
        temperature=0.5,
    )

    xx_input = torch.randn(2, 1000)

    filt.train()
    output_train = filt(xx_input)

    filt.eval()
    output_eval = filt(xx_input)

    assert output_train.shape == output_eval.shape == (2, 4, 1000)


def test_selected_frequencies():
    """Test frequency selection tracking."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.5,
        amp_n_pool_ratio=2.5,
    )

    pha_freqs, amp_freqs = filt.get_selected_frequencies()

    assert len(pha_freqs) == 2
    assert len(amp_freqs) == 2
    assert all(4 <= freq_val <= 30 for freq_val in pha_freqs)
    assert all(60 <= freq_val <= 150 for freq_val in amp_freqs)


def test_hard_selection():
    """Test hard selection mode."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.5,
        amp_n_pool_ratio=2.5,
        hard_selection=True,
    )

    xx_input = torch.randn(2, 1000)
    output = filt(xx_input)

    assert output.shape == (2, 4, 1000)


def test_info_property():
    """Test info property returns filter details."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=3,
        amp_n_bands=2,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
    )

    info = filt.info

    assert "pha_weights" in info
    assert "amp_weights" in info
    assert "pha_center_freqs" in info
    assert "amp_center_freqs" in info
    assert info["pha_weights"].shape[0] == filt.pha_n_pool
    assert info["amp_weights"].shape[0] == filt.amp_n_pool


def test_gradient_flow():
    """Test gradient flow through learnable weights."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
    )
    filt.train()

    xx_input = torch.randn(2, 1000)
    output = filt(xx_input)
    loss = output.sum()
    loss.backward()

    # Check that learnable parameters have gradients
    assert filt.pha_logits.grad is not None
    assert filt.amp_logits.grad is not None


def test_gradient_flow():
    """Test gradient flow through learnable weights."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        hard_selection=False,  # Ensure soft selection
    )
    filt.train()

    xx_input = torch.randn(2, 1000, requires_grad=True)
    output = filt(xx_input)
    loss = output.mean()
    loss.backward()

    assert filt.pha_logits.grad is not None
    assert filt.amp_logits.grad is not None


def test_trainability():
    """Test that weights actually change during training."""
    filt = PooledBandPassFilter(
        fs=500,
        pha_range_hz=(4, 30),
        amp_range_hz=(60, 150),
        pha_n_bands=2,
        amp_n_bands=2,
        pha_n_pool_ratio=2.0,
        amp_n_pool_ratio=2.0,
        hard_selection=False,
    )
    filt.train()

    initial_pha_logits = filt.pha_logits.data.clone()
    initial_amp_logits = filt.amp_logits.data.clone()

    optimizer = torch.optim.Adam(filt.parameters(), lr=0.1)
    xx_input = torch.randn(4, 1000, requires_grad=True)
    target = torch.randn(4, 4, 1000)

    for epoch_idx in range(10):
        optimizer.zero_grad()
        output = filt(xx_input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    pha_changed = not torch.allclose(
        initial_pha_logits, filt.pha_logits.data, atol=1e-4
    )
    amp_changed = not torch.allclose(
        initial_amp_logits, filt.amp_logits.data, atol=1e-4
    )

    assert pha_changed, "Phase logits should change during training"
    assert amp_changed, "Amplitude logits should change during training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
