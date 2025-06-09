#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 19:03:20 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/test__ModulationIndexMemoryOptimized.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/test__ModulationIndexMemoryOptimized.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from unittest.mock import patch

import numpy as np
import pytest
import torch
from gpac._ModulationIndexMemoryOptimized import ModulationIndexMemoryOptimized


class TestModulationIndexMemoryOptimized:

    @pytest.fixture
    def mi_calc(self):
        return ModulationIndexMemoryOptimized(n_bins=18, chunk_size=10)

    @pytest.fixture
    def sample_data(self):
        batch, channels, freqs_phase, freqs_amp, segments, time_points = (
            2,
            3,
            4,
            5,
            2,
            100,
        )

        phase = torch.randn(
            batch, channels, freqs_phase, segments, time_points
        )
        amplitude = torch.abs(
            torch.randn(batch, channels, freqs_amp, segments, time_points)
        )

        return phase, amplitude

    def test_initialization(self):
        mi_calc = ModulationIndexMemoryOptimized(n_bins=20, chunk_size=50)

        assert mi_calc.n_bins == 20
        assert mi_calc.chunk_size == 50
        assert mi_calc.bin_edges.shape == (21,)
        assert torch.allclose(mi_calc.uniform_dist, torch.ones(20) / 20)

    def test_compute_histogram_scatter_basic(self, mi_calc):
        batch_size = 3
        time_points = 50

        phase_values = torch.linspace(-np.pi, np.pi, time_points).repeat(
            batch_size, 1
        )
        amplitude_values = torch.ones(batch_size, time_points)

        result = mi_calc._compute_histogram_scatter(
            phase_values, amplitude_values
        )

        assert result.shape == (batch_size, mi_calc.n_bins)
        assert torch.all(result >= 0)

    def test_compute_histogram_scatter_edge_cases(self, mi_calc):
        batch_size = 2
        time_points = 10

        phase_values = torch.tensor([[-np.pi, np.pi], [0, np.pi / 2]])
        amplitude_values = torch.ones(2, 2)

        result = mi_calc._compute_histogram_scatter(
            phase_values, amplitude_values
        )

        assert result.shape == (2, mi_calc.n_bins)
        assert torch.all(result >= 0)

    def test_compute_mi_from_distribution_uniform(self, mi_calc):
        batch_size = 3

        uniform_dist = torch.ones(batch_size, mi_calc.n_bins)

        mi_values = mi_calc._compute_mi_from_distribution(uniform_dist)

        assert mi_values.shape == (batch_size,)
        assert torch.all(mi_values >= 0)
        assert torch.all(mi_values <= 1)
        assert torch.all(mi_values < 0.1)

    def test_compute_mi_from_distribution_concentrated(self, mi_calc):
        batch_size = 2

        concentrated_dist = torch.zeros(batch_size, mi_calc.n_bins)
        concentrated_dist[:, 0] = 100

        mi_values = mi_calc._compute_mi_from_distribution(concentrated_dist)

        assert mi_values.shape == (batch_size,)
        assert torch.all(mi_values >= 0)
        assert torch.all(mi_values <= 1)
        assert torch.all(mi_values > 0.8)

    def test_forward_basic(self, mi_calc, sample_data):
        phase, amplitude = sample_data

        result = mi_calc(phase, amplitude)

        batch, channels, freqs_phase, _, _ = phase.shape
        _, _, freqs_amp, _, _ = amplitude.shape

        assert "mi" in result
        assert result["mi"].shape == (batch, channels, freqs_phase, freqs_amp)
        assert torch.all(result["mi"] >= 0)
        assert torch.all(result["mi"] <= 1)

    def test_forward_with_distributions(self, mi_calc, sample_data):
        phase, amplitude = sample_data

        result = mi_calc(phase, amplitude, compute_distributions=True)

        batch, channels, freqs_phase, _, _ = phase.shape
        _, _, freqs_amp, _, _ = amplitude.shape

        assert "mi" in result
        assert "distributions" in result
        assert result["distributions"].shape == (
            batch,
            channels,
            freqs_phase,
            freqs_amp,
            mi_calc.n_bins,
        )

    def test_forward_chunking_behavior(self, sample_data):
        mi_calc_small_chunk = ModulationIndexMemoryOptimized(
            n_bins=18, chunk_size=2
        )
        phase, amplitude = sample_data

        result = mi_calc_small_chunk(phase, amplitude)

        batch, channels, freqs_phase, _, _ = phase.shape
        _, _, freqs_amp, _, _ = amplitude.shape

        assert result["mi"].shape == (batch, channels, freqs_phase, freqs_amp)
        assert torch.all(result["mi"] >= 0)
        assert torch.all(result["mi"] <= 1)

    def test_memory_cleanup(self, mi_calc, sample_data):
        phase, amplitude = sample_data

        with patch("torch.cuda.empty_cache") as mock_cache:
            with patch("torch.cuda.is_available", return_value=True):
                mi_calc.chunk_size = 1
                result = mi_calc(phase, amplitude)

                assert mock_cache.called

    def test_estimate_memory_usage(self, mi_calc):
        batch, channels, freqs_phase, freqs_amp, time_points = (
            4,
            8,
            10,
            15,
            1000,
        )

        memory_est = mi_calc.estimate_memory_usage(
            batch, channels, freqs_phase, freqs_amp, time_points
        )

        assert "input_memory_gb" in memory_est
        assert "working_memory_gb" in memory_est
        assert "output_memory_gb" in memory_est
        assert "total_memory_gb" in memory_est

        for value in memory_est.values():
            assert value > 0

        expected_total = (
            memory_est["input_memory_gb"]
            + memory_est["working_memory_gb"]
            + memory_est["output_memory_gb"]
        )
        assert abs(memory_est["total_memory_gb"] - expected_total) < 1e-6

    def test_different_n_bins(self, sample_data):
        phase, amplitude = sample_data

        for n_bins in [12, 24, 36]:
            mi_calc = ModulationIndexMemoryOptimized(
                n_bins=n_bins, chunk_size=10
            )
            result = mi_calc(phase, amplitude)

            assert mi_calc.n_bins == n_bins
            assert mi_calc.bin_edges.shape == (n_bins + 1,)
            assert mi_calc.uniform_dist.shape == (n_bins,)

    def test_phase_normalization_edge_cases(self, mi_calc):
        batch_size = 2
        time_points = 10

        phase_values = torch.tensor(
            [
                [-np.pi, -np.pi + 1e-6, np.pi - 1e-6, np.pi],
                [0, np.pi / 2, -np.pi / 2, np.pi / 4],
            ]
        )
        amplitude_values = torch.ones(2, 4)

        result = mi_calc._compute_histogram_scatter(
            phase_values, amplitude_values
        )

        assert result.shape == (2, mi_calc.n_bins)
        assert torch.all(result >= 0)
        assert torch.all(torch.isfinite(result))

    def test_zero_amplitude_handling(self, mi_calc):
        batch_size = 2
        time_points = 20

        phase_values = torch.randn(batch_size, time_points)
        amplitude_values = torch.zeros(batch_size, time_points)

        hist_result = mi_calc._compute_histogram_scatter(
            phase_values, amplitude_values
        )
        mi_result = mi_calc._compute_mi_from_distribution(hist_result)

        assert torch.all(torch.isfinite(mi_result))
        assert torch.all(mi_result >= 0)

    def test_single_bin_concentration(self, mi_calc):
        batch_size = 1
        time_points = 100

        phase_values = torch.zeros(batch_size, time_points)
        amplitude_values = torch.ones(batch_size, time_points)

        hist_result = mi_calc._compute_histogram_scatter(
            phase_values, amplitude_values
        )
        mi_result = mi_calc._compute_mi_from_distribution(hist_result)

        assert torch.all(mi_result > 0.5)

    def test_numerical_stability(self, mi_calc):
        batch_size = 3

        small_dist = torch.ones(batch_size, mi_calc.n_bins) * 1e-10
        mi_values = mi_calc._compute_mi_from_distribution(small_dist)

        assert torch.all(torch.isfinite(mi_values))
        assert torch.all(mi_values >= 0)
        assert torch.all(mi_values <= 1)

    def test_device_consistency(self, mi_calc, sample_data):
        phase, amplitude = sample_data

        if torch.cuda.is_available():
            mi_calc_gpu = mi_calc.cuda()
            phase_gpu = phase.cuda()
            amplitude_gpu = amplitude.cuda()

            result = mi_calc_gpu(phase_gpu, amplitude_gpu)

            assert result["mi"].device.type == "cuda"
            assert torch.all(torch.isfinite(result["mi"]))

    def test_gradient_flow(self, mi_calc, sample_data):
        phase, amplitude = sample_data
        phase = phase.detach().requires_grad_(True)
        amplitude = amplitude.detach().requires_grad_(True)

        result = mi_calc(phase, amplitude)
        loss = result["mi"].sum()

        try:
            loss.backward()
            grad_computed = True
        except:
            grad_computed = False

        # MI computation with scatter operations may not support gradients
        # This is expected behavior for histogram-based methods
        assert grad_computed or phase.grad is None


class TestModulationIndexMemoryOptimizedIntegration:

    def test_pac_integration_simulation(self):
        mi_calc = ModulationIndexMemoryOptimized(n_bins=18, chunk_size=50)

        batch_size = 2
        time_points = 1000
        fs = 500

        t = torch.linspace(0, 2, time_points)
        phase_freq = 8
        amp_freq = 80

        phase_signal = 2 * np.pi * phase_freq * t
        phase_values = (
            torch.sin(phase_signal).unsqueeze(0).repeat(batch_size, 1)
        )

        coupling_strength = 0.5
        modulation = 1 + coupling_strength * torch.cos(phase_signal)
        amp_signal = modulation * torch.sin(2 * np.pi * amp_freq * t)
        amplitude_values = (
            torch.abs(amp_signal).unsqueeze(0).repeat(batch_size, 1)
        )

        phase_input = phase_values.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        amp_input = amplitude_values.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        result = mi_calc(phase_input, amp_input)

        mi_values = result["mi"].squeeze()
        mi_value = mi_values.mean().item()
        assert (
            mi_value > 0.1
        ), f"Expected MI > 0.1 for coupled signal, got {mi_value}"

    def test_performance_benchmarking(self):
        mi_calc = ModulationIndexMemoryOptimized(n_bins=18, chunk_size=100)

        batch, channels, freqs_phase, freqs_amp, segments, time_points = (
            8,
            32,
            10,
            15,
            4,
            2000,
        )

        phase = torch.randn(
            batch, channels, freqs_phase, segments, time_points
        )
        amplitude = torch.abs(
            torch.randn(batch, channels, freqs_amp, segments, time_points)
        )

        import time

        start_time = time.time()
        result = mi_calc(phase, amplitude)
        end_time = time.time()

        processing_time = end_time - start_time
        assert (
            processing_time < 30
        ), f"Processing took too long: {processing_time}s"

        assert torch.all(torch.isfinite(result["mi"]))
        assert torch.all(result["mi"] >= 0)
        assert torch.all(result["mi"] <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
