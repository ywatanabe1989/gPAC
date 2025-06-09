#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 00:00:36 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/core/test__Hilbert.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/core/test__Hilbert.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pytest
import torch
from gpac.core._Hilbert import Hilbert
from scipy.signal import hilbert as scipy_hilbert


class TestHilbert:

    @pytest.fixture
    def hilbert_transform(self):
        return Hilbert(seq_len=1000, dim=-1, fp16=False)

    @pytest.fixture
    def test_signals(self):
        """Generate test signals for validation."""
        t_vals = torch.linspace(0, 2, 1000)

        # Simple sine wave
        sine_wave = torch.sin(2 * np.pi * 10 * t_vals)

        # Complex signal with multiple frequencies
        multi_freq = (
            torch.sin(2 * np.pi * 5 * t_vals)
            + 0.5 * torch.sin(2 * np.pi * 15 * t_vals)
            + 0.3 * torch.sin(2 * np.pi * 30 * t_vals)
        )

        # Noisy signal
        noisy_signal = sine_wave + 0.1 * torch.randn_like(sine_wave)

        return {
            "sine": sine_wave,
            "multi_freq": multi_freq,
            "noisy": noisy_signal,
        }

    def test_scipy_compatibility(self, hilbert_transform, test_signals):
        """Test numerical compatibility with scipy.signal.hilbert."""
        for signal_name, signal in test_signals.items():
            # PyTorch implementation
            torch_result = hilbert_transform.get_analytic_signal(signal)

            # Scipy implementation
            scipy_result = scipy_hilbert(signal.numpy())
            scipy_complex = torch.from_numpy(scipy_result).to(torch.complex64)

            # Compare real and imaginary parts
            real_corr = torch.corrcoef(
                torch.stack(
                    [torch_result.real.flatten(), scipy_complex.real.flatten()]
                )
            )[0, 1]

            imag_corr = torch.corrcoef(
                torch.stack(
                    [torch_result.imag.flatten(), scipy_complex.imag.flatten()]
                )
            )[0, 1]

            # Should have near-perfect correlation
            assert (
                real_corr > 0.999
            ), f"Real part correlation {real_corr} too low for {signal_name}"
            assert (
                imag_corr > 0.999
            ), f"Imag part correlation {imag_corr} too low for {signal_name}"

    def test_phase_amplitude_extraction(self, hilbert_transform, test_signals):
        """Test phase and amplitude extraction."""
        for signal_name, signal in test_signals.items():
            # Forward method (returns stacked [phase, amplitude])
            forward_result = hilbert_transform(signal)
            phase_forward = forward_result[..., 0]
            amplitude_forward = forward_result[..., 1]

            # Extract method (returns tuple)
            phase_extract, amplitude_extract = (
                hilbert_transform.extract_phase_amplitude(signal)
            )

            # Should be identical
            assert torch.allclose(phase_forward, phase_extract, atol=1e-6)
            assert torch.allclose(
                amplitude_forward, amplitude_extract, atol=1e-6
            )

    def test_batch_processing(self, hilbert_transform):
        """Test batch and multi-dimensional processing."""
        # Create batch of signals
        batch_size, n_channels, seq_len = 4, 3, 500
        x_batch = torch.randn(batch_size, n_channels, seq_len)

        # Process batch
        result_batch = hilbert_transform(x_batch)

        # Check output shape - should preserve all dims except add last dim for [phase, amp]
        expected_shape = (batch_size, n_channels, seq_len, 2)
        assert result_batch.shape == expected_shape

        # Process individual signals and compare
        for batch_idx in range(batch_size):
            for chan_idx in range(n_channels):
                single_signal = x_batch[batch_idx, chan_idx, :]
                single_result = hilbert_transform(single_signal)

                batch_result = result_batch[batch_idx, chan_idx, :, :]

                assert torch.allclose(single_result, batch_result, atol=1e-5)

    @pytest.mark.skip(
        reason="Torch.compile has known issues with complex gradients"
    )
    def test_gradient_flow(self, hilbert_transform):
        """Test that gradients flow properly through the transform."""
        # Create signal that requires grad
        signal = torch.randn(100, requires_grad=True)

        # Forward pass
        result = hilbert_transform(signal)
        phase, amplitude = result[..., 0], result[..., 1]

        # Compute loss (arbitrary objective)
        loss = amplitude.mean() + phase.std()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert signal.grad is not None
        assert not torch.isnan(signal.grad).any()
        assert not torch.isinf(signal.grad).any()

    def test_device_compatibility(self, hilbert_transform):
        """Test GPU/CPU device compatibility."""
        signal_cpu = torch.randn(100)

        # CPU processing
        result_cpu = hilbert_transform(signal_cpu)

        if torch.cuda.is_available():
            # Move to GPU
            hilbert_gpu = hilbert_transform.cuda()
            signal_gpu = signal_cpu.cuda()

            # GPU processing
            result_gpu = hilbert_gpu(signal_gpu)

            # Results should be nearly identical
            assert torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-5)

    def test_fp16_mode(self):
        """Test half precision mode."""
        hilbert_fp16 = Hilbert(seq_len=100, fp16=True)
        signal = torch.randn(100)

        # Should work without errors
        result = hilbert_fp16(signal)

        # Output should be reasonable
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    @pytest.mark.skip(
        reason="Torch.compile stride assertion issues with different dims"
    )
    def test_different_dimensions(self):
        """Test transform along different dimensions."""
        signal_3d = torch.randn(10, 20, 100)

        # Transform along last dimension
        hilbert_last = Hilbert(seq_len=100, dim=-1)
        result_last = hilbert_last(signal_3d)

        # Shapes should match input plus added dimension for [phase, amp]
        assert result_last.shape == (10, 20, 100, 2)

    def test_edge_cases(self, hilbert_transform):
        """Test edge cases and boundary conditions."""
        # Very short signal
        short_signal = torch.randn(4)
        result_short = hilbert_transform(short_signal)
        assert result_short.shape == (
            4,
            2,
        )  # Preserve signal length, add [phase, amp]

        # Constant signal
        constant_signal = torch.ones(100)
        result_constant = hilbert_transform(constant_signal)

        # Phase should be near zero, amplitude should be near 1
        phase_const = result_constant[..., 0]
        amp_const = result_constant[..., 1]

        assert torch.allclose(
            phase_const, torch.zeros_like(phase_const), atol=1e-3
        )
        assert torch.allclose(amp_const, torch.ones_like(amp_const), atol=1e-3)

    def test_analytic_signal_properties(self, hilbert_transform, test_signals):
        """Test mathematical properties of analytic signal."""
        for signal_name, signal in test_signals.items():
            analytic = hilbert_transform.get_analytic_signal(signal)

            # Real part should equal original signal
            assert torch.allclose(analytic.real, signal, atol=1e-5)

            # Instantaneous amplitude should be positive
            amplitude = torch.abs(analytic)
            assert (amplitude >= 0).all()

            # Phase should be in [-π, π] range
            phase = torch.angle(analytic)
            assert (phase >= -np.pi).all()
            assert (phase <= np.pi).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
