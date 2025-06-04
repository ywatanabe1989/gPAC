#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 20:50:00 (ywatanabe)"
# File: ./tests/gpac/test__Hilbert.py

import torch
import pytest
import numpy as np
from gpac._Hilbert import Hilbert


class TestHilbert:
    """Test suite for Hilbert transform following AAA pattern and testing guidelines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.seq_len = 1000
        self.n_freqs = 5
        self.steepness = 50.0  # Default steepness

    # =============================================================================
    # Forward Pass Tests
    # =============================================================================

    def test_forward_pass_basic(self):
        """Test basic forward pass with default parameters."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        batch_size, n_segments = 2, 3
        x = torch.randn(batch_size, n_segments, self.n_freqs, self.seq_len)

        # Act
        output = hilbert(x)

        # Assert
        expected_shape = (batch_size, n_segments, self.n_freqs, self.seq_len, 2)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()
        # Last dimension should be [phase, amplitude]
        assert output.shape[-1] == 2

    def test_forward_pass_different_dimensions(self):
        """Test forward pass with different input dimensions."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len, dim=-1)

        # Test different valid shapes
        test_shapes = [
            (1, 1, 1, self.seq_len),  # Minimal shape
            (4, 2, 5, self.seq_len),  # Standard shape
            (10, 5, 3, self.seq_len),  # Larger shape
        ]

        for shape in test_shapes:
            x = torch.randn(*shape)
            output = hilbert(x)

            assert output.shape == shape + (2,)
            assert torch.isfinite(output).all()

    def test_forward_pass_custom_steepness(self):
        """Test forward pass with custom steepness parameter."""
        # Arrange
        steepness_values = [10.0, 50.0, 100.0, 200.0]
        x = torch.randn(2, 1, 3, self.seq_len)

        for steepness in steepness_values:
            hilbert = Hilbert(seq_len=self.seq_len, steepness=steepness)
            output = hilbert(x)

            assert output.shape == (2, 1, 3, self.seq_len, 2)
            assert torch.isfinite(output).all()

    # =============================================================================
    # Input Shape Tests
    # =============================================================================

    def test_input_shape_validation(self):
        """Test that various input shapes are handled correctly."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)

        # Test minimum dimensionality (4D required)
        valid_4d = torch.randn(1, 1, 1, self.seq_len)
        output = hilbert(valid_4d)
        assert output.shape == (1, 1, 1, self.seq_len, 2)

        # Test with more dimensions
        valid_5d = torch.randn(2, 3, 4, 5, self.seq_len)
        hilbert_5d = Hilbert(seq_len=self.seq_len, dim=-1)
        output_5d = hilbert_5d(valid_5d)
        assert output_5d.shape == (2, 3, 4, 5, self.seq_len, 2)

    def test_dimension_parameter(self):
        """Test hilbert transform along different dimensions."""
        # Test with different dimension parameters
        shapes_and_dims = [
            ((2, 3, self.seq_len, 4), -2),  # Transform along second-to-last
            ((2, self.seq_len, 3, 4), 1),  # Transform along dim 1
            ((self.seq_len, 2, 3, 4), 0),  # Transform along dim 0
        ]

        for shape, dim in shapes_and_dims:
            x = torch.randn(*shape)
            hilbert = Hilbert(seq_len=self.seq_len, dim=dim)
            output = hilbert(x)

            # Output should have extra dimension of size 2 at the end
            expected_shape = shape + (2,)
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()

    # =============================================================================
    # Output Shape and Content Tests
    # =============================================================================

    def test_output_phase_amplitude_structure(self):
        """Test that output correctly contains phase and amplitude."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        x = torch.randn(2, 1, 3, self.seq_len)

        # Act
        output = hilbert(x)
        phase = output[..., 0]
        amplitude = output[..., 1]

        # Assert
        # Phase should be in [-pi, pi]
        assert (phase >= -np.pi).all() and (phase <= np.pi).all()
        # Amplitude should be non-negative
        assert (amplitude >= 0).all()

    def test_output_consistency(self):
        """Test output consistency with same input."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        x = torch.randn(2, 1, 3, self.seq_len)

        # Act
        output1 = hilbert(x)
        output2 = hilbert(x)

        # Assert - outputs should be identical for same input
        assert torch.allclose(output1, output2)

    # =============================================================================
    # Parameter Validation Tests
    # =============================================================================

    def test_parameter_validation_seq_len(self):
        """Test sequence length parameter validation."""
        # Test invalid sequence length
        with pytest.raises(ValueError, match="seq_len must be positive"):
            Hilbert(seq_len=0)

        with pytest.raises(ValueError, match="seq_len must be positive"):
            Hilbert(seq_len=-100)

    def test_parameter_validation_steepness(self):
        """Test steepness parameter validation."""
        # Test invalid steepness
        with pytest.raises(ValueError, match="steepness must be positive"):
            Hilbert(seq_len=self.seq_len, steepness=0)

        with pytest.raises(ValueError, match="steepness must be positive"):
            Hilbert(seq_len=self.seq_len, steepness=-10.0)

    def test_fp16_mode(self):
        """Test half precision mode."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len, fp16=True)
        x = torch.randn(2, 1, 3, self.seq_len)

        # Act
        output = hilbert(x)

        # Assert
        assert output.shape == (2, 1, 3, self.seq_len, 2)
        assert torch.isfinite(output).all()
        # Note: Internal computation uses fp16 but output might be fp32

    # =============================================================================
    # Differentiability Tests
    # =============================================================================

    def test_backward_pass(self):
        """Test gradient flow through Hilbert transform."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        x = torch.randn(2, 1, 3, self.seq_len, requires_grad=True)

        # Act
        output = hilbert(x)
        loss = output.sum()
        loss.backward()

        # Assert
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_gradient_stability(self):
        """Test gradient stability with different input scales."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        scales = [1e-3, 1.0, 1e3]

        for scale in scales:
            x = torch.randn(2, 1, 3, self.seq_len) * scale
            x.requires_grad = True

            output = hilbert(x)
            loss = output.mean()
            loss.backward()

            # Check gradients are finite and reasonable
            assert torch.isfinite(x.grad).all()
            assert x.grad.abs().max() < 1e6  # No gradient explosion

    def test_differentiable_approximation(self):
        """Test that the differentiable approximation works correctly."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len, steepness=100.0)
        x = torch.randn(2, 1, 3, self.seq_len, requires_grad=True)

        # Act
        output = hilbert(x)

        # Create a loss that depends on both phase and amplitude
        phase = output[..., 0]
        amplitude = output[..., 1]
        loss = phase.mean() + amplitude.mean()
        loss.backward()

        # Assert
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Gradient should be non-zero (transform is not constant)
        assert x.grad.abs().sum() > 0

    # =============================================================================
    # Edge Cases and Special Inputs
    # =============================================================================

    def test_zero_input(self):
        """Test handling of zero input."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        x = torch.zeros(2, 1, 3, self.seq_len)

        # Act
        output = hilbert(x)

        # Assert
        assert torch.isfinite(output).all()
        phase = output[..., 0]
        amplitude = output[..., 1]

        # For zero input, amplitude should be near zero
        assert amplitude.abs().max() < 1e-6

    def test_constant_input(self):
        """Test handling of constant DC input."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        x = torch.ones(2, 1, 3, self.seq_len) * 2.0

        # Act
        output = hilbert(x)

        # Assert
        assert torch.isfinite(output).all()
        # DC component should be preserved in some form

    def test_sinusoidal_input(self):
        """Test with known sinusoidal input."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        t = torch.linspace(0, 2 * np.pi, self.seq_len)
        freq = 5.0  # 5 Hz
        x = torch.sin(2 * np.pi * freq * t / self.seq_len)
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch dims

        # Act
        output = hilbert(x)
        amplitude = output[..., 1]

        # Assert
        # For a pure sinusoid, amplitude should be relatively constant
        assert torch.isfinite(output).all()
        # Check amplitude variation is small (after removing edge effects)
        center_amp = amplitude[..., self.seq_len // 4 : -self.seq_len // 4]
        amp_std = center_amp.std()
        amp_mean = center_amp.mean()
        assert amp_std / amp_mean < 0.6  # Less than 60% variation

    # =============================================================================
    # Integration Tests
    # =============================================================================

    def test_batch_processing(self):
        """Test processing multiple batches efficiently."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)
        batch_sizes = [1, 4, 16, 32]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 2, 3, self.seq_len)
            output = hilbert(x)

            assert output.shape[0] == batch_size
            assert torch.isfinite(output).all()

    def test_frequency_content_preservation(self):
        """Test that frequency content is preserved."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)

        # Create multi-frequency signal
        t = torch.linspace(0, 1, self.seq_len)
        freqs = [5.0, 10.0, 20.0]
        x = sum(torch.sin(2 * np.pi * f * t) for f in freqs)
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Act
        output = hilbert(x)

        # Assert
        assert torch.isfinite(output).all()
        # The transform should preserve the signal structure
        amplitude = output[..., 1]
        assert amplitude.mean() > 0  # Non-zero amplitude

    def test_edge_effects(self):
        """Test edge effects handling."""
        # Arrange
        hilbert = Hilbert(seq_len=self.seq_len)

        # Create signal with sharp transitions at edges
        x = torch.zeros(1, 1, 1, self.seq_len)
        x[..., self.seq_len // 4 : 3 * self.seq_len // 4] = 1.0

        # Act
        output = hilbert(x)

        # Assert
        assert torch.isfinite(output).all()
        # Edge effects should not cause NaN or Inf

    def test_different_steepness_effects(self):
        """Test how different steepness values affect the transform."""
        # Arrange
        x = torch.randn(1, 1, 1, self.seq_len)
        steepness_values = [10.0, 50.0, 100.0, 500.0]
        outputs = []

        for steepness in steepness_values:
            hilbert = Hilbert(seq_len=self.seq_len, steepness=steepness)
            output = hilbert(x)
            outputs.append(output)

        # Assert
        # All outputs should be valid
        for output in outputs:
            assert torch.isfinite(output).all()

        # Higher steepness should approach ideal Hilbert transform
        # (This is a qualitative test - exact validation would require
        # comparison with scipy.signal.hilbert)


# Main block for standalone testing

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Hilbert.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-28 19:17:31 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_Hilbert.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_Hilbert.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import torch  # 1.7.1
# import torch.nn as nn
# from torch.fft import fft, ifft
#
#
# class Hilbert(nn.Module):
#     """
#     Differentiable Hilbert transform module for extracting instantaneous phase and amplitude.
#
#     Uses a sigmoid approximation of the Heaviside step function to ensure differentiability.
#
#     Parameters
#     ----------
#     seq_len : int
#         Expected sequence length (for pre-computing frequency grid)
#     dim : int
#         Dimension along which to apply transform (default: -1)
#     fp16 : bool
#         Whether to use half precision
#     in_place : bool
#         Whether to modify input in-place (saves memory)
#     steepness : float
#         Steepness of sigmoid approximation (default: 50, higher = sharper transition)
#     """
#
#     def __init__(self, seq_len, dim=-1, fp16=False, in_place=False, steepness=50):
#         super().__init__()
#         self.dim = dim
#         self.fp16 = fp16
#         self.in_place = in_place
#         self.seq_len = seq_len
#         self.steepness = steepness
#
#         # Pre-compute frequency grid for efficiency
#         self._create_frequency_grid(seq_len)
#
#     def _create_frequency_grid(self, n):
#         """Create frequency grid for differentiable Hilbert transform."""
#         # Frequency grid: [0, 1/n, 2/n, ..., (n-1)/2/n, -(n/2)/n, ..., -1/n]
#         positive_freqs = torch.arange(0, (n - 1) // 2 + 1) / float(n)
#         negative_freqs = torch.arange(-(n // 2), 0) / float(n)
#         f = torch.cat([positive_freqs, negative_freqs])
#         self.register_buffer("freq_grid", f)
#
#     def forward(self, x):
#         """
#         Apply Hilbert transform to extract phase and amplitude.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input signal
#
#         Returns
#         -------
#         torch.Tensor
#             Output with shape [..., 2] where last dimension contains [phase, amplitude]
#         """
#         # Handle precision
#         if self.fp16:
#             x = x.half()
#
#         # Clone if not in-place
#         if not self.in_place:
#             x = x.clone()
#
#         # Store original dtype for restoration
#         orig_dtype = x.dtype
#
#         # FFT requires float32
#         x_float = x.float()
#
#         # Apply FFT
#         x_fft = fft(x_float, n=self.seq_len, dim=self.dim)
#
#         # Apply differentiable frequency domain filter
#         # H(f) ≈ 2 * sigmoid(steepness * f)
#         # This approximates: 2 for positive frequencies, 0 for negative frequencies
#         step_function = torch.sigmoid(self.steepness * self.freq_grid.type_as(x_float))
#         x_fft_hilbert = x_fft * 2 * step_function
#
#         # Convert back to time domain
#         x_analytic = ifft(x_fft_hilbert, dim=self.dim)
#
#         # Extract phase and amplitude
#         phase = torch.atan2(x_analytic.imag, x_analytic.real)
#         amplitude = x_analytic.abs()
#
#         # Stack phase and amplitude
#         output = torch.stack([phase, amplitude], dim=-1)
#
#         # Restore original precision if needed
#         if orig_dtype == torch.float16:
#             output = output.half()
#
#         return output
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_Hilbert.py
# --------------------------------------------------------------------------------
