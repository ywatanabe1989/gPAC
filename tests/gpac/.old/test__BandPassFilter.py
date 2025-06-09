#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 20:45:00 (ywatanabe)"
# File: ./tests/gpac/test__BandPassFilter.py

import torch
import pytest
import numpy as np
import sys
import os

from gpac._BandPassFilter import BandPassFilter


class TestBandPassFilter:
    """Test suite for BandPassFilter following AAA pattern and testing guidelines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.seq_len = 1000
        self.fs = 256.0  # Standard EEG sampling rate
        self.pha_start_hz = 4.0
        self.pha_end_hz = 8.0
        self.pha_n_bands = 2
        self.amp_start_hz = 30.0
        self.amp_end_hz = 100.0
        self.amp_n_bands = 3

    # =============================================================================
    # Forward Pass Tests
    # =============================================================================

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input (batch, segments, time)."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )
        batch_size, n_segments = 4, 2
        x = torch.randn(batch_size, n_segments, self.seq_len)

        # Act
        output = filter_obj(x)

        # Assert
        expected_shape = (
            batch_size,
            n_segments,
            self.pha_n_bands + self.amp_n_bands,
            self.seq_len,
        )
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    def test_forward_pass_trainable_mode(self):
        """Test forward pass in trainable mode."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=True,
        )
        x = torch.randn(2, 1, self.seq_len)

        # Act
        output = filter_obj(x)

        # Assert
        assert output.shape == (2, 1, self.pha_n_bands + self.amp_n_bands, self.seq_len)
        assert hasattr(filter_obj.filter, "low_hz_")  # Trainable parameters
        assert hasattr(filter_obj.filter, "band_hz_")

    # =============================================================================
    # Input Shape Tests
    # =============================================================================

    def test_input_shape_validation(self):
        """Test various input shapes are handled correctly."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )

        # Test various valid shapes
        valid_shapes = [
            (2, 1, self.seq_len),  # (batch, segments, time)
            (4, 3, self.seq_len),  # Multiple segments
            (1, 1, self.seq_len),  # Single sample
        ]

        for shape in valid_shapes:
            x = torch.randn(*shape)
            output = filter_obj(x)
            assert output.shape[:-1] == shape[:-1] + (
                self.pha_n_bands + self.amp_n_bands,
            )

    def test_invalid_input_shapes(self):
        """Test that invalid input shapes are handled correctly."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )

        # Test that 1D and 2D inputs are automatically converted to 3D
        # 1D input (seq_len,) -> (1, 1, seq_len)
        x_1d = torch.randn(self.seq_len)
        output_1d = filter_obj(x_1d)
        assert output_1d.shape == (
            1,
            1,
            self.pha_n_bands + self.amp_n_bands,
            self.seq_len,
        )

        # 2D input (batch, seq_len) -> (batch, 1, seq_len)
        x_2d = torch.randn(2, self.seq_len)
        output_2d = filter_obj(x_2d)
        assert output_2d.shape == (
            2,
            1,
            self.pha_n_bands + self.amp_n_bands,
            self.seq_len,
        )

        # 4D input should raise error
        x_4d = torch.randn(2, 1, 1, self.seq_len)
        with pytest.raises((ValueError, AssertionError)):
            filter_obj(x_4d)

    # =============================================================================
    # Output Shape Tests
    # =============================================================================

    def test_output_shape_consistency(self):
        """Test output shape is consistent with input batch and segments."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )

        # Test different batch and segment sizes
        test_cases = [
            (1, 1),  # Single batch, single segment
            (5, 1),  # Multiple batches, single segment
            (1, 10),  # Single batch, multiple segments
            (4, 8),  # Multiple batches and segments
        ]

        for batch_size, n_segments in test_cases:
            x = torch.randn(batch_size, n_segments, self.seq_len)
            output = filter_obj(x)

            assert output.shape[0] == batch_size
            assert output.shape[1] == n_segments
            assert output.shape[2] == self.pha_n_bands + self.amp_n_bands
            assert output.shape[3] == self.seq_len

    # =============================================================================
    # Parameter Validation Tests
    # =============================================================================

    def test_parameter_validation_frequencies(self):
        """Test frequency parameter validation."""
        # Test invalid frequency ranges
        with pytest.raises(ValueError, match="Invalid phase frequency range"):
            BandPassFilter(
                seq_len=self.seq_len,
                fs=self.fs,
                pha_start_hz=10.0,
                pha_end_hz=5.0,  # End < Start
                pha_n_bands=2,
                amp_start_hz=30.0,
                amp_end_hz=100.0,
                amp_n_bands=3,
                trainable=False,
            )

        with pytest.raises(ValueError, match="Invalid amplitude frequency range"):
            BandPassFilter(
                seq_len=self.seq_len,
                fs=self.fs,
                pha_start_hz=4.0,
                pha_end_hz=8.0,
                pha_n_bands=2,
                amp_start_hz=100.0,
                amp_end_hz=50.0,  # End < Start
                amp_n_bands=3,
                trainable=False,
            )

    def test_parameter_validation_bands(self):
        """Test band count parameter validation."""
        # Test invalid band counts
        with pytest.raises(ValueError, match="Number of bands must be positive"):
            BandPassFilter(
                seq_len=self.seq_len,
                fs=self.fs,
                pha_start_hz=4.0,
                pha_end_hz=8.0,
                pha_n_bands=0,  # Invalid
                amp_start_hz=30.0,
                amp_end_hz=100.0,
                amp_n_bands=3,
                trainable=False,
            )

    def test_parameter_validation_sampling(self):
        """Test sampling rate and sequence length validation."""
        # Test invalid sampling rate
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            BandPassFilter(
                seq_len=self.seq_len,
                fs=0,  # Invalid
                pha_start_hz=4.0,
                pha_end_hz=8.0,
                pha_n_bands=2,
                amp_start_hz=30.0,
                amp_end_hz=100.0,
                amp_n_bands=3,
                trainable=False,
            )

        # Test invalid sequence length
        with pytest.raises(ValueError, match="seq_len must be positive"):
            BandPassFilter(
                seq_len=0,  # Invalid
                fs=self.fs,
                pha_start_hz=4.0,
                pha_end_hz=8.0,
                pha_n_bands=2,
                amp_start_hz=30.0,
                amp_end_hz=100.0,
                amp_n_bands=3,
                trainable=False,
            )

    def test_nyquist_frequency_validation(self):
        """Test that frequencies respect Nyquist limit."""
        # Test phase frequency exceeding Nyquist
        with pytest.raises(ValueError, match="exceeds Nyquist frequency"):
            BandPassFilter(
                seq_len=self.seq_len,
                fs=100.0,  # Nyquist = 50 Hz
                pha_start_hz=4.0,
                pha_end_hz=60.0,  # > Nyquist
                pha_n_bands=2,
                amp_start_hz=30.0,
                amp_end_hz=40.0,
                amp_n_bands=3,
                trainable=False,
            )

    # =============================================================================
    # Differentiability Tests
    # =============================================================================

    def test_backward_pass_static_mode(self):
        """Test that gradients don't flow in static mode."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )
        x = torch.randn(2, 1, self.seq_len, requires_grad=True)

        # Act
        output = filter_obj(x)
        loss = output.sum()

        # Assert - no parameters should have gradients
        for name, param in filter_obj.named_parameters():
            assert not param.requires_grad

    def test_backward_pass_trainable_mode(self):
        """Test gradient flow in trainable mode."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=True,
        )
        x = torch.randn(2, 1, self.seq_len, requires_grad=True)

        # Act
        output = filter_obj(x)
        loss = output.sum()
        loss.backward()

        # Assert - trainable parameters should have gradients
        trainable_params = ["pha_low", "pha_high", "amp_low", "amp_high"]
        for name, param in filter_obj.named_parameters():
            if any(tp in name for tp in trainable_params):
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_gradient_stability(self):
        """Test gradient stability with different inputs."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=True,
        )

        # Test with different input magnitudes
        input_scales = [1e-3, 1.0, 1e3]

        for scale in input_scales:
            x = torch.randn(2, 1, self.seq_len) * scale
            x.requires_grad = True

            output = filter_obj(x)
            loss = output.mean()
            loss.backward()

            # Check gradients are finite and reasonable
            assert torch.isfinite(x.grad).all()
            assert x.grad.abs().max() < 1e6  # No gradient explosion

            # Clear gradients for next iteration
            filter_obj.zero_grad()

    # =============================================================================
    # Edge Cases and Special Inputs
    # =============================================================================

    def test_zero_input(self):
        """Test handling of zero input."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )
        x = torch.zeros(2, 1, self.seq_len)

        # Act
        output = filter_obj(x)

        # Assert
        assert torch.isfinite(output).all()
        assert output.abs().max() < 1e-6  # Output should be near zero

    def test_constant_input(self):
        """Test handling of constant DC input."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )
        x = torch.ones(2, 1, self.seq_len) * 5.0  # DC signal

        # Act
        output = filter_obj(x)

        # Assert - bandpass should remove DC component
        assert torch.isfinite(output).all()
        assert output.mean().abs() < 10.0  # Reduced DC component after filtering

    def test_fp16_mode(self):
        """Test half precision mode."""
        # Arrange
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
            fp16=True,
        )
        x = torch.randn(2, 1, self.seq_len)

        # Act
        output = filter_obj(x)

        # Assert
        assert output.dtype == torch.float16  # With fp16=True, output should be float16
        assert torch.isfinite(output).all()

    # =============================================================================
    # Integration Tests
    # =============================================================================

    def test_frequency_band_generation(self):
        """Test that frequency bands are generated correctly."""
        # Arrange & Act
        filter_obj = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=4.0,
            pha_end_hz=8.0,
            pha_n_bands=2,
            amp_start_hz=30.0,
            amp_end_hz=90.0,
            amp_n_bands=3,
            trainable=False,
        )

        # Assert
        assert len(filter_obj.pha_mids) == 2
        assert len(filter_obj.amp_mids) == 3
        assert filter_obj.pha_mids[0] >= 4.0
        assert filter_obj.pha_mids[-1] <= 8.0
        assert filter_obj.amp_mids[0] >= 30.0
        assert filter_obj.amp_mids[-1] <= 90.0

    def test_mode_switching(self):
        """Test switching between static and trainable modes."""
        # Test that both modes produce valid outputs
        x = torch.randn(2, 1, self.seq_len)

        # Static mode
        static_filter = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=False,
        )
        static_output = static_filter(x)

        # Trainable mode
        trainable_filter = BandPassFilter(
            seq_len=self.seq_len,
            fs=self.fs,
            pha_start_hz=self.pha_start_hz,
            pha_end_hz=self.pha_end_hz,
            pha_n_bands=self.pha_n_bands,
            amp_start_hz=self.amp_start_hz,
            amp_end_hz=self.amp_end_hz,
            amp_n_bands=self.amp_n_bands,
            trainable=True,
        )
        trainable_output = trainable_filter(x)

        # Both should produce valid outputs of same shape
        assert static_output.shape == trainable_output.shape
        assert torch.isfinite(static_output).all()
        assert torch.isfinite(trainable_output).all()


# Main block for standalone testing

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-28 19:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_BandPassFilter.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Unified BandPassFilter module that combines static and differentiable filtering modes.
# Provides a single interface for both trainable and non-trainable bandpass filtering.
# """
#
# import torch
# import torch.nn as nn
#
# from ._Filters._DifferentiableBandpassFilter import DifferentiableBandpassFilter
# from ._Filters._StaticBandpassFilter import StaticBandpassFilter
#
#
# class BandPassFilter(nn.Module):
#     """
#     Unified bandpass filter that switches between static and differentiable modes.
#
#     Parameters
#     ----------
#     seq_len : int
#         Length of the input sequence
#     fs : float
#         Sampling frequency
#     pha_start_hz : float
#         Start frequency for phase bands
#     pha_end_hz : float
#         End frequency for phase bands
#     pha_n_bands : int
#         Number of phase bands
#     amp_start_hz : float
#         Start frequency for amplitude bands
#     amp_end_hz : float
#         End frequency for amplitude bands
#     amp_n_bands : int
#         Number of amplitude bands
#     fp16 : bool
#         Whether to use half precision
#     trainable : bool
#         Whether to use trainable (differentiable) filters
#     """
#
#     def __init__(
#         self,
#         seq_len,
#         fs,
#         pha_start_hz=2,
#         pha_end_hz=20,
#         pha_n_bands=50,
#         amp_start_hz=60,
#         amp_end_hz=160,
#         amp_n_bands=30,
#         fp16=False,
#         trainable=False,
#     ):
#         super().__init__()
#
#         self.seq_len = seq_len
#         self.fs = fs
#         self.fp16 = fp16
#         self.trainable = trainable
#         self.pha_n_bands = pha_n_bands
#         self.amp_n_bands = amp_n_bands
#
#         if trainable:
#             # Use differentiable bandpass filter
#             self.filter = DifferentiableBandpassFilter(
#                 sig_len=seq_len,
#                 fs=fs,
#                 pha_low_hz=pha_start_hz,
#                 pha_high_hz=pha_end_hz,
#                 pha_n_bands=pha_n_bands,
#                 amp_low_hz=amp_start_hz,
#                 amp_high_hz=amp_end_hz,
#                 amp_n_bands=amp_n_bands,
#                 cycle=3,
#                 fp16=fp16,
#             )
#             # Store mid frequencies for PAC
#             self.pha_mids = self.filter.pha_mids
#             self.amp_mids = self.filter.amp_mids
#         else:
#             # Use static bandpass filter
#             # First calculate the bands
#             pha_bands = self._calc_bands_pha(pha_start_hz, pha_end_hz, pha_n_bands)
#             amp_bands = self._calc_bands_amp(amp_start_hz, amp_end_hz, amp_n_bands)
#             all_bands = torch.vstack([pha_bands, amp_bands])
#
#             self.filter = StaticBandpassFilter(
#                 bands=all_bands,
#                 fs=fs,
#                 seq_len=seq_len,
#                 fp16=fp16,
#             )
#             # Store mid frequencies for PAC
#             self.pha_mids = pha_bands.mean(-1)
#             self.amp_mids = amp_bands.mean(-1)
#
#     def forward(self, x, edge_len=0):
#         """
#         Apply bandpass filtering.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input signal with shape (batch_size, n_segments, seq_len)
#         edge_len : int
#             Number of samples to remove from edges
#
#         Returns
#         -------
#         torch.Tensor
#             Filtered signal with shape (batch_size, n_segments, n_bands, seq_len)
#         """
#         return self.filter(x, edge_len=edge_len)
#
#     @staticmethod
#     def _calc_bands_pha(start_hz=2, end_hz=20, n_bands=100):
#         """Calculate phase frequency bands."""
#         start_hz = start_hz if start_hz is not None else 2
#         end_hz = end_hz if end_hz is not None else 20
#         mid_hz = torch.linspace(start_hz, end_hz, n_bands)
#         return torch.cat(
#             (
#                 mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 4.0,
#                 mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 4.0,
#             ),
#             dim=1,
#         )
#
#     @staticmethod
#     def _calc_bands_amp(start_hz=30, end_hz=160, n_bands=100):
#         """Calculate amplitude frequency bands."""
#         start_hz = start_hz if start_hz is not None else 30
#         end_hz = end_hz if end_hz is not None else 160
#         mid_hz = torch.linspace(start_hz, end_hz, n_bands)
#         return torch.cat(
#             (
#                 mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 8.0,
#                 mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 8.0,
#             ),
#             dim=1,
#         )
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_BandPassFilter.py
# --------------------------------------------------------------------------------
