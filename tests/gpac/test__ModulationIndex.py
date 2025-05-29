#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 20:55:00 (ywatanabe)"
# File: ./tests/gpac/test__ModulationIndex.py

import torch
import pytest
import numpy as np
from gpac._ModulationIndex import ModulationIndex


class TestModulationIndex:
    """Test suite for ModulationIndex following AAA pattern and testing guidelines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.n_bins = 18
        self.temperature = 0.1
        self.batch_size = 2
        self.n_channels = 3
        self.n_freqs_phase = 4
        self.n_freqs_amplitude = 5
        self.n_segments = 6
        self.seq_len = 100

    # =============================================================================
    # Forward Pass Tests
    # =============================================================================

    def test_forward_pass_basic(self):
        """Test basic forward pass with default parameters."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(
            self.batch_size,
            self.n_channels,
            self.n_freqs_phase,
            self.n_segments,
            self.seq_len,
        )
        # Wrap phase to [-pi, pi]
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        amplitude = torch.abs(
            torch.randn(
                self.batch_size,
                self.n_channels,
                self.n_freqs_amplitude,
                self.n_segments,
                self.seq_len,
            )
        )

        # Act
        output = mi(phase, amplitude, compute_distributions=True)

        # Assert
        assert isinstance(output, dict)
        assert "mi" in output
        assert "mi_per_segment" in output
        assert "amplitude_distributions" in output
        assert "phase_bin_centers" in output
        assert "phase_bin_edges" in output

        # Check shapes
        assert output["mi"].shape == (
            self.batch_size,
            self.n_channels,
            self.n_freqs_phase,
            self.n_freqs_amplitude,
        )
        assert output["mi_per_segment"].shape == (
            self.batch_size,
            self.n_channels,
            self.n_freqs_phase,
            self.n_freqs_amplitude,
            self.n_segments,
        )
        assert output["amplitude_distributions"].shape == (
            self.batch_size,
            self.n_channels,
            self.n_freqs_phase,
            self.n_freqs_amplitude,
            self.n_bins,
        )
        assert torch.isfinite(output["mi"]).all()

    def test_forward_pass_custom_parameters(self):
        """Test forward pass with custom parameters."""
        # Arrange
        custom_bins = 36
        custom_temp = 0.05
        mi = ModulationIndex(n_bins=custom_bins, temperature=custom_temp)
        phase = torch.randn(1, 1, 1, 1, 50) * np.pi  # Phase in radians
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        amplitude = torch.abs(torch.randn(1, 1, 1, 1, 50))

        # Act
        output = mi(phase, amplitude, compute_distributions=True)

        # Assert
        assert output["amplitude_distributions"].shape[-1] == custom_bins
        assert output["phase_bin_centers"].shape[0] == custom_bins
        assert output["phase_bin_edges"].shape[0] == custom_bins + 1

    # =============================================================================
    # Input Shape Tests
    # =============================================================================

    def test_input_shape_validation(self):
        """Test that input shapes are validated correctly."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Valid minimal shape
        phase = torch.randn(1, 1, 1, 1, 10)
        amplitude = torch.randn(1, 1, 1, 1, 10)
        output = mi(phase, amplitude)
        assert output["mi"].shape == (1, 1, 1, 1)

    def test_input_shape_mismatch_batch(self):
        """Test error on batch size mismatch."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(2, 1, 1, 1, 10)
        amplitude = torch.randn(3, 1, 1, 1, 10)  # Different batch size

        # Act & Assert
        with pytest.raises(ValueError, match="Batch size mismatch"):
            mi(phase, amplitude)

    def test_input_shape_mismatch_channels(self):
        """Test error on channel size mismatch."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(1, 2, 1, 1, 10)
        amplitude = torch.randn(1, 3, 1, 1, 10)  # Different channels

        # Act & Assert
        with pytest.raises(ValueError, match="Channel size mismatch"):
            mi(phase, amplitude)

    def test_input_shape_mismatch_time(self):
        """Test error on time dimension mismatch."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(1, 1, 1, 1, 10)
        amplitude = torch.randn(1, 1, 1, 1, 20)  # Different time length

        # Act & Assert
        with pytest.raises(ValueError, match="Time dimension mismatch"):
            mi(phase, amplitude)

    def test_input_shape_mismatch_segments(self):
        """Test error on segment dimension mismatch."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(1, 1, 2, 3, 10)
        amplitude = torch.randn(1, 1, 2, 4, 10)  # Different segments

        # Act & Assert
        with pytest.raises(ValueError, match="Segment dimension mismatch"):
            mi(phase, amplitude)

    # =============================================================================
    # Output Shape Tests
    # =============================================================================

    def test_output_shapes_consistency(self):
        """Test that all output shapes are consistent."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        test_cases = [
            (1, 1, 1, 1, 1, 10),  # Minimal
            (2, 3, 4, 5, 6, 50),  # Standard
            (5, 2, 10, 8, 3, 100),  # Large
        ]

        for batch, ch, f_pha, f_amp, seg, time in test_cases:
            phase = torch.randn(batch, ch, f_pha, seg, time)
            amplitude = torch.randn(batch, ch, f_amp, seg, time)

            output = mi(phase, amplitude, compute_distributions=True)

            # Check all shapes
            assert output["mi"].shape == (batch, ch, f_pha, f_amp)
            assert output["mi_per_segment"].shape == (batch, ch, f_pha, f_amp, seg)
            assert output["amplitude_distributions"].shape == (
                batch,
                ch,
                f_pha,
                f_amp,
                self.n_bins,
            )

    # =============================================================================
    # Parameter Validation Tests
    # =============================================================================

    def test_parameter_validation_n_bins(self):
        """Test n_bins parameter validation."""
        # Test invalid n_bins
        with pytest.raises(ValueError, match="n_bins must be positive"):
            ModulationIndex(n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be positive"):
            ModulationIndex(n_bins=-5)

    def test_parameter_validation_temperature(self):
        """Test temperature parameter validation."""
        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            ModulationIndex(n_bins=18, temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            ModulationIndex(n_bins=18, temperature=-0.1)

    # =============================================================================
    # Differentiability Tests
    # =============================================================================

    def test_backward_pass(self):
        """Test gradient flow through ModulationIndex."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(2, 1, 2, 1, 50, requires_grad=True)
        amplitude = torch.randn(2, 1, 2, 1, 50, requires_grad=True)

        # Act
        output = mi(phase, amplitude)
        loss = output["mi"].sum()
        loss.backward()

        # Assert
        assert phase.grad is not None
        assert amplitude.grad is not None
        assert torch.isfinite(phase.grad).all()
        assert torch.isfinite(amplitude.grad).all()

    def test_gradient_stability(self):
        """Test gradient stability with different scales."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        scales = [1e-2, 1.0, 1e2]

        for scale in scales:
            phase = torch.randn(1, 1, 1, 1, 50, requires_grad=True)
            amplitude = torch.abs(torch.randn(1, 1, 1, 1, 50)) * scale
            amplitude.requires_grad = True

            output = mi(phase, amplitude)
            loss = output["mi"].mean()
            loss.backward()

            # Check gradients are finite
            assert torch.isfinite(phase.grad).all()
            assert torch.isfinite(amplitude.grad).all()

    def test_soft_binning_differentiability(self):
        """Test that soft binning maintains differentiability."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins, temperature=0.1)

        # Create phase-dependent amplitude to ensure phase gradients
        phase = torch.linspace(-np.pi, np.pi, 100).reshape(1, 1, 1, 1, 100)
        phase.requires_grad = True
        # Amplitude that varies with phase - this ensures phase gradients
        amplitude = torch.exp(-((phase - 0.5) ** 2) / 0.5) + 0.1

        # Act
        output = mi(phase, amplitude)
        loss = output["mi"].sum()
        loss.backward()

        # Assert - gradients should flow
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()
        # Phase should have gradients since amplitude depends on phase
        assert phase.grad.abs().sum() > 0

    # =============================================================================
    # Edge Cases and Special Inputs
    # =============================================================================

    def test_uniform_amplitude_distribution(self):
        """Test MI calculation with uniform amplitude distribution."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.rand(1, 1, 1, 1, 1000) * 2 * np.pi - np.pi
        amplitude = torch.ones_like(phase)  # Constant amplitude

        # Act
        output = mi(phase, amplitude)

        # Assert - MI should be close to 0 for uniform distribution
        assert output["mi"].item() < 0.1  # Small MI value

    def test_concentrated_amplitude_distribution(self):
        """Test MI calculation with concentrated amplitude distribution."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Create phase-locked amplitude
        phase = torch.linspace(-np.pi, np.pi, 1000).reshape(1, 1, 1, 1, 1000)
        # Amplitude peaks at phase = 0
        amplitude = torch.exp(-(phase**2) / 0.1)

        # Act
        output = mi(phase, amplitude)

        # Assert - MI should be high for concentrated distribution
        assert output["mi"].item() > 0.5  # High MI value

    def test_zero_amplitude(self):
        """Test handling of zero amplitude."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(1, 1, 1, 1, 50)
        amplitude = torch.zeros_like(phase)

        # Act
        output = mi(phase, amplitude)

        # Assert
        assert torch.isfinite(output["mi"]).all()
        # MI should be 1.0 when amplitude is 0 (due to uniform distribution)
        assert torch.allclose(output["mi"], torch.ones_like(output["mi"]), atol=1e-6)

    def test_phase_wrapping(self):
        """Test that phase values are handled correctly (circular data)."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Test with phase values outside [-pi, pi]
        phase = torch.randn(1, 1, 1, 1, 100) * 10  # Large range
        amplitude = torch.abs(torch.randn_like(phase))

        # Act
        output = mi(phase, amplitude)

        # Assert
        assert torch.isfinite(output["mi"]).all()
        assert (output["mi"] >= 0).all() and (output["mi"] <= 1).all()

    # =============================================================================
    # Entropy and MI Calculation Tests
    # =============================================================================

    def test_mi_range(self):
        """Test that MI values are in valid range [0, 1]."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Test multiple random inputs
        for _ in range(10):
            phase = torch.randn(2, 2, 2, 2, 50)
            amplitude = torch.abs(torch.randn(2, 2, 2, 2, 50))

            output = mi(phase, amplitude)

            # MI should be between 0 and 1
            assert (output["mi"] >= 0).all()
            assert (output["mi"] <= 1).all()

    def test_amplitude_distribution_normalization(self):
        """Test that amplitude distributions are properly normalized."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        phase = torch.randn(1, 1, 1, 1, 100)
        amplitude = torch.abs(torch.randn_like(phase))

        # Act
        output = mi(phase, amplitude, compute_distributions=True)

        # Assert - distributions should sum to 1
        dist_sums = output["amplitude_distributions"].sum(dim=-1)
        assert torch.allclose(dist_sums, torch.ones_like(dist_sums), atol=1e-6)

    # =============================================================================
    # Temperature Effects Tests
    # =============================================================================

    def test_temperature_effects(self):
        """Test how temperature affects soft binning."""
        # Arrange
        # Create phase that's concentrated around 0 with some spread
        phase = torch.randn(1, 1, 1, 1, 100) * 0.5  # Concentrated around 0
        # Create amplitude that varies with phase
        amplitude = torch.exp(-phase.abs())  # Higher amplitude near phase=0

        temperatures = [0.01, 0.1, 1.0]
        mi_values = []

        for temp in temperatures:
            mi = ModulationIndex(n_bins=self.n_bins, temperature=temp)
            output = mi(phase, amplitude)
            mi_values.append(output["mi"].item())

        # Assert - different temperatures should give different results
        # Lower temperature should give higher MI (sharper binning)
        assert mi_values[0] > mi_values[1] > mi_values[2]

    def test_low_temperature_approximates_hard_binning(self):
        """Test that low temperature approximates hard binning."""
        # Arrange
        mi_low_temp = ModulationIndex(n_bins=self.n_bins, temperature=0.001)
        mi_high_temp = ModulationIndex(n_bins=self.n_bins, temperature=1.0)

        # Create phase values concentrated in one bin with amplitude modulation
        phase = torch.zeros(1, 1, 1, 1, 100) + 0.1 * torch.randn(1, 1, 1, 1, 100)
        amplitude = torch.ones_like(phase) + 0.5 * torch.randn_like(phase).abs()

        # Act
        output_low = mi_low_temp(phase, amplitude)
        output_high = mi_high_temp(phase, amplitude)

        # Assert - low temperature should give higher MI (more concentrated)
        assert output_low["mi"].item() > output_high["mi"].item()

    # =============================================================================
    # Integration Tests
    # =============================================================================

    def test_multiple_segments_averaging(self):
        """Test that MI is correctly averaged across segments."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)
        n_segments = 5

        # Create data with varying MI across segments
        phase = torch.randn(1, 1, 1, n_segments, 100)
        amplitude = torch.abs(torch.randn_like(phase))

        # Act
        output = mi(phase, amplitude, compute_distributions=True)

        # Assert
        assert output["mi_per_segment"].shape[-1] == n_segments
        # Average of per-segment MI should equal overall MI
        mi_avg = output["mi_per_segment"].mean(dim=-1)
        assert torch.allclose(mi_avg, output["mi"], atol=1e-6)

    def test_phase_bin_centers_and_edges(self):
        """Test that phase bins are correctly initialized."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Act
        phase = torch.randn(1, 1, 1, 1, 10)
        amplitude = torch.randn_like(phase)
        output = mi(phase, amplitude)

        # Assert
        edges = output["phase_bin_edges"]
        centers = output["phase_bin_centers"]

        # Edges should span [-pi, pi]
        assert torch.allclose(edges[0], torch.tensor(-np.pi))
        assert torch.allclose(edges[-1], torch.tensor(np.pi))

        # Centers should be midpoints
        expected_centers = (edges[1:] + edges[:-1]) / 2
        assert torch.allclose(centers, expected_centers)

    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        # Arrange
        mi = ModulationIndex(n_bins=self.n_bins)

        # Create identical data in different batch positions
        single_phase = torch.randn(1, 1, 1, 1, 100)
        single_amp = torch.abs(torch.randn_like(single_phase))

        batch_phase = single_phase.repeat(3, 1, 1, 1, 1)
        batch_amp = single_amp.repeat(3, 1, 1, 1, 1)

        # Act
        single_output = mi(single_phase, single_amp)
        batch_output = mi(batch_phase, batch_amp)

        # Assert - all batch elements should have same MI
        for i in range(3):
            assert torch.allclose(
                batch_output["mi"][i], single_output["mi"][0], atol=1e-6
            )


# Main block for standalone testing

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_ModulationIndex.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-08 05:39:21 (ywatanabe)"
# # Author: Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)
# # File: /data/gpac/src/gpac/_ModulationIndex.py
#
# import math
# from typing import Dict, Optional, Tuple
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ModulationIndex(nn.Module):
#     """
#     Compute modulation index to quantify phase-amplitude coupling.
#
#     The modulation index quantifies the non-uniformity of amplitude
#     distribution across phase bins using Shannon entropy.
#
#     Uses differentiable soft binning with softmax for gradient-based optimization.
#     """
#
#     def __init__(
#         self,
#         n_bins: int = 18,
#         temperature: float = 0.1
#     ) -> None:
#         """
#         Initialize ModulationIndex calculator.
#
#         Parameters
#         ----------
#         n_bins : int
#             Number of phase bins (default: 18, i.e., 20 degrees per bin)
#         temperature : float
#             Temperature parameter for soft binning (default: 0.1)
#             Lower values approach hard binning
#         """
#         super().__init__()
#
#         # Input validation
#         if n_bins <= 0:
#             raise ValueError(f"n_bins must be positive, got {n_bins}")
#         if temperature <= 0:
#             raise ValueError(f"temperature must be positive, got {temperature}")
#
#         self.n_bins = n_bins
#         self.epsilon = 1e-10  # To avoid log(0)
#         self.temperature = temperature
#
#         # Pre-calculate phase bins and Shannon entropy normalization
#         self.register_buffer(
#             "phase_bins",
#             torch.linspace(-np.pi, np.pi, n_bins + 1)
#         )
#         self.register_buffer(
#             "phase_bin_centers",
#             (self.phase_bins[1:] + self.phase_bins[:-1]) / 2
#         )
#         self.register_buffer(
#             "uniform_entropy",
#             torch.tensor(np.log(n_bins))
#         )
#
#     def forward(self, phase: torch.Tensor, amplitude: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Calculate modulation index from phase and amplitude signals.
#
#         Parameters
#         ----------
#         phase : torch.Tensor
#             Phase values in radians with shape:
#             (batch, channels, freqs_phase, segments, time)
#         amplitude : torch.Tensor
#             Amplitude values with shape:
#             (batch, channels, freqs_amplitude, segments, time)
#
#         Returns
#         -------
#         dict
#             Dictionary containing:
#             - 'mi': Modulation index values, shape (batch, channels, freqs_phase, freqs_amplitude)
#             - 'mi_per_segment': MI values per segment before averaging
#             - 'amplitude_distributions': Amplitude probability distributions across phase bins
#             - 'phase_bin_centers': Center values of phase bins in radians
#             - 'phase_bin_edges': Edge values of phase bins in radians
#         """
#         # Get dimensions
#         batch, channels, freqs_phase, segments, time = phase.shape
#         batch_amp, channels_amp, freqs_amplitude, segments_amp, time_amp = amplitude.shape
#
#         # Validate shape compatibility
#         if batch != batch_amp:
#             raise ValueError(f"Batch size mismatch: phase={batch}, amplitude={batch_amp}")
#         if channels != channels_amp:
#             raise ValueError(f"Channel size mismatch: phase={channels}, amplitude={channels_amp}")
#         if time != time_amp:
#             raise ValueError(f"Time dimension mismatch: phase={time}, amplitude={time_amp}")
#         if phase.shape[2:4] != amplitude.shape[2:4]:
#             raise ValueError(
#                 f"Frequency/segment dimension mismatch: phase={phase.shape[2:4]}, amplitude={amplitude.shape[2:4]}"
#             )
#
#         # Flatten for binning
#         phase_flat = phase.reshape(-1)
#         amplitude_flat = amplitude.reshape(-1)
#
#         # Soft binning for differentiability
#         weights = self._phase_binning(phase_flat)
#         # weights shape: (n_phases, n_bins)
#
#         # Compute amplitude distribution across phase bins
#         mi_all = []
#         mi_per_segment_all = []
#         amp_dists_all = []
#
#         # Process each batch-channel-frequency combination
#         total_combinations = batch * channels * freqs_phase * freqs_amplitude
#         for idx in range(total_combinations):
#             # Calculate indices
#             b = idx // (channels * freqs_phase * freqs_amplitude)
#             remainder = idx % (channels * freqs_phase * freqs_amplitude)
#             c = remainder // (freqs_phase * freqs_amplitude)
#             remainder = remainder % (freqs_phase * freqs_amplitude)
#             f_pha = remainder // freqs_amplitude
#             f_amp = remainder % freqs_amplitude
#
#             # Extract data for this combination
#             start_idx = idx * segments * time
#             end_idx = start_idx + segments * time
#             weights_comb = weights[start_idx:end_idx]
#             amplitude_comb = amplitude_flat[start_idx:end_idx]
#
#             # Compute MI per segment
#             segment_mis = []
#             for seg in range(segments):
#                 seg_start = seg * time
#                 seg_end = seg_start + time
#                 seg_weights = weights_comb[seg_start:seg_end]
#                 seg_amplitude = amplitude_comb[seg_start:seg_end]
#
#                 # Weighted amplitude distribution
#                 amp_dist = torch.sum(
#                     seg_weights.T * seg_amplitude.unsqueeze(0),
#                     dim=1
#                 ) / (seg_weights.sum(dim=0) + self.epsilon)
#
#                 # Normalize
#                 amp_dist = amp_dist / (amp_dist.sum() + self.epsilon)
#
#                 # Calculate entropy and MI
#                 seg_entropy = -torch.sum(amp_dist * torch.log(amp_dist + self.epsilon))
#                 seg_mi = (self.uniform_entropy - seg_entropy) / self.uniform_entropy
#                 segment_mis.append(seg_mi)
#
#             # Average across segments
#             mi_per_segment = torch.stack(segment_mis)
#             mi = mi_per_segment.mean()
#
#             # Overall amplitude distribution
#             amp_dist_all = torch.sum(
#                 weights_comb.T * amplitude_comb.unsqueeze(0),
#                 dim=1
#             ) / (weights_comb.sum(dim=0) + self.epsilon)
#             amp_dist_all = amp_dist_all / (amp_dist_all.sum() + self.epsilon)
#
#             mi_all.append(mi)
#             mi_per_segment_all.append(mi_per_segment)
#             amp_dists_all.append(amp_dist_all)
#
#         # Reshape results back to original dimensions
#         mi_tensor = torch.stack(mi_all).reshape(batch, channels, freqs_phase, freqs_amplitude)
#         mi_per_segment_tensor = torch.stack(mi_per_segment_all).reshape(
#             batch, channels, freqs_phase, freqs_amplitude, segments
#         )
#         amp_dists_tensor = torch.stack(amp_dists_all).reshape(
#             batch, channels, freqs_phase, freqs_amplitude, self.n_bins
#         )
#
#         return {
#             "mi": mi_tensor,
#             "mi_per_segment": mi_per_segment_tensor,
#             "amplitude_distributions": amp_dists_tensor,
#             "phase_bin_centers": self.phase_bin_centers,
#             "phase_bin_edges": self.phase_bins,
#         }
#
#     def _phase_binning(self, phase: torch.Tensor) -> torch.Tensor:
#         """
#         Assign phases to bins using differentiable soft binning with softmax.
#
#         Parameters
#         ----------
#         phase : torch.Tensor
#             Phase values in radians, shape (n_phases,)
#
#         Returns
#         -------
#         torch.Tensor
#             Soft bin assignments, shape (n_phases, n_bins)
#         """
#         # Expand dimensions for broadcasting
#         phase_expanded = phase.unsqueeze(-1)  # (n_phases, 1)
#         centers_expanded = self.phase_bin_centers.unsqueeze(0)  # (1, n_bins)
#
#         # Compute circular distance
#         diff = phase_expanded - centers_expanded
#         # Wrap to [-pi, pi]
#         period = 2 * np.pi
#         diff = diff - period * torch.round(diff / period)
#
#         # Convert distance to similarity (closer = higher)
#         similarity = -torch.abs(diff) / self.temperature
#
#         # Apply softmax to get weights
#         weights = F.softmax(similarity, dim=-1)
#
#         return weights
#
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_ModulationIndex.py
# --------------------------------------------------------------------------------
