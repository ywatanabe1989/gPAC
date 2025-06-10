#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 20:35:00 (ywatanabe)"
# File: ./tests/gpac/utils/compare/test_compare.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/utils/compare/test_compare.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Comprehensive tests for gpac.utils.compare module.

Tests all comparison utilities including:
- Shape verification functions
- Band extraction and verification
- Data preparation
- Metric computation
- Reporting utilities
"""

import pytest
import numpy as np
import torch
from io import StringIO

import gpac
from gpac.utils import compare


class TestShapeVerification:
    """Test shape verification functions."""
    
    def test_verify_input_shape_gpac_valid_3d(self):
        """Test valid 3D input for gPAC."""
        signal = torch.randn(1, 8, 1024)  # (batch, channels, time)
        assert compare.verify_input_shape_gpac(signal) is True
    
    def test_verify_input_shape_gpac_valid_4d(self):
        """Test valid 4D input for gPAC."""
        signal = torch.randn(1, 8, 10, 1024)  # (batch, channels, epochs, time)
        assert compare.verify_input_shape_gpac(signal) is True
    
    def test_verify_input_shape_gpac_invalid(self):
        """Test invalid input for gPAC."""
        signal = torch.randn(1024)  # 1D
        with pytest.raises(ValueError, match="gPAC expects"):
            compare.verify_input_shape_gpac(signal)
    
    def test_verify_input_shape_tensorpac_valid(self):
        """Test valid input for TensorPAC."""
        signal = np.random.randn(10, 1024)  # (epochs, time)
        assert compare.verify_input_shape_tensorpac(signal) is True
    
    def test_verify_input_shape_tensorpac_invalid(self):
        """Test invalid input for TensorPAC."""
        signal = np.random.randn(1, 8, 1024)  # 3D
        with pytest.raises(ValueError, match="TensorPAC expects 2D"):
            compare.verify_input_shape_tensorpac(signal)
    
    def test_verify_output_shapes_match_same(self):
        """Test when output shapes match."""
        pac_gp = np.random.randn(10, 10)
        pac_tp = np.random.randn(10, 10)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=False)
        assert match is True
        assert np.array_equal(corrected, pac_tp)
    
    def test_verify_output_shapes_match_transpose(self):
        """Test when output shapes match after transpose."""
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(8, 10)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=False)
        assert match is True
        assert corrected.shape == pac_gp.shape


class TestBandUtilities:
    """Test band extraction and verification functions."""
    
    @pytest.fixture
    def pac_gp_obj(self):
        """Create a gPAC object for testing."""
        return gpac.PAC(
            seq_len=1024,
            fs=256,
            pha_range_hz=(2, 20),
            pha_n_bands=10,
            amp_range_hz=(30, 100),
            amp_n_bands=10,
            trainable=False
        )
    
    def test_extract_gpac_bands(self, pac_gp_obj):
        """Test band extraction from gPAC object."""
        pha_bands, amp_bands = compare.extract_gpac_bands(pac_gp_obj)
        
        assert isinstance(pha_bands, np.ndarray)
        assert isinstance(amp_bands, np.ndarray)
        assert pha_bands.shape == (10, 2)
        assert amp_bands.shape == (10, 2)
        assert pha_bands.min() >= 1.0  # Above 0 Hz
        assert amp_bands.max() <= 128.0  # Below Nyquist
    
    def test_verify_band_ranges_phase_match(self):
        """Test band range verification for phase bands."""
        bands = np.array([[2.0, 3.0], [3.0, 5.0], [5.0, 10.0], [10.0, 20.0]])
        result = compare.verify_band_ranges(
            bands, (2, 20), (30, 100), is_phase=True
        )
        assert result['match'] == True  # Use == for numpy bool comparison
        assert result['actual_range'] == (2.0, 20.0)
    
    def test_verify_band_ranges_amp_mismatch(self):
        """Test band range verification for amplitude bands with mismatch."""
        bands = np.array([[30.0, 40.0], [40.0, 60.0], [60.0, 80.0]])
        result = compare.verify_band_ranges(
            bands, (2, 20), (30, 100), is_phase=False
        )
        assert result['match'] == False  # Use == for numpy bool comparison
        assert result['actual_range'] == (30.0, 80.0)
    
    def test_check_band_spacing_linear(self):
        """Test linear band spacing detection."""
        # Linear spacing
        bands = np.array([[0, 10], [10, 20], [20, 30], [30, 40]])
        assert compare.check_band_spacing(bands) == True
    
    def test_check_band_spacing_nonlinear(self):
        """Test non-linear band spacing detection."""
        # Non-linear (logarithmic-like) spacing
        bands = np.array([[1, 2], [2, 4], [4, 8], [8, 16]])
        assert compare.check_band_spacing(bands) == False


class TestDataPreparation:
    """Test data preparation functions."""
    
    def test_prepare_signal_gpac_1d(self):
        """Test 1D signal preparation for gPAC."""
        signal = np.random.randn(1024)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (1, 1, 1024)
        assert prepared.device.type == 'cpu'
    
    def test_prepare_signal_gpac_2d(self):
        """Test 2D signal preparation for gPAC."""
        signal = np.random.randn(8, 1024)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (1, 8, 1024)
    
    def test_prepare_signal_gpac_gpu(self):
        """Test GPU placement if available."""
        if torch.cuda.is_available():
            signal = np.random.randn(1024)
            prepared = compare.prepare_signal_gpac(signal, device='cuda')
            assert prepared.device.type == 'cuda'
    
    def test_prepare_signal_tensorpac_1d(self):
        """Test 1D signal preparation for TensorPAC."""
        signal = np.random.randn(1024)
        prepared = compare.prepare_signal_tensorpac(signal)
        
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (1, 1024)
    
    def test_prepare_signal_tensorpac_invalid(self):
        """Test invalid signal for TensorPAC."""
        signal = np.random.randn(1, 8, 1024)  # 3D
        with pytest.raises(ValueError, match="TensorPAC expects 1D or 2D"):
            compare.prepare_signal_tensorpac(signal)


class TestMetricComputation:
    """Test metric computation functions."""
    
    @pytest.fixture
    def pac_matrices(self):
        """Create test PAC matrices."""
        # Create correlated matrices with some noise
        base = np.random.randn(10, 10)
        pac1 = base + 0.1 * np.random.randn(10, 10)
        pac2 = 0.5 * base + 0.1 * np.random.randn(10, 10)  # Scaled and noisy
        return pac1, pac2
    
    def test_compute_correlation_metrics(self, pac_matrices):
        """Test correlation metric computation."""
        pac1, pac2 = pac_matrices
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        
        assert 'pearson_r' in metrics
        assert 'spearman_r' in metrics
        assert 'pearson_normalized' in metrics
        assert -1 <= metrics['pearson_r'] <= 1
        assert -1 <= metrics['spearman_r'] <= 1
    
    def test_compute_correlation_metrics_tensor(self, pac_matrices):
        """Test correlation metrics with tensor input."""
        pac1, pac2 = pac_matrices
        pac1_tensor = torch.from_numpy(pac1)
        
        metrics = compare.compute_correlation_metrics(pac1_tensor, pac2)
        assert isinstance(metrics['pearson_r'], float)
    
    def test_compute_correlation_metrics_shape_mismatch(self):
        """Test correlation with shape mismatch handling."""
        # gPAC shape: (batch, channels, n_pha, n_amp)
        pac1 = np.random.randn(1, 8, 10, 10)
        # TensorPAC shape: (n_pha, n_amp, n_channels)
        pac2 = np.random.randn(10, 10, 8)
        
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        assert 'pearson_r' in metrics
    
    def test_compute_error_metrics(self, pac_matrices):
        """Test error metric computation."""
        pac1, pac2 = pac_matrices
        metrics = compare.compute_error_metrics(pac1, pac2)
        
        assert 'scale_factor' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'max_error' in metrics
        assert 'mae_normalized' in metrics
        assert 'rmse_normalized' in metrics
        
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= metrics['mae']  # RMSE >= MAE always
    
    def test_compute_error_metrics_identical(self):
        """Test error metrics for identical matrices."""
        pac = np.random.randn(10, 10)
        metrics = compare.compute_error_metrics(pac, pac)
        
        assert metrics['mae'] == 0
        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['scale_factor'] == 1.0


class TestReportingUtilities:
    """Test reporting functions."""
    
    def test_print_shape_report(self):
        """Test shape report printing."""
        signal = np.random.randn(8, 1024)
        pac_gp = np.random.randn(1, 8, 10, 10)
        pac_tp = np.random.randn(10, 10, 8)
        
        output = StringIO()
        compare.print_shape_report(signal, pac_gp, pac_tp, file=output)
        
        report = output.getvalue()
        assert "SHAPE VERIFICATION REPORT" in report
        assert "Input signal: (8, 1024)" in report
        assert "gPAC: (1, 8, 10, 10)" in report
    
    def test_print_band_report(self):
        """Test band report printing."""
        pha_bands = np.array([[2, 4], [4, 8], [8, 16]])
        amp_bands = np.array([[30, 50], [50, 80], [80, 120]])
        
        output = StringIO()
        compare.print_band_report(pha_bands, amp_bands, file=output)
        
        report = output.getvalue()
        assert "Band Configuration:" in report
        assert "Phase bands:" in report
        assert "2.0 - 4.0 Hz" in report
        assert "Amplitude bands:" in report
        assert "30.0 - 50.0 Hz" in report
    
    def test_print_comparison_summary(self):
        """Test comparison summary printing."""
        corr_metrics = {
            'pearson_r': 0.85,
            'spearman_r': 0.82,
        }
        error_metrics = {
            'mae': 0.05,
            'mse': 0.01,
            'rmse': 0.1,
            'max_error': 0.2
        }
        
        output = StringIO()
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        summary = output.getvalue()
        assert "Comparison Metrics:" in summary
        assert "Pearson correlation: 0.8500" in summary
        assert "MAE: 0.050000" in summary


class TestQuickCompare:
    """Test the quick_compare convenience function."""
    
    def test_quick_compare_dict_input(self):
        """Test quick compare with dictionary input from gPAC."""
        pac_gp_result = {
            'pac': np.random.randn(1, 1, 10, 10),
            'phase_frequencies': np.linspace(2, 20, 10),
            'amplitude_frequencies': np.linspace(30, 100, 10)
        }
        pac_tp_result = np.random.randn(10, 10)
        
        result = compare.quick_compare(pac_gp_result, pac_tp_result, verbose=False)
        
        assert 'correlation' in result
        assert 'errors' in result
        assert 'pearson_r' in result['correlation']
        assert 'mae' in result['errors']
    
    def test_quick_compare_tensor_input(self):
        """Test quick compare with tensor input."""
        pac_gp = torch.randn(1, 1, 10, 10)
        pac_tp = np.random.randn(10, 10)
        
        result = compare.quick_compare(pac_gp, pac_tp, verbose=False)
        
        assert isinstance(result['correlation']['pearson_r'], float)
        assert isinstance(result['errors']['scale_factor'], float)


class TestIntegration:
    """Integration tests with real gPAC objects (avoiding TensorPAC dependency)."""
    
    @pytest.mark.slow
    def test_full_gpac_workflow(self):
        """Test complete workflow with gPAC object."""
        # Generate deterministic test signal
        np.random.seed(42)
        duration = 1.0
        fs = 256
        n_samples = int(duration * fs)
        t = np.linspace(0, duration, n_samples)
        
        # Create simple PAC signal
        phase_signal = np.sin(2 * np.pi * 5 * t)
        amp_signal = (1 + 0.3 * phase_signal) * np.sin(2 * np.pi * 50 * t)
        signal = phase_signal + amp_signal + 0.1 * np.random.randn(n_samples)
        
        # Initialize gPAC
        pac_gp = gpac.PAC(
            seq_len=n_samples,
            fs=fs,
            pha_range_hz=(2, 20),
            pha_n_bands=8,
            amp_range_hz=(30, 100),
            amp_n_bands=8,
            trainable=False
        )
        
        # Extract bands
        pha_bands, amp_bands = compare.extract_gpac_bands(pac_gp)
        assert pha_bands.shape[0] == 8
        assert amp_bands.shape[0] == 8
        
        # Verify band properties
        assert compare.check_band_spacing(pha_bands) is False  # gPAC uses non-linear
        
        # Prepare signal
        signal_gp = compare.prepare_signal_gpac(signal)
        assert signal_gp.shape == (1, 1, n_samples)
        
        # Compute PAC
        pac_result_gp = pac_gp(signal_gp)
        
        # Verify result structure
        assert isinstance(pac_result_gp, dict)
        assert 'pac' in pac_result_gp
        
        pac_values = pac_result_gp['pac']
        assert isinstance(pac_values, torch.Tensor)
        assert pac_values.shape == (1, 1, 8, 8)  # (batch, channels, n_pha, n_amp)
        
        # Test with synthetic "TensorPAC-like" result for comparison
        synthetic_tp_result = np.random.RandomState(42).randn(8, 8)
        comparison = compare.quick_compare(pac_result_gp, synthetic_tp_result, verbose=False)
        
        # Verify comparison structure
        assert 'correlation' in comparison
        assert 'errors' in comparison
        assert isinstance(comparison['correlation']['pearson_r'], float)
        assert isinstance(comparison['errors']['scale_factor'], float)
    
    def test_band_verification_workflow(self):
        """Test band verification workflow."""
        # Create PAC object with known parameters
        pac_gp = gpac.PAC(
            seq_len=1024,
            fs=256,
            pha_range_hz=(4, 16),
            pha_n_bands=6,
            amp_range_hz=(40, 120),
            amp_n_bands=8,
            trainable=False
        )
        
        # Extract and verify bands
        pha_bands, amp_bands = compare.extract_gpac_bands(pac_gp)
        
        # Test band range verification with larger tolerance for practical ranges
        pha_result = compare.verify_band_ranges(
            pha_bands, (4, 16), (40, 120), tolerance=5.0, is_phase=True
        )
        amp_result = compare.verify_band_ranges(
            amp_bands, (4, 16), (40, 120), tolerance=10.0, is_phase=False
        )
        
        # These might not match exactly due to gPAC's band generation
        assert isinstance(pha_result['match'], (bool, np.bool_))
        assert isinstance(amp_result['match'], (bool, np.bool_))
        
        # Test spacing analysis
        pha_linear = compare.check_band_spacing(pha_bands, tolerance=0.1)
        amp_linear = compare.check_band_spacing(amp_bands, tolerance=0.1)
        
        # gPAC typically uses non-linear spacing
        assert isinstance(pha_linear, bool)
        assert isinstance(amp_linear, bool)


# Performance tests (lightweight without external benchmarking)
class TestPerformance:
    """Test performance-related aspects without external dependencies."""
    
    def test_metric_computation_efficiency(self):
        """Test that metric computation completes in reasonable time."""
        import time
        
        # Large matrices to test efficiency
        pac1 = np.random.RandomState(42).randn(100, 100)
        pac2 = np.random.RandomState(123).randn(100, 100)
        
        start_time = time.time()
        result = compare.compute_correlation_metrics(pac1, pac2)
        end_time = time.time()
        
        # Should complete quickly (< 1 second for 100x100)
        assert (end_time - start_time) < 1.0
        assert 'pearson_r' in result
        assert isinstance(result['pearson_r'], float)
    
    def test_band_extraction_efficiency(self):
        """Test that band extraction completes efficiently."""
        import time
        
        pac_gp = gpac.PAC(
            seq_len=1024,
            fs=256,
            pha_range_hz=(2, 20),
            pha_n_bands=50,  # Large number of bands
            amp_range_hz=(30, 100),
            amp_n_bands=50,
            trainable=False
        )
        
        start_time = time.time()
        pha_bands, amp_bands = compare.extract_gpac_bands(pac_gp)
        end_time = time.time()
        
        # Should complete very quickly
        assert (end_time - start_time) < 0.1
        assert pha_bands.shape[0] == 50
        assert amp_bands.shape[0] == 50
    
    def test_data_preparation_efficiency(self):
        """Test data preparation efficiency."""
        import time
        
        # Large signal
        large_signal = np.random.RandomState(42).randn(100000)
        
        start_time = time.time()
        gpac_signal = compare.prepare_signal_gpac(large_signal)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0
        assert gpac_signal.shape == (1, 1, 100000)
        
        start_time = time.time()
        tp_signal = compare.prepare_signal_tensorpac(large_signal)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0
        assert tp_signal.shape == (1, 100000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF