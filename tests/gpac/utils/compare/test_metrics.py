#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 20:43:00 (ywatanabe)"
# File: ./tests/gpac/utils/compare/test_metrics.py

import pytest
import numpy as np
import torch
from gpac.utils import compare


class TestCorrelationMetrics:
    """Test correlation metric computation functions."""
    
    @pytest.fixture
    def identical_matrices(self):
        """Create identical matrices for testing."""
        base = np.random.RandomState(42).randn(10, 10)
        return base, base.copy()
    
    @pytest.fixture
    def correlated_matrices(self):
        """Create correlated matrices with known relationship."""
        np.random.seed(42)
        base = np.random.randn(10, 10)
        pac1 = base + 0.1 * np.random.randn(10, 10)
        pac2 = 0.8 * base + 0.2 * np.random.randn(10, 10)
        return pac1, pac2
    
    @pytest.fixture
    def uncorrelated_matrices(self):
        """Create uncorrelated matrices."""
        np.random.seed(42)
        pac1 = np.random.randn(10, 10)
        np.random.seed(123)
        pac2 = np.random.randn(10, 10)
        return pac1, pac2
    
    def test_compute_correlation_metrics_identical(self, identical_matrices):
        """Test correlation metrics for identical matrices."""
        pac1, pac2 = identical_matrices
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        
        assert 'pearson_r' in metrics
        assert 'pearson_p' in metrics
        assert 'spearman_r' in metrics
        assert 'spearman_p' in metrics
        assert 'pearson_normalized' in metrics
        
        # Perfect correlation
        assert abs(metrics['pearson_r'] - 1.0) < 1e-10
        assert abs(metrics['spearman_r'] - 1.0) < 1e-10
        assert abs(metrics['pearson_normalized'] - 1.0) < 1e-10
        assert metrics['pearson_p'] < 0.001
    
    def test_compute_correlation_metrics_correlated(self, correlated_matrices):
        """Test correlation metrics for correlated matrices."""
        pac1, pac2 = correlated_matrices
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        
        # Should be positively correlated
        assert metrics['pearson_r'] > 0.5
        assert metrics['spearman_r'] > 0.5
        assert metrics['pearson_normalized'] > 0.5
        assert metrics['pearson_p'] < 0.001
    
    def test_compute_correlation_metrics_uncorrelated(self, uncorrelated_matrices):
        """Test correlation metrics for uncorrelated matrices."""
        pac1, pac2 = uncorrelated_matrices
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        
        # Should be weakly correlated
        assert abs(metrics['pearson_r']) < 0.5
        assert abs(metrics['spearman_r']) < 0.5
        assert abs(metrics['pearson_normalized']) < 0.5
    
    def test_compute_correlation_metrics_tensor_input(self, correlated_matrices):
        """Test correlation metrics with PyTorch tensor input."""
        pac1, pac2 = correlated_matrices
        pac1_tensor = torch.from_numpy(pac1).float()
        
        metrics = compare.compute_correlation_metrics(pac1_tensor, pac2)
        
        assert isinstance(metrics['pearson_r'], float)
        assert isinstance(metrics['pearson_p'], float)
        assert -1 <= metrics['pearson_r'] <= 1
        assert 0 <= metrics['pearson_p'] <= 1
    
    def test_compute_correlation_metrics_shape_mismatch_4d_3d(self):
        """Test correlation with 4D vs 3D shape mismatch."""
        np.random.seed(42)
        # gPAC shape: (batch, channels, n_pha, n_amp)
        pac1 = np.random.randn(2, 3, 8, 10)
        # TensorPAC shape: (n_pha, n_amp, n_channels)
        pac2 = np.random.randn(8, 10, 3)
        
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        assert 'pearson_r' in metrics
        assert isinstance(metrics['pearson_r'], float)
    
    def test_compute_correlation_metrics_shape_mismatch_4d_2d(self):
        """Test correlation with 4D vs 2D shape mismatch."""
        np.random.seed(42)
        # gPAC shape: (batch, channels, n_pha, n_amp)
        pac1 = np.random.randn(2, 3, 8, 10)
        # TensorPAC shape: (n_pha, n_amp)
        pac2 = np.random.randn(8, 10)
        
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        assert 'pearson_r' in metrics
        assert isinstance(metrics['pearson_r'], float)
    
    def test_compute_correlation_metrics_transpose_handling(self):
        """Test correlation with transpose requirement."""
        np.random.seed(42)
        pac1 = np.random.randn(8, 10)
        pac2 = np.random.randn(10, 8)  # Transposed
        
        metrics = compare.compute_correlation_metrics(pac1, pac2)
        assert 'pearson_r' in metrics
        assert isinstance(metrics['pearson_r'], float)
    
    def test_compute_correlation_metrics_incompatible_shapes(self):
        """Test correlation with incompatible shapes."""
        pac1 = np.random.randn(8, 10)
        pac2 = np.random.randn(5, 7)  # Completely different
        
        with pytest.raises(ValueError, match="Cannot reconcile shapes"):
            compare.compute_correlation_metrics(pac1, pac2)


class TestErrorMetrics:
    """Test error metric computation functions."""
    
    @pytest.fixture
    def scaled_matrices(self):
        """Create matrices with known scale difference."""
        np.random.seed(42)
        base = np.random.randn(10, 10)
        pac1 = base
        pac2 = 2.0 * base  # Exactly 2x scale
        return pac1, pac2
    
    def test_compute_error_metrics_identical(self):
        """Test error metrics for identical matrices."""
        np.random.seed(42)
        pac = np.random.randn(10, 10)
        metrics = compare.compute_error_metrics(pac, pac)
        
        assert 'scale_factor' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'max_error' in metrics
        assert 'mae_normalized' in metrics
        assert 'rmse_normalized' in metrics
        
        # Perfect match
        assert metrics['scale_factor'] == 1.0
        assert metrics['mae'] == 0.0
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['max_error'] == 0.0
        assert metrics['mae_normalized'] == 0.0
        assert metrics['rmse_normalized'] == 0.0
    
    def test_compute_error_metrics_scaled(self, scaled_matrices):
        """Test error metrics for scaled matrices."""
        pac1, pac2 = scaled_matrices
        metrics = compare.compute_error_metrics(pac1, pac2)
        
        # Should detect 2x scale
        assert abs(metrics['scale_factor'] - 2.0) < 1e-10
        
        # Raw errors should be non-zero
        assert metrics['mae'] > 0
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
        assert metrics['max_error'] > 0
        
        # Normalized errors should be 0 (perfect after normalization)
        assert abs(metrics['mae_normalized']) < 1e-10
        assert abs(metrics['rmse_normalized']) < 1e-10
    
    def test_compute_error_metrics_properties(self):
        """Test mathematical properties of error metrics."""
        np.random.seed(42)
        pac1 = np.random.randn(10, 10)
        pac2 = pac1 + 0.1 * np.random.randn(10, 10)  # Add noise
        
        metrics = compare.compute_error_metrics(pac1, pac2)
        
        # RMSE should be >= MAE
        assert metrics['rmse'] >= metrics['mae']
        
        # All error metrics should be positive
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['max_error'] >= 0
        assert metrics['mae_normalized'] >= 0
        assert metrics['rmse_normalized'] >= 0
        
        # MSE should equal RMSE squared
        assert abs(metrics['mse'] - metrics['rmse']**2) < 1e-10
    
    def test_compute_error_metrics_tensor_input(self):
        """Test error metrics with PyTorch tensor input."""
        np.random.seed(42)
        pac1_np = np.random.randn(8, 10)
        pac2_np = pac1_np + 0.1 * np.random.randn(8, 10)
        
        pac1_tensor = torch.from_numpy(pac1_np).float()
        
        metrics = compare.compute_error_metrics(pac1_tensor, pac2_np)
        
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['scale_factor'], float)
        assert metrics['mae'] > 0
    
    def test_compute_error_metrics_zero_division_handling(self):
        """Test handling of zero division in scale factor."""
        pac1 = np.zeros((5, 5))
        pac2 = np.random.randn(5, 5)
        
        metrics = compare.compute_error_metrics(pac1, pac2)
        
        # Scale factor should be inf when pac1.max() == 0
        assert metrics['scale_factor'] == np.inf
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
    
    def test_compute_error_metrics_shape_handling(self):
        """Test error metrics with different input shapes."""
        np.random.seed(42)
        # Test 4D vs 2D case
        pac1 = np.random.randn(1, 1, 8, 10)
        pac2 = np.random.randn(8, 10)
        
        metrics = compare.compute_error_metrics(pac1, pac2)
        assert 'mae' in metrics
        assert isinstance(metrics['mae'], float)
        
        # Test 4D vs 3D case
        pac1 = np.random.randn(2, 3, 8, 10)
        pac2 = np.random.randn(8, 10, 3)
        
        metrics = compare.compute_error_metrics(pac1, pac2)
        assert 'mae' in metrics
        assert isinstance(metrics['mae'], float)


class TestQuickCompare:
    """Test the quick_compare convenience function."""
    
    def test_quick_compare_dict_input(self):
        """Test quick_compare with dictionary input from gPAC."""
        np.random.seed(42)
        pac_data = np.random.randn(1, 1, 10, 10)
        pac_gp_result = {
            'pac': pac_data,
            'phase_frequencies': np.linspace(2, 20, 10),
            'amplitude_frequencies': np.linspace(30, 100, 10)
        }
        pac_tp_result = np.random.randn(10, 10)
        
        result = compare.quick_compare(pac_gp_result, pac_tp_result, verbose=False)
        
        assert 'correlation' in result
        assert 'errors' in result
        assert 'pearson_r' in result['correlation']
        assert 'mae' in result['errors']
        assert isinstance(result['correlation']['pearson_r'], float)
        assert isinstance(result['errors']['scale_factor'], float)
    
    def test_quick_compare_tensor_input(self):
        """Test quick_compare with tensor input."""
        np.random.seed(42)
        pac_gp = torch.randn(1, 1, 10, 10)
        pac_tp = np.random.randn(10, 10)
        
        result = compare.quick_compare(pac_gp, pac_tp, verbose=False)
        
        assert isinstance(result['correlation']['pearson_r'], float)
        assert isinstance(result['errors']['scale_factor'], float)
        assert 'correlation' in result
        assert 'errors' in result
    
    def test_quick_compare_direct_array_input(self):
        """Test quick_compare with direct array input."""
        np.random.seed(42)
        pac_gp = np.random.randn(10, 10)
        pac_tp = np.random.randn(10, 10)
        
        result = compare.quick_compare(pac_gp, pac_tp, verbose=False)
        
        assert 'correlation' in result
        assert 'errors' in result
        assert isinstance(result['correlation']['pearson_r'], float)
    
    def test_quick_compare_verbose_output(self, capsys):
        """Test quick_compare verbose output."""
        np.random.seed(42)
        pac_gp = np.random.randn(10, 10)
        pac_tp = 2.0 * pac_gp  # Perfect correlation, 2x scale
        
        result = compare.quick_compare(pac_gp, pac_tp, verbose=True)
        
        captured = capsys.readouterr()
        assert "Quick Comparison Results:" in captured.out
        assert "Correlation:" in captured.out
        assert "Scale factor:" in captured.out
        assert "Normalized MAE:" in captured.out
        
        # Should show high correlation
        assert result['correlation']['pearson_r'] > 0.99
        assert abs(result['errors']['scale_factor'] - 2.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF