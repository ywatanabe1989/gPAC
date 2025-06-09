#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 20:45:00 (ywatanabe)"
# File: ./tests/gpac/utils/compare/test_reporting.py

import pytest
import numpy as np
import io
import sys
from gpac.utils import compare


class TestShapeReporting:
    """Test shape reporting functions."""
    
    def test_print_shape_report_basic(self):
        """Test basic shape report functionality."""
        signal = np.random.RandomState(42).randn(8, 1024)
        pac_gp = np.random.RandomState(42).randn(1, 8, 10, 10)
        pac_tp = np.random.RandomState(42).randn(10, 10, 8)
        
        # Capture output
        output = io.StringIO()
        compare.print_shape_report(signal, pac_gp, pac_tp, file=output)
        
        report = output.getvalue()
        
        # Check required sections
        assert "SHAPE VERIFICATION REPORT" in report
        assert "=" in report  # Header formatting
        assert "Input signal: (8, 1024)" in report
        assert "gPAC: (1, 8, 10, 10)" in report
        assert "TensorPAC: (10, 10, 8)" in report
        assert "batch, channels, n_pha, n_amp" in report
        
        # Check match indicator
        assert ("✅ YES" in report) or ("❌ NO" in report)
    
    def test_print_shape_report_matching_shapes(self):
        """Test shape report with matching shapes."""
        signal = np.random.RandomState(42).randn(1024)
        pac_gp = np.random.RandomState(42).randn(1, 1, 10, 8)
        pac_tp = np.random.RandomState(42).randn(10, 8)  # Matching last 2 dims
        
        output = io.StringIO()
        compare.print_shape_report(signal, pac_gp, pac_tp, file=output)
        
        report = output.getvalue()
        assert "✅ YES" in report
    
    def test_print_shape_report_non_matching_shapes(self):
        """Test shape report with non-matching shapes."""
        signal = np.random.RandomState(42).randn(1024)
        pac_gp = np.random.RandomState(42).randn(1, 1, 10, 8)
        pac_tp = np.random.RandomState(42).randn(8, 12)  # Non-matching
        
        output = io.StringIO()
        compare.print_shape_report(signal, pac_gp, pac_tp, file=output)
        
        report = output.getvalue()
        assert "❌ NO" in report
    
    def test_print_shape_report_stdout_default(self, capsys):
        """Test that shape report prints to stdout by default."""
        signal = np.random.RandomState(42).randn(1024)
        pac_gp = np.random.RandomState(42).randn(1, 1, 5, 5)
        pac_tp = np.random.RandomState(42).randn(5, 5)
        
        compare.print_shape_report(signal, pac_gp, pac_tp)
        
        captured = capsys.readouterr()
        assert "SHAPE VERIFICATION REPORT" in captured.out
        assert "Input signal: (1024,)" in captured.out


class TestBandReporting:
    """Test band reporting functions."""
    
    def test_print_band_report_basic(self):
        """Test basic band report functionality."""
        pha_bands = np.array([[2, 4], [4, 8], [8, 16]])
        amp_bands = np.array([[30, 50], [50, 80], [80, 120]])
        
        output = io.StringIO()
        compare.print_band_report(pha_bands, amp_bands, file=output)
        
        report = output.getvalue()
        
        # Check main sections
        assert "Band Configuration:" in report
        assert "Phase bands:" in report
        assert "Amplitude bands:" in report
        
        # Check phase band details
        assert "Band 0: 2.0 - 4.0 Hz (center: 3.0 Hz)" in report
        assert "Band 1: 4.0 - 8.0 Hz (center: 6.0 Hz)" in report
        assert "Band 2: 8.0 - 16.0 Hz (center: 12.0 Hz)" in report
        
        # Check amplitude band details
        assert "Band 0: 30.0 - 50.0 Hz (center: 40.0 Hz)" in report
        assert "Band 1: 50.0 - 80.0 Hz (center: 65.0 Hz)" in report
        assert "Band 2: 80.0 - 120.0 Hz (center: 100.0 Hz)" in report
    
    def test_print_band_report_single_bands(self):
        """Test band report with single band arrays."""
        pha_bands = np.array([[5, 10]])
        amp_bands = np.array([[40, 60]])
        
        output = io.StringIO()
        compare.print_band_report(pha_bands, amp_bands, file=output)
        
        report = output.getvalue()
        assert "Band 0: 5.0 - 10.0 Hz (center: 7.5 Hz)" in report
        assert "Band 0: 40.0 - 60.0 Hz (center: 50.0 Hz)" in report
    
    def test_print_band_report_many_bands(self):
        """Test band report with many bands."""
        # Create 20 phase bands and 15 amplitude bands
        pha_bands = np.array([[i, i+2] for i in range(1, 41, 2)])  # 20 bands
        amp_bands = np.array([[30+i*5, 35+i*5] for i in range(15)])  # 15 bands
        
        output = io.StringIO()
        compare.print_band_report(pha_bands, amp_bands, file=output)
        
        report = output.getvalue()
        
        # Check first and last bands
        assert "Band 0: 1.0 - 3.0 Hz" in report
        assert "Band 19: 39.0 - 41.0 Hz" in report
        assert "Band 0: 30.0 - 35.0 Hz" in report
        assert "Band 14: 100.0 - 105.0 Hz" in report
    
    def test_print_band_report_floating_point_precision(self):
        """Test band report with floating point values."""
        pha_bands = np.array([[2.123, 4.567], [4.567, 8.901]])
        amp_bands = np.array([[30.333, 50.666]])
        
        output = io.StringIO()
        compare.print_band_report(pha_bands, amp_bands, file=output)
        
        report = output.getvalue()
        
        # Check that values are properly formatted to 1 decimal place
        assert "2.1 - 4.6 Hz (center: 3.3 Hz)" in report
        # The center calculation might round slightly differently
        assert ("4.6 - 8.9 Hz (center: 6.8 Hz)" in report or 
                "4.6 - 8.9 Hz (center: 6.7 Hz)" in report)
        assert "30.3 - 50.7 Hz (center: 40.5 Hz)" in report
    
    def test_print_band_report_stdout_default(self, capsys):
        """Test that band report prints to stdout by default."""
        pha_bands = np.array([[2, 4], [4, 8]])
        amp_bands = np.array([[30, 50]])
        
        compare.print_band_report(pha_bands, amp_bands)
        
        captured = capsys.readouterr()
        assert "Band Configuration:" in captured.out
        assert "Phase bands:" in captured.out


class TestComparisonSummary:
    """Test comparison summary reporting."""
    
    def test_print_comparison_summary_basic(self):
        """Test basic comparison summary functionality."""
        corr_metrics = {
            'pearson_r': 0.8567,
            'spearman_r': 0.8234,
            'pearson_p': 0.001,
            'spearman_p': 0.002
        }
        error_metrics = {
            'mae': 0.05432,
            'mse': 0.01234,
            'rmse': 0.11111,
            'max_error': 0.23456
        }
        
        output = io.StringIO()
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        summary = output.getvalue()
        
        # Check main sections
        assert "Comparison Metrics:" in summary
        
        # Check correlation metrics (4 decimal places)
        assert "Pearson correlation: 0.8567" in summary
        assert "Spearman correlation: 0.8234" in summary
        
        # Check error metrics (6 decimal places)
        assert "MAE: 0.054320" in summary
        assert "MSE: 0.012340" in summary
        assert "RMSE: 0.111110" in summary
        assert "Max absolute error: 0.234560" in summary
    
    def test_print_comparison_summary_perfect_correlation(self):
        """Test summary with perfect correlation."""
        corr_metrics = {
            'pearson_r': 1.0,
            'spearman_r': 1.0,
        }
        error_metrics = {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'max_error': 0.0
        }
        
        output = io.StringIO()
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        summary = output.getvalue()
        assert "Pearson correlation: 1.0000" in summary
        assert "MAE: 0.000000" in summary
        assert "RMSE: 0.000000" in summary
    
    def test_print_comparison_summary_negative_correlation(self):
        """Test summary with negative correlation."""
        corr_metrics = {
            'pearson_r': -0.7543,
            'spearman_r': -0.6789,
        }
        error_metrics = {
            'mae': 1.23456,
            'mse': 2.34567,
            'rmse': 1.53146,
            'max_error': 3.45678
        }
        
        output = io.StringIO()
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        summary = output.getvalue()
        assert "Pearson correlation: -0.7543" in summary
        assert "Spearman correlation: -0.6789" in summary
    
    def test_print_comparison_summary_missing_max_error(self):
        """Test summary when max_error is missing from error_metrics."""
        corr_metrics = {
            'pearson_r': 0.5,
            'spearman_r': 0.4,
        }
        error_metrics = {
            'mae': 0.1,
            'mse': 0.01,
            'rmse': 0.1,
            # max_error intentionally missing
        }
        
        output = io.StringIO()
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        summary = output.getvalue()
        # Should show 0.000000 for missing max_error
        assert "Max absolute error: 0.000000" in summary
    
    def test_print_comparison_summary_stdout_default(self, capsys):
        """Test that comparison summary prints to stdout by default."""
        corr_metrics = {'pearson_r': 0.8, 'spearman_r': 0.7}
        error_metrics = {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1, 'max_error': 0.2}
        
        compare.print_comparison_summary(corr_metrics, error_metrics)
        
        captured = capsys.readouterr()
        assert "Comparison Metrics:" in captured.out
        assert "Pearson correlation: 0.8000" in captured.out


class TestIntegratedReporting:
    """Test integrated reporting scenarios."""
    
    def test_complete_reporting_workflow(self):
        """Test complete reporting workflow with all functions."""
        # Create test data
        signal = np.random.RandomState(42).randn(8, 1024)
        pac_gp = np.random.RandomState(42).randn(1, 8, 10, 12)
        pac_tp = np.random.RandomState(42).randn(10, 12, 8)
        
        pha_bands = np.array([[2, 4], [4, 8], [8, 16], [16, 20]])
        amp_bands = np.array([[30, 50], [50, 80], [80, 120]])
        
        corr_metrics = {'pearson_r': 0.65, 'spearman_r': 0.62}
        error_metrics = {'mae': 0.15, 'mse': 0.05, 'rmse': 0.22, 'max_error': 0.8}
        
        # Capture all output
        output = io.StringIO()
        
        # Run all reporting functions
        compare.print_shape_report(signal, pac_gp, pac_tp, file=output)
        compare.print_band_report(pha_bands, amp_bands, file=output)
        compare.print_comparison_summary(corr_metrics, error_metrics, file=output)
        
        report = output.getvalue()
        
        # Verify all sections are present
        assert "SHAPE VERIFICATION REPORT" in report
        assert "Band Configuration:" in report
        assert "Comparison Metrics:" in report
        
        # Verify specific details
        assert "(8, 1024)" in report  # Input shape
        assert "(1, 8, 10, 12)" in report  # gPAC shape
        assert "2.0 - 4.0 Hz" in report  # Phase band
        assert "30.0 - 50.0 Hz" in report  # Amplitude band
        assert "0.6500" in report  # Pearson correlation
        assert "0.150000" in report  # MAE
    
    def test_file_output_consistency(self):
        """Test that file output is consistent across functions."""
        signal = np.random.RandomState(42).randn(1024)
        pac_gp = np.random.RandomState(42).randn(1, 1, 5, 5)
        pac_tp = np.random.RandomState(42).randn(5, 5)
        
        pha_bands = np.array([[2, 4]])
        amp_bands = np.array([[30, 50]])
        
        corr_metrics = {'pearson_r': 0.5, 'spearman_r': 0.4}
        error_metrics = {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1, 'max_error': 0.2}
        
        # Test with file object
        with io.StringIO() as f:
            compare.print_shape_report(signal, pac_gp, pac_tp, file=f)
            compare.print_band_report(pha_bands, amp_bands, file=f)
            compare.print_comparison_summary(corr_metrics, error_metrics, file=f)
            
            file_output = f.getvalue()
        
        # Test with default (stdout capture)
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        compare.print_shape_report(signal, pac_gp, pac_tp)
        compare.print_band_report(pha_bands, amp_bands)
        compare.print_comparison_summary(corr_metrics, error_metrics)
        
        stdout_output = sys.stdout.getvalue()
        sys.stdout = original_stdout
        
        # Outputs should be identical
        assert file_output == stdout_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF