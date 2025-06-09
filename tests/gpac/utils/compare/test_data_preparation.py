#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 20:44:30 (ywatanabe)"
# File: ./tests/gpac/utils/compare/test_data_preparation.py

import pytest
import numpy as np
import torch
from gpac.utils import compare


class TestDataPreparation:
    """Test data preparation functions for gPAC and TensorPAC."""
    
    def test_prepare_signal_gpac_1d_basic(self):
        """Test basic 1D signal preparation for gPAC."""
        signal = np.random.RandomState(42).randn(1024)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (1, 1, 1024)  # (batch, channels, time)
        assert prepared.dtype == torch.float32
        assert prepared.device.type == 'cpu'
        
        # Check that data is preserved
        np.testing.assert_array_almost_equal(
            prepared.squeeze().numpy(), signal, decimal=6
        )
    
    def test_prepare_signal_gpac_2d_basic(self):
        """Test basic 2D signal preparation for gPAC."""
        signal = np.random.RandomState(42).randn(8, 1024)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (1, 8, 1024)  # (batch, channels, time)
        assert prepared.dtype == torch.float32
        
        # Check that data is preserved
        np.testing.assert_array_almost_equal(
            prepared.squeeze(0).numpy(), signal, decimal=6
        )
    
    def test_prepare_signal_gpac_3d_passthrough(self):
        """Test 3D signal preparation (should pass through)."""
        signal = np.random.RandomState(42).randn(2, 8, 1024)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (2, 8, 1024)  # Should maintain shape
        assert prepared.dtype == torch.float32
        
        # Check that data is preserved
        np.testing.assert_array_almost_equal(
            prepared.numpy(), signal, decimal=6
        )
    
    def test_prepare_signal_gpac_custom_params(self):
        """Test signal preparation with custom parameters."""
        signal = np.random.RandomState(42).randn(1024)
        
        # Test custom batch and channel sizes (should be ignored for 1D)
        prepared = compare.prepare_signal_gpac(
            signal, batch_size=4, n_channels=16
        )
        
        # Should still be (1, 1, 1024) for 1D input
        assert prepared.shape == (1, 1, 1024)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_signal_gpac_gpu_device(self):
        """Test GPU device placement if CUDA is available."""
        signal = np.random.RandomState(42).randn(1024)
        prepared = compare.prepare_signal_gpac(signal, device='cuda')
        
        assert prepared.device.type == 'cuda'
        assert prepared.shape == (1, 1, 1024)
        
        # Check data preservation
        np.testing.assert_array_almost_equal(
            prepared.cpu().squeeze().numpy(), signal, decimal=6
        )
    
    def test_prepare_signal_gpac_invalid_device(self):
        """Test handling of invalid device specification."""
        signal = np.random.RandomState(42).randn(1024)
        
        # Invalid device should raise RuntimeError
        with pytest.raises(RuntimeError, match="Expected one of"):
            compare.prepare_signal_gpac(signal, device='invalid_device')
    
    def test_prepare_signal_gpac_empty_signal(self):
        """Test preparation of empty signal."""
        signal = np.array([])
        prepared = compare.prepare_signal_gpac(signal)
        
        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape == (1, 1, 0)
    
    def test_prepare_signal_gpac_single_sample(self):
        """Test preparation of single sample signal."""
        signal = np.array([1.5])
        prepared = compare.prepare_signal_gpac(signal)
        
        assert prepared.shape == (1, 1, 1)
        assert prepared.item() == 1.5
    
    def test_prepare_signal_gpac_large_signal(self):
        """Test preparation of large signal."""
        signal = np.random.RandomState(42).randn(100000)
        prepared = compare.prepare_signal_gpac(signal)
        
        assert prepared.shape == (1, 1, 100000)
        assert prepared.dtype == torch.float32
        
        # Spot check some values
        np.testing.assert_array_almost_equal(
            prepared[0, 0, :100].numpy(), signal[:100], decimal=6
        )


class TestTensorPACDataPreparation:
    """Test data preparation functions for TensorPAC."""
    
    def test_prepare_signal_tensorpac_1d_basic(self):
        """Test basic 1D signal preparation for TensorPAC."""
        signal = np.random.RandomState(42).randn(1024)
        prepared = compare.prepare_signal_tensorpac(signal)
        
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (1, 1024)  # (epochs, time)
        assert prepared.dtype == signal.dtype
        
        # Check that data is preserved
        np.testing.assert_array_equal(prepared[0], signal)
    
    def test_prepare_signal_tensorpac_2d_basic(self):
        """Test basic 2D signal preparation for TensorPAC."""
        signal = np.random.RandomState(42).randn(10, 1024)
        prepared = compare.prepare_signal_tensorpac(signal)
        
        assert isinstance(prepared, np.ndarray)
        assert prepared.shape == (10, 1024)  # Should maintain shape
        assert prepared.dtype == signal.dtype
        
        # Check that data is preserved
        np.testing.assert_array_equal(prepared, signal)
    
    def test_prepare_signal_tensorpac_3d_invalid(self):
        """Test that 3D signal raises error for TensorPAC."""
        signal = np.random.RandomState(42).randn(2, 8, 1024)
        
        with pytest.raises(ValueError, match="TensorPAC expects 1D or 2D"):
            compare.prepare_signal_tensorpac(signal)
    
    def test_prepare_signal_tensorpac_4d_invalid(self):
        """Test that 4D signal raises error for TensorPAC."""
        signal = np.random.RandomState(42).randn(2, 8, 10, 1024)
        
        with pytest.raises(ValueError, match="TensorPAC expects 1D or 2D"):
            compare.prepare_signal_tensorpac(signal)
    
    def test_prepare_signal_tensorpac_empty_signal(self):
        """Test preparation of empty signal."""
        signal = np.array([])
        prepared = compare.prepare_signal_tensorpac(signal)
        
        assert prepared.shape == (1, 0)
    
    def test_prepare_signal_tensorpac_single_sample(self):
        """Test preparation of single sample signal."""
        signal = np.array([2.5])
        prepared = compare.prepare_signal_tensorpac(signal)
        
        assert prepared.shape == (1, 1)
        assert prepared[0, 0] == 2.5
    
    def test_prepare_signal_tensorpac_different_dtypes(self):
        """Test preparation with different numpy dtypes."""
        # Test float64
        signal_f64 = np.random.RandomState(42).randn(1024).astype(np.float64)
        prepared_f64 = compare.prepare_signal_tensorpac(signal_f64)
        assert prepared_f64.dtype == np.float64
        
        # Test float32
        signal_f32 = np.random.RandomState(42).randn(1024).astype(np.float32)
        prepared_f32 = compare.prepare_signal_tensorpac(signal_f32)
        assert prepared_f32.dtype == np.float32
        
        # Test int32 (should preserve)
        signal_int = np.random.RandomState(42).randint(-100, 100, 1024).astype(np.int32)
        prepared_int = compare.prepare_signal_tensorpac(signal_int)
        assert prepared_int.dtype == np.int32


class TestShapeVerificationInPreparation:
    """Test that preparation functions properly verify shapes."""
    
    def test_prepare_signal_gpac_calls_verification(self):
        """Test that gPAC preparation calls shape verification."""
        # This should work fine
        signal = np.random.RandomState(42).randn(1024)
        prepared = compare.prepare_signal_gpac(signal)
        assert prepared.shape == (1, 1, 1024)
        
        # After preparation, it should pass verification
        assert compare.verify_input_shape_gpac(prepared) is True
    
    def test_prepare_signal_tensorpac_calls_verification(self):
        """Test that TensorPAC preparation calls shape verification."""
        # This should work fine
        signal = np.random.RandomState(42).randn(1024)
        prepared = compare.prepare_signal_tensorpac(signal)
        assert prepared.shape == (1, 1024)
        
        # After preparation, it should pass verification
        assert compare.verify_input_shape_tensorpac(prepared) is True
    
    def test_prepare_signal_tensorpac_verification_failure(self):
        """Test that TensorPAC preparation fails verification for invalid inputs."""
        # Prepare signal that would fail TensorPAC verification
        signal_3d = np.random.RandomState(42).randn(2, 8, 1024)
        
        # This should raise an error during preparation
        with pytest.raises(ValueError, match="TensorPAC expects 1D or 2D"):
            compare.prepare_signal_tensorpac(signal_3d)


class TestConsistencyBetweenPreparationMethods:
    """Test consistency and compatibility between preparation methods."""
    
    def test_prepare_signals_from_same_1d_source(self):
        """Test that both methods handle same 1D source consistently."""
        source_signal = np.random.RandomState(42).randn(1024)
        
        # Prepare for both frameworks
        gpac_signal = compare.prepare_signal_gpac(source_signal)
        tensorpac_signal = compare.prepare_signal_tensorpac(source_signal)
        
        # Check shapes are as expected
        assert gpac_signal.shape == (1, 1, 1024)
        assert tensorpac_signal.shape == (1, 1024)
        
        # Check data preservation
        np.testing.assert_array_almost_equal(
            gpac_signal.squeeze().numpy(), source_signal, decimal=6
        )
        np.testing.assert_array_equal(
            tensorpac_signal.squeeze(), source_signal
        )
    
    def test_conversion_between_formats(self):
        """Test conversion between gPAC and TensorPAC formats."""
        # Start with gPAC-compatible signal
        gpac_signal = torch.randn(1, 8, 1024)
        
        # Convert to numpy for TensorPAC (extract first channel)
        tensorpac_signal = gpac_signal[0, 0].numpy()  # Extract single channel
        prepared_tp = compare.prepare_signal_tensorpac(tensorpac_signal)
        
        assert prepared_tp.shape == (1, 1024)
        
        # Convert back to gPAC format
        prepared_gp = compare.prepare_signal_gpac(tensorpac_signal)
        
        assert prepared_gp.shape == (1, 1, 1024)
        
        # Check data preservation through conversion
        np.testing.assert_array_almost_equal(
            prepared_gp.squeeze().numpy(), 
            tensorpac_signal, 
            decimal=6
        )
    
    def test_data_type_consistency(self):
        """Test data type handling consistency."""
        source_signal = np.random.RandomState(42).randn(1024).astype(np.float32)
        
        # gPAC preparation
        gpac_signal = compare.prepare_signal_gpac(source_signal)
        assert gpac_signal.dtype == torch.float32
        
        # TensorPAC preparation
        tensorpac_signal = compare.prepare_signal_tensorpac(source_signal)
        assert tensorpac_signal.dtype == np.float32
        
        # Values should be very close (accounting for float32 precision)
        np.testing.assert_array_almost_equal(
            gpac_signal.squeeze().numpy(), 
            tensorpac_signal.squeeze(), 
            decimal=6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF