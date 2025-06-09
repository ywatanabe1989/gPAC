#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-06 20:42:00 (ywatanabe)"
# File: ./tests/gpac/utils/compare/test_shape_verification.py

import pytest
import numpy as np
import torch
from gpac.utils import compare


class TestShapeVerificationDetailed:
    """Detailed tests for shape verification functions."""
    
    def test_verify_input_shape_gpac_3d_batch_variations(self):
        """Test various 3D batch configurations."""
        # Single batch
        signal = torch.randn(1, 8, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
        
        # Multiple batches
        signal = torch.randn(16, 8, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
        
        # Single channel
        signal = torch.randn(1, 1, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
    
    def test_verify_input_shape_gpac_4d_variations(self):
        """Test various 4D configurations."""
        # Standard 4D
        signal = torch.randn(1, 8, 10, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
        
        # Single epoch
        signal = torch.randn(1, 8, 1, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
        
        # Many epochs
        signal = torch.randn(1, 8, 100, 1024)
        assert compare.verify_input_shape_gpac(signal) is True
    
    def test_verify_input_shape_gpac_edge_cases(self):
        """Test edge cases for gPAC shape verification."""
        # 2D input (invalid)
        signal = torch.randn(8, 1024)
        with pytest.raises(ValueError, match="gPAC expects"):
            compare.verify_input_shape_gpac(signal)
        
        # 5D input (invalid)
        signal = torch.randn(1, 8, 10, 5, 1024)
        with pytest.raises(ValueError, match="gPAC expects"):
            compare.verify_input_shape_gpac(signal)
        
        # Custom expected dimensions
        signal = torch.randn(1, 8, 10, 5, 1024)
        assert compare.verify_input_shape_gpac(signal, expected_dims=[5]) is True
    
    def test_verify_input_shape_tensorpac_edge_cases(self):
        """Test edge cases for TensorPAC shape verification."""
        # Minimum valid shape
        signal = np.random.randn(1, 100)
        assert compare.verify_input_shape_tensorpac(signal) is True
        
        # Large shape
        signal = np.random.randn(1000, 10000)
        assert compare.verify_input_shape_tensorpac(signal) is True
        
        # 1D input (invalid)
        signal = np.random.randn(1024)
        with pytest.raises(ValueError, match="TensorPAC expects 2D"):
            compare.verify_input_shape_tensorpac(signal)
        
        # 3D input (invalid)
        signal = np.random.randn(1, 8, 1024)
        with pytest.raises(ValueError, match="TensorPAC expects 2D"):
            compare.verify_input_shape_tensorpac(signal)
    
    def test_verify_output_shapes_match_comprehensive(self):
        """Comprehensive tests for output shape matching."""
        # Perfect match
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(10, 8)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=False)
        assert match is True
        assert np.array_equal(corrected, pac_tp)
        
        # Transpose match
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(8, 10)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=False)
        assert match is True
        assert corrected.shape == (10, 8)
        
        # Same dimensions, different order (should fail)
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(8, 10)
        # Force no transpose by using 3D
        pac_gp_3d = pac_gp.reshape(10, 8, 1)
        pac_tp_3d = pac_tp.reshape(8, 10, 1)
        match, corrected = compare.verify_output_shapes_match(pac_gp_3d, pac_tp_3d, verbose=False)
        # This should fail since transpose won't work for 3D
        
        # Completely different shapes
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(5, 12)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=False)
        assert match is False
        assert np.array_equal(corrected, pac_tp)
    
    def test_verify_output_shapes_match_verbose(self):
        """Test verbose output for shape mismatch."""
        import io
        import sys
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        pac_gp = np.random.randn(10, 8)
        pac_tp = np.random.randn(8, 10)
        match, corrected = compare.verify_output_shapes_match(pac_gp, pac_tp, verbose=True)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "Shape mismatch" in output
        assert "Fixed by transposing" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF