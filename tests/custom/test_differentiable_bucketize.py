#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 12:05:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/tests/custom/test_differentiable_bucketize.py

"""
Comprehensive tests for differentiable bucketize functions.

Tests include:
- Gradient flow verification
- Comparison with standard torch.bucketize
- Edge cases and numerical stability
- Different soft binning methods
"""

import pytest
import torch
import numpy as np
from torch.autograd import gradcheck
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gpac._differentiable_bucketize import (
    differentiable_bucketize,
    differentiable_bucketize_indices,
    DifferentiableBucketize,
    differentiable_phase_binning,
)


class TestDifferentiableBucketize:
    """Test suite for differentiable bucketize functions."""
    
    @pytest.fixture
    def simple_boundaries(self):
        """Simple boundary case for testing."""
        return torch.tensor([0., 1., 2., 3., 4.], dtype=torch.float32)
    
    @pytest.fixture
    def phase_boundaries(self):
        """Phase boundaries from -π to π."""
        return torch.linspace(-np.pi, np.pi, 19, dtype=torch.float32)
    
    def test_basic_functionality(self, simple_boundaries):
        """Test basic soft binning functionality."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
        
        # Test softmax method
        soft_bins = differentiable_bucketize(x, simple_boundaries, temperature=0.1)
        
        # Check shape
        assert soft_bins.shape == (4, 4)  # 4 inputs, 4 bins
        
        # Check that weights sum to 1
        assert torch.allclose(soft_bins.sum(dim=-1), torch.ones(4))
        
        # Check that each value is mostly in its correct bin
        hard_bins = soft_bins.argmax(dim=-1)
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(hard_bins, expected)
    
    def test_gradient_flow(self, simple_boundaries):
        """Test that gradients flow through the operation."""
        x = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        
        # Forward pass
        soft_bins = differentiable_bucketize(x, simple_boundaries)
        loss = soft_bins.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_temperature_effect(self, simple_boundaries):
        """Test effect of temperature on softness of binning."""
        x = torch.tensor([1.3], requires_grad=True)  # Between bins 1 and 2
        
        # Low temperature (hard binning)
        soft_bins_hard = differentiable_bucketize(
            x, simple_boundaries, temperature=0.01
        )
        
        # High temperature (soft binning)
        soft_bins_soft = differentiable_bucketize(
            x, simple_boundaries, temperature=1.0
        )
        
        # Hard binning should be more concentrated
        entropy_hard = -(soft_bins_hard * torch.log(soft_bins_hard + 1e-9)).sum()
        entropy_soft = -(soft_bins_soft * torch.log(soft_bins_soft + 1e-9)).sum()
        
        assert entropy_hard < entropy_soft
    
    def test_different_methods(self, simple_boundaries):
        """Test different soft binning methods."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
        
        # Test all methods
        for method in ["softmax", "sigmoid", "gaussian"]:
            soft_bins = differentiable_bucketize(
                x, simple_boundaries, method=method
            )
            
            # Basic checks
            assert soft_bins.shape == (4, 4)
            assert torch.allclose(soft_bins.sum(dim=-1), torch.ones(4), atol=1e-6)
            assert (soft_bins >= 0).all()
            assert (soft_bins <= 1).all()
            
            # Test gradient flow
            loss = soft_bins.sum()
            x.grad = None
            loss.backward()
            assert x.grad is not None
    
    def test_circular_binning(self, phase_boundaries):
        """Test circular binning for phase data."""
        # Test values near -π and π (should be in similar bins)
        x = torch.tensor([-3.1, 3.1], requires_grad=True)
        
        soft_bins = differentiable_bucketize(
            x, phase_boundaries, circular=True, temperature=0.1
        )
        
        # Check that -3.1 and 3.1 are in similar bins
        assert torch.allclose(soft_bins[0], soft_bins[1], atol=0.1)
    
    def test_indices_version(self, simple_boundaries):
        """Test the version that returns weighted indices."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
        
        # Get weighted indices
        soft_indices = differentiable_bucketize_indices(
            x, simple_boundaries, temperature=0.01
        )
        
        # Should be close to [0, 1, 2, 3]
        expected = torch.tensor([0., 1., 2., 3.])
        assert torch.allclose(soft_indices, expected, atol=0.1)
        
        # Test gradient flow
        loss = soft_indices.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_module_version(self, simple_boundaries):
        """Test the nn.Module version."""
        module = DifferentiableBucketize(
            simple_boundaries, temperature=0.1, return_indices=True
        )
        
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
        indices = module(x)
        
        expected = torch.tensor([0., 1., 2., 3.])
        assert torch.allclose(indices, expected, atol=0.1)
    
    def test_phase_binning_convenience(self):
        """Test the phase binning convenience function."""
        phases = torch.tensor([0., np.pi/2, np.pi, -np.pi/2], requires_grad=True)
        
        soft_bins = differentiable_phase_binning(phases, n_bins=18)
        
        # Check shape
        assert soft_bins.shape == (4, 18)
        
        # Check properties
        assert torch.allclose(soft_bins.sum(dim=-1), torch.ones(4))
        assert (soft_bins >= 0).all()
        
        # Test gradient flow
        loss = soft_bins.sum()
        loss.backward()
        assert phases.grad is not None
    
    def test_edge_cases(self, simple_boundaries):
        """Test edge cases and boundary conditions."""
        # Values exactly on boundaries
        x = torch.tensor([0., 1., 2., 3., 4.], requires_grad=True)
        
        # Test with right=False (default)
        soft_bins_left = differentiable_bucketize(
            x, simple_boundaries, temperature=0.01
        )
        
        # Test with right=True
        soft_bins_right = differentiable_bucketize(
            x, simple_boundaries, temperature=0.01, right=True
        )
        
        # The binning should be different for boundary values
        assert not torch.allclose(soft_bins_left, soft_bins_right)
    
    def test_numerical_stability(self, simple_boundaries):
        """Test numerical stability with extreme values."""
        # Very large values
        x_large = torch.tensor([1e6], requires_grad=True)
        soft_bins = differentiable_bucketize(x_large, simple_boundaries)
        assert torch.isfinite(soft_bins).all()
        
        # Very small temperature
        x = torch.tensor([1.5], requires_grad=True)
        soft_bins = differentiable_bucketize(
            x, simple_boundaries, temperature=1e-6
        )
        assert torch.isfinite(soft_bins).all()
        assert not torch.isnan(soft_bins).any()
    
    def test_batch_processing(self, simple_boundaries):
        """Test with batched inputs."""
        # 2D input
        x = torch.randn(10, 20, requires_grad=True)
        soft_bins = differentiable_bucketize(x, simple_boundaries)
        
        assert soft_bins.shape == (10, 20, 4)  # 4 bins
        assert torch.allclose(soft_bins.sum(dim=-1), torch.ones(10, 20))
        
        # Test gradient flow
        loss = soft_bins.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_comparison_with_hard_bucketize(self, simple_boundaries):
        """Compare with standard torch.bucketize in the limit."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        
        # Hard bucketize
        hard_indices = torch.bucketize(x, simple_boundaries, right=False)
        hard_indices = torch.clamp(hard_indices - 1, min=0, max=3)
        
        # Soft bucketize with very low temperature
        soft_indices = differentiable_bucketize_indices(
            x, simple_boundaries, temperature=0.001
        )
        soft_indices_rounded = torch.round(soft_indices)
        
        # Should be very close
        assert torch.allclose(
            soft_indices_rounded.float(), 
            hard_indices.float(), 
            atol=0.1
        )


class TestGradientChecking:
    """Rigorous gradient checking using torch.autograd.gradcheck."""
    
    def test_gradcheck_softmax(self):
        """Gradient check for softmax method."""
        boundaries = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
        
        def func(x):
            return differentiable_bucketize(
                x, boundaries, temperature=1.0, method="softmax"
            ).sum()
        
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        assert gradcheck(func, (x,), eps=1e-6)
    
    def test_gradcheck_sigmoid(self):
        """Gradient check for sigmoid method."""
        boundaries = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
        
        def func(x):
            return differentiable_bucketize(
                x, boundaries, temperature=1.0, method="sigmoid"
            ).sum()
        
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        assert gradcheck(func, (x,), eps=1e-6)
    
    def test_gradcheck_gaussian(self):
        """Gradient check for gaussian method."""
        boundaries = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
        
        def func(x):
            return differentiable_bucketize(
                x, boundaries, temperature=1.0, method="gaussian"
            ).sum()
        
        x = torch.randn(5, dtype=torch.float64, requires_grad=True)
        assert gradcheck(func, (x,), eps=1e-6)
    
    def test_gradcheck_circular(self):
        """Gradient check for circular binning."""
        boundaries = torch.linspace(-np.pi, np.pi, 5, dtype=torch.float64)
        
        def func(x):
            return differentiable_bucketize(
                x, boundaries, temperature=1.0, circular=True
            ).sum()
        
        x = torch.randn(3, dtype=torch.float64, requires_grad=True) * np.pi
        assert gradcheck(func, (x,), eps=1e-6)


class TestIntegrationWithModulationIndex:
    """Test integration with ModulationIndex calculations."""
    
    def test_drop_in_replacement(self):
        """Test as drop-in replacement in MI calculation."""
        from gpac._ModulationIndex import ModulationIndex
        
        # Create a modified MI class using differentiable bucketize
        class DifferentiableMI(ModulationIndex):
            def _phase_to_masks(self, pha, phase_bin_cutoffs):
                # Use differentiable bucketize
                soft_bins = differentiable_bucketize(
                    pha, phase_bin_cutoffs, 
                    temperature=0.1,
                    circular=True
                )
                # Convert to boolean-like masks for compatibility
                return soft_bins
        
        # Test with random data
        pha = torch.randn(1, 1, 2, 1, 100) * np.pi
        amp = torch.randn(1, 1, 2, 1, 100).abs()
        
        mi_module = DifferentiableMI(n_bins=18)
        result = mi_module(pha, amp)
        
        assert 'mi' in result
        assert torch.isfinite(result['mi']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF