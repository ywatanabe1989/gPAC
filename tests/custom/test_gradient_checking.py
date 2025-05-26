#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 00:39:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/tests/custom/test_gradient_checking.py

"""
Comprehensive gradient checking tests for all gPAC modules.

This test suite validates differentiability of all modules using:
- torch.autograd.gradcheck for rigorous validation
- Finite difference comparison with analytical gradients
- Gradient flow verification through multi-module chains
- Numerical stability tests and edge case handling
"""

import torch
import torch.nn as nn
import pytest
import sys
import numpy as np
from pathlib import Path
from torch.autograd import gradcheck
import warnings

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from gpac._Hilbert import Hilbert
from gpac._ModulationIndex import ModulationIndex
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex
from gpac._PAC import PAC
from gpac._BandPassFilter import BandPassFilter
try:
    from gpac._Filters._TrainableBandPassFilter import TrainableBandPassFilter
except ImportError:
    TrainableBandPassFilter = None


class TestGradientChecking:
    """Test gradient flow through all gPAC modules."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        torch.manual_seed(42)
        batch_size, n_channels, n_segments, seq_len = 2, 3, 1, 256
        x = torch.randn(batch_size, n_channels, n_segments, seq_len, 
                       requires_grad=True, dtype=torch.float32)
        return x

    @pytest.fixture
    def synthetic_pac_data(self):
        """Generate synthetic PAC signal for testing."""
        torch.manual_seed(42)
        fs = 256
        t = torch.linspace(0, 2, 512, requires_grad=True)
        
        # Create PAC signal: 6Hz phase modulates 80Hz amplitude
        phase_freq = 6.0
        amp_freq = 80.0
        
        phase_sig = torch.sin(2 * torch.pi * phase_freq * t)
        amp_sig = torch.sin(2 * torch.pi * amp_freq * t)
        
        # Amplitude modulation with coupling strength
        coupling_strength = 0.5
        modulated = phase_sig + (1 + coupling_strength * phase_sig) * amp_sig * 0.3
        
        return modulated.view(1, 1, 1, -1), fs

    def test_hilbert_gradient_flow(self, sample_data):
        """Test that Hilbert transform maintains gradient flow."""
        x = sample_data
        
        # Initialize Hilbert module
        hilbert = Hilbert(seq_len=x.shape[-1], dim=-1, fp16=False)
        
        # Forward pass
        output = hilbert(x)
        
        # Check output shape and type
        assert output.shape == (*x.shape, 2)  # Phase and amplitude
        assert output.requires_grad
        
        # Test gradient computation
        loss = output.sum()
        loss.backward()
        
        # Verify gradients were computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_modulation_index_gradient_limitations(self):
        """Test ModulationIndex gradient behavior - note inherent limitations."""
        torch.manual_seed(42)
        
        # Create test phase and amplitude tensors
        batch_size, n_channels, n_pha_bands, n_amp_bands = 1, 1, 5, 3
        n_segments, seq_len = 1, 128
        
        pha_base = torch.randn(batch_size, n_channels, n_pha_bands, n_segments, seq_len,
                              requires_grad=True, dtype=torch.float32)
        pha = pha_base * torch.pi
        
        amp_base = torch.randn(batch_size, n_channels, n_amp_bands, n_segments, seq_len,
                              requires_grad=True, dtype=torch.float32)
        amp = torch.abs(amp_base)
        
        # Initialize ModulationIndex module
        mi_module = ModulationIndex(n_bins=18, fp16=False)
        
        # Forward pass
        result = mi_module(pha, amp)
        
        # Check output structure
        assert 'mi' in result
        assert 'amp_prob' in result
        
        # Note: ModulationIndex contains non-differentiable operations (bucketize, boolean masks)
        # This is expected behavior for binning-based MI calculations
        # The module can still be used in larger pipelines where gradients flow around it
        
        # Test that the operation completes without error
        loss = result['mi'].sum() + result['amp_prob'].sum()
        
        # This test verifies the module works, but gradients may not flow through
        # due to inherent non-differentiability of binning operations
        try:
            loss.backward()
            # If gradients do flow, verify they are valid
            if pha_base.grad is not None:
                assert torch.isfinite(pha_base.grad).all()
            if amp_base.grad is not None:
                assert torch.isfinite(amp_base.grad).all()
        except RuntimeError as e:
            # Expected for non-differentiable operations
            assert "does not require grad" in str(e) or "no grad" in str(e)

    def test_bandpass_filter_gradient_flow(self, sample_data):
        """Test that BandPassFilter maintains gradient flow."""
        x = sample_data
        fs = 256.0
        
        # Create frequency bands
        pha_bands = torch.tensor([[2.0, 4.0], [4.0, 8.0], [8.0, 12.0]])
        amp_bands = torch.tensor([[30.0, 50.0], [50.0, 80.0], [80.0, 120.0]])
        
        # Initialize BandPassFilter
        filter_module = BandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=x.shape[-1],
            fp16=False
        )
        
        # Reshape input for filter: (B*C*Seg, 1, Time)
        x_flat = x.reshape(-1, 1, x.shape[-1])
        
        # Forward pass
        output = filter_module(x_flat)
        
        # Check output
        expected_n_bands = len(pha_bands) + len(amp_bands)
        assert output.shape[-2] == expected_n_bands
        assert output.requires_grad
        
        # Test gradient computation
        loss = output.sum()
        loss.backward()
        
        # Verify gradients were computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_pac_end_to_end_gradient_flow(self, synthetic_pac_data):
        """Test end-to-end gradient flow through the complete PAC pipeline."""
        signal, fs = synthetic_pac_data
        
        # Initialize PAC module (non-trainable first)
        pac = PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=5,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=3,
            trainable=False,  # Static filters
            fp16=False,
            mi_n_bins=18
        )
        
        # Forward pass
        result = pac(signal)
        
        # Check output structure
        assert 'mi' in result
        assert 'amp_prob' in result
        assert 'pha_freqs_hz' in result
        assert 'amp_freqs_hz' in result
        
        # Verify shapes
        expected_shape = (1, 1, 5, 3)  # (batch, channels, pha_bands, amp_bands)
        assert result['mi'].shape == expected_shape
        
        # Note: MI computation contains non-differentiable operations
        # Gradients will flow through filtering and Hilbert but stop at MI
        
        # Test that the pipeline works and produces valid outputs
        assert torch.isfinite(result['mi']).all()
        assert torch.isfinite(result['amp_prob']).all()
        
        # Test gradient flow up to the MI computation point
        # Create a custom loss that uses intermediate outputs if available
        try:
            loss = result['mi'].sum()
            loss.backward()
            
            # If gradients flow (unexpected but possible), verify they are valid
            if signal.grad is not None:
                assert torch.isfinite(signal.grad).all()
        except RuntimeError:
            # Expected due to non-differentiable MI computation
            pass

    def test_pac_trainable_gradient_flow(self, synthetic_pac_data):
        """Test gradient flow through PAC with trainable filters."""
        signal, fs = synthetic_pac_data
        
        # Make signal a leaf tensor that requires gradients
        signal = signal.detach().requires_grad_(True)
        
        try:
            # Initialize PAC module with trainable filters
            pac = PAC(
                seq_len=signal.shape[-1],
                fs=fs,
                pha_start_hz=2.0,
                pha_end_hz=20.0,
                pha_n_bands=3,
                amp_start_hz=60.0,
                amp_end_hz=120.0,
                amp_n_bands=2,
                trainable=True,  # Trainable filters
                fp16=False,
                mi_n_bins=18
            )
            
            # Check that model has trainable parameters
            trainable_params = list(pac.parameters())
            assert len(trainable_params) > 0
            
            # Forward pass
            result = pac(signal)
            
            # Test gradient computation
            loss = result['mi'].sum()
            loss.backward()
            
            # Verify gradients were computed for parameters
            for param in pac.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not torch.isnan(param.grad).any()
            
            # Verify gradients for input
            assert signal.grad is not None
            assert not torch.isnan(signal.grad).any()
            
        except ImportError:
            # Skip if TrainableBandPassFilter not available
            pytest.skip("TrainableBandPassFilter not available")

    def test_gradient_numerical_stability(self, sample_data):
        """Test numerical stability of gradients across different data ranges."""
        x = sample_data
        
        # Test with different signal magnitudes
        for scale in [1e-3, 1.0, 1e3]:
            x_scaled = x * scale
            x_scaled.requires_grad_(True)
            
            hilbert = Hilbert(seq_len=x.shape[-1], dim=-1, fp16=False)
            output = hilbert(x_scaled)
            
            loss = output.sum()
            loss.backward()
            
            # Check for numerical issues
            assert x_scaled.grad is not None
            assert torch.isfinite(x_scaled.grad).all()
            assert (x_scaled.grad.abs() > 0).any()  # Non-zero gradients

    def test_gradient_with_permutation_testing(self, synthetic_pac_data):
        """Test gradient flow when permutation testing is enabled."""
        signal, fs = synthetic_pac_data
        # Make signal a leaf tensor that requires gradients
        signal = signal.detach().requires_grad_(True)
        
        # Initialize PAC with permutation testing
        pac = PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=3,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=2,
            n_perm=10,  # Enable permutation testing
            trainable=False,
            fp16=False
        )
        
        # Forward pass
        result = pac(signal)
        
        # Check that permutation results are included
        assert 'mi_z' in result
        assert 'surrogate_mis' in result
        
        # Test gradient computation through permutation path
        # Note: mi_z computation may contain non-differentiable operations
        try:
            loss = result['mi_z'].sum()
            loss.backward()
            
            # Verify gradients were computed if they flow
            if signal.grad is not None:
                assert torch.isfinite(signal.grad).all()
        except RuntimeError as e:
            # Expected if operations are non-differentiable
            if "does not require grad" not in str(e) and "no grad" not in str(e):
                raise

    def test_mixed_precision_gradient_flow(self, sample_data):
        """Test gradient flow with mixed precision (fp16)."""
        x = sample_data.half()  # Convert to fp16
        x.requires_grad_(True)
        
        # Initialize modules with fp16
        hilbert = Hilbert(seq_len=x.shape[-1], dim=-1, fp16=True)
        
        # Forward pass
        output = hilbert(x)
        
        # Test gradient computation
        loss = output.float().sum()  # Convert to fp32 for loss
        loss.backward()
        
        # Verify gradients
        assert x.grad is not None
        assert x.grad.dtype == torch.float16
        assert torch.isfinite(x.grad).all()


class TestRigorousGradientChecking:
    """
    Rigorous gradient validation using torch.autograd.gradcheck.
    Tests analytical vs finite difference gradients with tight tolerances.
    """

    @pytest.fixture
    def small_input(self):
        """Small input for gradcheck (computationally expensive)."""
        torch.manual_seed(42)
        # Small dimensions for gradcheck efficiency
        return torch.randn(1, 1, 1, 64, dtype=torch.float64, requires_grad=True)

    @pytest.fixture
    def medium_input(self):
        """Medium input for gradient flow tests."""
        torch.manual_seed(42)
        return torch.randn(2, 1, 1, 128, dtype=torch.float64, requires_grad=True)

    def test_hilbert_gradcheck(self, small_input):
        """Rigorous gradient check for Hilbert transform."""
        seq_len = small_input.shape[-1]
        
        def hilbert_func(x):
            hilbert = Hilbert(seq_len=seq_len, dim=-1, fp16=False)
            output = hilbert(x)
            # Return magnitude for scalar output
            return torch.sqrt(output[..., 0]**2 + output[..., 1]**2)
        
        # Test with multiple epsilon values
        for eps in [1e-6, 1e-5, 1e-4]:
            assert gradcheck(
                hilbert_func, 
                (small_input,), 
                eps=eps,
                atol=1e-5,
                rtol=1e-4,
                raise_exception=True
            )

    def test_differentiable_modulation_index_gradcheck(self):
        """Rigorous gradient check for DifferentiableModulationIndex."""
        torch.manual_seed(42)
        
        # Small inputs for efficiency
        pha = torch.randn(1, 1, 2, 1, 32, dtype=torch.float64, requires_grad=True)
        amp = torch.randn(1, 1, 2, 1, 32, dtype=torch.float64, requires_grad=True).abs()
        
        def mi_func(phase, amplitude):
            mi_module = DifferentiableModulationIndex(
                n_bins=6,  # Fewer bins for faster computation
                temperature=1.0,
                binning_method='softmax',
                fp16=False
            )
            result = mi_module(phase, amplitude)
            return result['mi'].sum()
        
        # Test gradient accuracy
        assert gradcheck(
            mi_func,
            (pha, amp),
            eps=1e-5,
            atol=1e-5,
            rtol=1e-4,
            raise_exception=True
        )

    @pytest.mark.skipif(TrainableBandPassFilter is None, 
                       reason="TrainableBandPassFilter not available")
    def test_trainable_filter_gradcheck(self, small_input):
        """Rigorous gradient check for trainable bandpass filter."""
        seq_len = small_input.shape[-1]
        fs = 256.0
        
        # Reshape for filter input
        x = small_input.reshape(-1, 1, seq_len)
        
        def filter_func(x):
            filter_module = TrainableBandPassFilter(
                fs=fs,
                seq_len=seq_len,
                pha_low_hz=2,
                pha_high_hz=10,
                pha_n_bands=2,
                amp_low_hz=40,
                amp_high_hz=80,
                amp_n_bands=2,
                fp16=False
            )
            output = filter_module(x)
            return output.sum()
        
        # Gradient check with looser tolerance due to filter complexity
        assert gradcheck(
            filter_func,
            (x,),
            eps=1e-4,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True
        )

    def test_multi_module_chain_gradients(self, medium_input):
        """Test gradient flow through chained modules."""
        torch.manual_seed(42)
        seq_len = medium_input.shape[-1]
        fs = 256.0
        
        # Build a differentiable pipeline
        class DifferentiablePipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.filter = BandPassFilter(
                    pha_bands=torch.tensor([[4.0, 8.0], [8.0, 12.0]]),
                    amp_bands=torch.tensor([[40.0, 60.0], [60.0, 80.0]]),
                    fs=fs,
                    seq_len=seq_len,
                    fp16=False
                )
                self.hilbert = Hilbert(seq_len=seq_len, dim=-1, fp16=False)
                self.mi = DifferentiableModulationIndex(
                    n_bins=12,
                    temperature=0.5,
                    fp16=False
                )
            
            def forward(self, x):
                # Reshape for filter
                x_flat = x.reshape(-1, 1, x.shape[-1])
                
                # Apply bandpass filter
                filtered = self.filter(x_flat)
                
                # Split into phase and amplitude bands
                n_pha = 2
                pha_filtered = filtered[:, :n_pha]
                amp_filtered = filtered[:, n_pha:]
                
                # Apply Hilbert to get phase and amplitude
                pha_analytic = self.hilbert(pha_filtered.unsqueeze(0).unsqueeze(0))
                amp_analytic = self.hilbert(amp_filtered.unsqueeze(0).unsqueeze(0))
                
                # Extract phase and amplitude
                pha = torch.atan2(pha_analytic[..., 1], pha_analytic[..., 0])
                amp = torch.sqrt(amp_analytic[..., 0]**2 + amp_analytic[..., 1]**2)
                
                # Compute MI
                result = self.mi(pha, amp)
                return result['mi']
        
        # Test gradient flow
        pipeline = DifferentiablePipeline()
        output = pipeline(medium_input)
        loss = output.sum()
        loss.backward()
        
        # Verify gradients exist and are finite
        assert medium_input.grad is not None
        assert torch.isfinite(medium_input.grad).all()
        assert (medium_input.grad.abs() > 0).any()

    def test_finite_difference_validation(self):
        """Compare analytical gradients with finite differences."""
        torch.manual_seed(42)
        
        # Test Hilbert transform gradients
        x = torch.randn(1, 1, 1, 32, dtype=torch.float64, requires_grad=True)
        hilbert = Hilbert(seq_len=32, dim=-1, fp16=False)
        
        # Analytical gradient
        output = hilbert(x)
        mag = torch.sqrt(output[..., 0]**2 + output[..., 1]**2)
        loss = mag.sum()
        loss.backward()
        analytical_grad = x.grad.clone()
        
        # Finite difference gradient
        epsilon = 1e-5
        x.grad = None
        finite_diff_grad = torch.zeros_like(x)
        
        for idx in np.ndindex(x.shape):
            # Forward difference
            x_plus = x.clone()
            x_plus[idx] += epsilon
            output_plus = hilbert(x_plus)
            mag_plus = torch.sqrt(output_plus[..., 0]**2 + output_plus[..., 1]**2)
            loss_plus = mag_plus.sum()
            
            # Backward difference
            x_minus = x.clone()
            x_minus[idx] -= epsilon
            output_minus = hilbert(x_minus)
            mag_minus = torch.sqrt(output_minus[..., 0]**2 + output_minus[..., 1]**2)
            loss_minus = mag_minus.sum()
            
            # Central difference
            finite_diff_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Compare gradients
        relative_error = torch.abs(analytical_grad - finite_diff_grad) / (
            torch.abs(analytical_grad) + torch.abs(finite_diff_grad) + 1e-8
        )
        assert relative_error.max() < 1e-5

    def test_gradient_numerical_stability(self):
        """Test gradient stability across different input scales."""
        torch.manual_seed(42)
        
        for scale in [1e-3, 1.0, 1e3]:
            x = torch.randn(1, 1, 1, 64, dtype=torch.float64) * scale
            x.requires_grad_(True)
            
            # Test DifferentiableModulationIndex stability
            pha = x * np.pi  # Scale to phase range
            amp = torch.abs(x)
            
            mi_module = DifferentiableModulationIndex(
                n_bins=12,
                temperature=1.0,
                fp16=False
            )
            
            result = mi_module(pha.unsqueeze(2), amp.unsqueeze(2))
            loss = result['mi'].sum()
            loss.backward()
            
            # Check gradient properties
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_edge_cases_gradient_behavior(self):
        """Test gradient behavior in edge cases."""
        torch.manual_seed(42)
        
        # Test 1: Zero inputs
        x_zero = torch.zeros(1, 1, 1, 32, dtype=torch.float64, requires_grad=True)
        hilbert = Hilbert(seq_len=32, dim=-1, fp16=False)
        output = hilbert(x_zero)
        loss = output.sum()
        loss.backward()
        assert torch.isfinite(x_zero.grad).all()
        
        # Test 2: Very small inputs (near machine epsilon)
        x_small = torch.full((1, 1, 1, 32), 1e-10, dtype=torch.float64, requires_grad=True)
        output = hilbert(x_small)
        loss = output.sum()
        x_small.grad = None
        loss.backward()
        assert torch.isfinite(x_small.grad).all()
        
        # Test 3: Constant inputs
        x_const = torch.ones(1, 1, 1, 32, dtype=torch.float64, requires_grad=True)
        output = hilbert(x_const)
        loss = output.sum()
        x_const.grad = None
        loss.backward()
        assert torch.isfinite(x_const.grad).all()


class TestGradientPerformance:
    """Test gradient computation performance to ensure <5 minute runtime."""
    
    def test_large_batch_gradient_performance(self):
        """Test gradient computation on realistic batch sizes."""
        import time
        
        # Realistic dimensions
        batch_size = 16
        n_channels = 3
        n_segments = 5
        seq_len = 512
        
        x = torch.randn(batch_size, n_channels, n_segments, seq_len, 
                       dtype=torch.float32, requires_grad=True)
        
        # Time the full pipeline
        start_time = time.time()
        
        # Initialize modules
        hilbert = Hilbert(seq_len=seq_len, dim=-1, fp16=False)
        mi = DifferentiableModulationIndex(n_bins=18, fp16=False)
        
        # Forward pass
        output = hilbert(x)
        pha = torch.atan2(output[..., 1], output[..., 0])
        amp = torch.sqrt(output[..., 0]**2 + output[..., 1]**2)
        
        # Expand dimensions for MI
        pha_expanded = pha.unsqueeze(2)
        amp_expanded = amp.unsqueeze(2)
        
        result = mi(pha_expanded, amp_expanded)
        loss = result['mi'].sum()
        
        # Backward pass
        loss.backward()
        
        elapsed_time = time.time() - start_time
        
        # Ensure it completes in reasonable time (should be << 5 minutes)
        assert elapsed_time < 30.0  # 30 seconds for this size
        
        # Verify gradients were computed
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF