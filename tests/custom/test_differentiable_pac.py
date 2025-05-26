#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-26 00:52:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/tests/custom/test_differentiable_pac.py

"""
Test differentiable PAC functionality when trainable=True.

Verifies that soft binning enables end-to-end gradient flow
for PAC computation with trainable filters.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from gpac._PAC import PAC
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex


class TestDifferentiablePAC:
    """Test differentiable PAC with trainable=True."""

    @pytest.fixture
    def synthetic_pac_signal(self):
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

    def test_differentiable_mi_module(self):
        """Test DifferentiableModulationIndex directly."""
        torch.manual_seed(42)
        
        # Create test data
        batch_size, n_channels, n_pha_bands, n_amp_bands = 1, 1, 3, 2
        n_segments, seq_len = 1, 128
        
        pha_base = torch.randn(batch_size, n_channels, n_pha_bands, n_segments, seq_len,
                              requires_grad=True, dtype=torch.float32)
        amp_base = torch.randn(batch_size, n_channels, n_amp_bands, n_segments, seq_len,
                              requires_grad=True, dtype=torch.float32)
        
        pha = pha_base * torch.pi
        amp = torch.abs(amp_base)
        
        # Test differentiable MI
        diff_mi = DifferentiableModulationIndex(
            n_bins=18, 
            temperature=1.0, 
            binning_method='softmax',
            fp16=False
        )
        
        result = diff_mi(pha, amp)
        
        # Check output structure
        assert 'mi' in result
        assert 'amp_prob' in result
        assert 'pha_bin_centers' in result
        
        # Test gradient flow
        loss = result['mi'].sum()
        loss.backward()
        
        # Verify gradients computed on base tensors
        assert pha_base.grad is not None
        assert amp_base.grad is not None
        assert torch.isfinite(pha_base.grad).all()
        assert torch.isfinite(amp_base.grad).all()

    def test_pac_trainable_mode_gradients(self, synthetic_pac_signal):
        """Test end-to-end gradient flow with trainable PAC."""
        signal, fs = synthetic_pac_signal
        
        try:
            # Initialize trainable PAC
            pac = PAC(
                seq_len=signal.shape[-1],
                fs=fs,
                pha_start_hz=2.0,
                pha_end_hz=20.0,
                pha_n_bands=3,
                amp_start_hz=60.0,
                amp_end_hz=120.0,
                amp_n_bands=2,
                trainable=True,  # Enable differentiable mode
                fp16=False,
                mi_n_bins=18,
                differentiable_mi_temperature=1.0,
                differentiable_mi_method='softmax'
            )
            
            # Verify we have trainable parameters
            params = list(pac.parameters())
            assert len(params) > 0, "PAC should have trainable parameters"
            
            # Forward pass
            result = pac(signal)
            
            # Check results
            assert 'mi' in result
            assert result['mi'].requires_grad
            
            # Test end-to-end gradient flow
            loss = result['mi'].sum()
            loss.backward()
            
            # Verify gradients on parameters
            for param in pac.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert torch.isfinite(param.grad).all()
            
            # Verify gradients on input
            assert signal.grad is not None
            assert torch.isfinite(signal.grad).all()
            
        except ImportError as e:
            pytest.skip(f"Trainable filters not available: {e}")

    def test_pac_static_vs_differentiable_comparison(self, synthetic_pac_signal):
        """Compare static vs differentiable MI results."""
        signal, fs = synthetic_pac_signal
        signal_static = signal.clone().detach()  # Remove gradients for static test
        
        # Static PAC (standard MI)
        pac_static = PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_n_bands=3,
            amp_n_bands=2,
            trainable=False,
            mi_n_bins=18
        )
        
        result_static = pac_static(signal_static)
        
        try:
            # Differentiable PAC (soft MI)
            pac_diff = PAC(
                seq_len=signal.shape[-1],
                fs=fs,
                pha_n_bands=3,
                amp_n_bands=2,
                trainable=True,
                mi_n_bins=18,
                differentiable_mi_temperature=0.1  # Lower temp for closer to hard binning
            )
            
            result_diff = pac_diff(signal)
            
            # Results should be in same ballpark (soft binning approximates hard binning)
            assert result_static['mi'].shape == result_diff['mi'].shape
            assert torch.allclose(result_static['mi'], result_diff['mi'], atol=0.5)
            
        except ImportError:
            pytest.skip("Trainable filters not available")

    def test_differentiable_mi_temperature_effect(self):
        """Test effect of temperature parameter on soft binning."""
        torch.manual_seed(42)
        
        # Create test data  
        pha_base = torch.randn(1, 1, 2, 1, 64, requires_grad=True)
        amp_base = torch.randn(1, 1, 2, 1, 64, requires_grad=True)
        pha = pha_base * torch.pi
        amp = torch.abs(amp_base)
        
        # Test different temperatures
        temps = [0.1, 1.0, 10.0]
        results = []
        
        for temp in temps:
            diff_mi = DifferentiableModulationIndex(
                n_bins=10, 
                temperature=temp,
                binning_method='softmax'
            )
            result = diff_mi(pha, amp)
            results.append(result['mi'])
        
        # Lower temperature should give sharper (more discrete) results
        # Higher temperature should give smoother results
        assert not torch.allclose(results[0], results[1])
        assert not torch.allclose(results[1], results[2])

    def test_differentiable_mi_binning_methods(self):
        """Test different soft binning methods."""
        torch.manual_seed(42)
        
        # Create test data
        pha_base = torch.randn(1, 1, 2, 1, 64, requires_grad=True)
        amp_base = torch.randn(1, 1, 2, 1, 64, requires_grad=True)
        pha = pha_base * torch.pi  
        amp = torch.abs(amp_base)
        
        # Test both methods
        methods = ['softmax', 'gaussian']
        results = []
        
        for method in methods:
            diff_mi = DifferentiableModulationIndex(
                n_bins=10,
                temperature=1.0,
                binning_method=method
            )
            result = diff_mi(pha, amp)
            results.append(result['mi'])
            
            # Test gradient flow
            loss = result['mi'].sum()
            loss.backward()
            assert pha_base.grad is not None
            pha_base.grad.zero_()  # Clear for next test
        
        # Results should be different but reasonable
        assert not torch.allclose(results[0], results[1])
        assert torch.isfinite(results[0]).all()
        assert torch.isfinite(results[1]).all()

    def test_pac_with_permutation_and_trainable(self, synthetic_pac_signal):
        """Test trainable PAC with permutation testing."""
        signal, fs = synthetic_pac_signal
        
        try:
            pac = PAC(
                seq_len=signal.shape[-1],
                fs=fs,
                pha_n_bands=2,
                amp_n_bands=2,
                n_perm=5,  # Small number for testing
                trainable=True,
                differentiable_mi_temperature=1.0
            )
            
            result = pac(signal)
            
            # Check permutation results included
            assert 'mi_z' in result
            assert 'surrogate_mis' in result
            
            # Test gradient flow through permutation path
            loss = result['mi_z'].sum()
            loss.backward()
            
            # Verify gradients
            assert signal.grad is not None
            assert torch.isfinite(signal.grad).all()
            
        except ImportError:
            pytest.skip("Trainable filters not available")


if __name__ == "__main__":
    pytest.main([__file__])

# EOF