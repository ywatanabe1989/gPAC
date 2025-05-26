#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pytest
import numpy as np
import sys
sys.path.append('src')

from gpac._BandPassFilter import BandPassFilter
from gpac._Filters._TrainableBandPassFilter import TrainableBandPassFilter
from gpac._Filters._StaticBandPassFilter import StaticBandPassFilter


class TestBandPassFilterDifferentiability:
    """Test suite for bandpass filter gradient computation and differentiability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fs = 1000.0
        self.seq_len = 2000
        self.pha_bands = [[4, 8], [8, 12]]
        self.amp_bands = [[80, 120], [120, 160]]

    def test_main_bandpass_filter_gradients(self):
        """Test that main BandPassFilter preserves gradients."""
        filter_obj = BandPassFilter(
            pha_bands=self.pha_bands,
            amp_bands=self.amp_bands,
            fs=self.fs,
            seq_len=self.seq_len
        )
        
        # Create input with gradients enabled
        x = torch.randn(1, 1, self.seq_len, requires_grad=True)
        
        # Forward pass
        filtered = filter_obj(x)
        
        # Compute dummy loss and backward pass
        loss = filtered.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        print("✅ Main BandPassFilter: Gradients flow correctly")

    def test_trainable_filter_parameter_gradients(self):
        """Test that TrainableBandPassFilter has learnable parameters with gradients."""
        filter_obj = TrainableBandPassFilter(
            fs=self.fs,
            seq_len=self.seq_len,
            pha_n_bands=2,
            amp_n_bands=2
        )
        
        # Check if using differentiable implementation
        if hasattr(filter_obj, '_use_differentiable') and filter_obj._use_differentiable:
            # Test differentiable mode
            assert hasattr(filter_obj, 'pha_mids')
            assert hasattr(filter_obj, 'amp_mids')
            assert filter_obj.pha_mids.requires_grad
            assert filter_obj.amp_mids.requires_grad
            
            # Test gradient computation through parameters
            x = torch.randn(1, 1, self.seq_len, requires_grad=True)
            filtered = filter_obj(x)
            loss = filtered.sum()
            loss.backward()
            
            # Check parameter gradients
            assert filter_obj.pha_mids.grad is not None
            assert filter_obj.amp_mids.grad is not None
            assert x.grad is not None
            print("✅ TrainableBandPassFilter: Full differentiable mode with parameter gradients")
        else:
            # Fallback mode
            x = torch.randn(1, 1, self.seq_len, requires_grad=True)
            filtered = filter_obj(x)
            loss = filtered.sum()
            loss.backward()
            
            assert x.grad is not None
            print("✅ TrainableBandPassFilter: Fallback mode with input gradients")

    def test_static_filter_no_parameters(self):
        """Test that StaticBandPassFilter has no learnable parameters."""
        filter_obj = StaticBandPassFilter(
            bands=[[8, 12]],
            fs=self.fs,
            seq_len=self.seq_len,
            band_type="phase"
        )
        
        # Check no learnable parameters
        params = list(filter_obj.parameters())
        assert len(params) == 0, "StaticBandPassFilter should have no learnable parameters"
        
        # But should still allow gradient flow through input
        x = torch.randn(1, 1, self.seq_len, requires_grad=True)
        filtered = filter_obj(x)
        loss = filtered.sum()
        loss.backward()
        
        assert x.grad is not None
        print("✅ StaticBandPassFilter: No parameters, but input gradients flow")

    def test_gradient_flow_complex_signal(self):
        """Test gradient flow with more complex signal processing."""
        filter_obj = TrainableBandPassFilter(
            fs=self.fs,
            seq_len=self.seq_len,
            pha_n_bands=1,
            amp_n_bands=1
        )
        
        # Create signal with known frequency content
        t = torch.linspace(0, self.seq_len/self.fs, self.seq_len, requires_grad=True)
        freq = torch.tensor(10.0, requires_grad=True)  # Learnable frequency
        signal = torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(0)
        
        # Apply filter
        filtered = filter_obj(signal)
        
        # Loss based on filtered signal energy
        loss = torch.mean(filtered ** 2)
        loss.backward()
        
        # Check gradients for both input frequency and filter parameters
        assert freq.grad is not None
        
        # Check filter parameter gradients (depends on torchaudio availability)
        if hasattr(filter_obj, '_use_differentiable') and filter_obj._use_differentiable:
            # Using torchaudio - check pha_mids
            assert hasattr(filter_obj, 'pha_mids')
            assert filter_obj.pha_mids.grad is not None
        else:
            # Fallback mode - check pha_low_hz
            assert hasattr(filter_obj, 'pha_low_hz')
            assert filter_obj.pha_low_hz.grad is not None
            
        print("✅ Complex signal: Gradients flow through signal frequency and filter parameters")

    def test_filtfilt_mode_differentiability(self):
        """Test differentiability with filtfilt mode (zero-phase filtering)."""
        filter_obj = BandPassFilter(
            pha_bands=[[8, 12]],
            amp_bands=[],
            fs=self.fs,
            seq_len=self.seq_len,
            filtfilt_mode=True
        )
        
        x = torch.randn(1, 1, self.seq_len, requires_grad=True)
        filtered = filter_obj(x)
        loss = filtered.sum()
        loss.backward()
        
        assert x.grad is not None
        print("✅ Filtfilt mode: Gradients preserved through zero-phase filtering")

    def test_fp16_mode_differentiability(self):
        """Test differentiability with half precision mode."""
        filter_obj = BandPassFilter(
            pha_bands=[[8, 12]],
            amp_bands=[],
            fs=self.fs,
            seq_len=self.seq_len,
            fp16=True
        )
        
        x = torch.randn(1, 1, self.seq_len, requires_grad=True, dtype=torch.float16)
        filtered = filter_obj(x)
        loss = filtered.sum()
        loss.backward()
        
        assert x.grad is not None
        print("✅ FP16 mode: Gradients preserved in half precision")

    def test_parameter_constraints_differentiability(self):
        """Test that parameter constraints don't break differentiability."""
        filter_obj = TrainableBandPassFilter(
            fs=self.fs,
            seq_len=self.seq_len,
            pha_low_hz=1000,  # Will be constrained
            pha_high_hz=2000,  # Will be constrained
            pha_n_bands=1,
            amp_n_bands=0
        )
        
        x = torch.randn(1, 1, self.seq_len)
        
        # Multiple forward passes to trigger constraint application
        for _ in range(3):
            filtered = filter_obj(x)
            loss = filtered.sum()
            loss.backward()
            
            # Check gradients still exist after constraints
            assert filter_obj.pha_low_hz.grad is not None
            filter_obj.zero_grad()
        
        print("✅ Parameter constraints: Gradients preserved through constraint application")

    def test_kernel_rebuild_differentiability(self):
        """Test that kernel rebuilding in trainable filter maintains differentiability."""
        filter_obj = TrainableBandPassFilter(
            fs=self.fs,
            seq_len=self.seq_len,
            pha_n_bands=1,
            amp_n_bands=0
        )
        
        x = torch.randn(1, 1, self.seq_len)
        
        # First forward pass
        filtered1 = filter_obj(x)
        
        # Modify parameters
        with torch.no_grad():
            filter_obj.pha_low_hz.add_(1.0)
        
        # Second forward pass (should rebuild kernels)
        filtered2 = filter_obj(x)
        loss = (filtered1 - filtered2).pow(2).sum()
        loss.backward()
        
        # Check gradients exist after kernel rebuild
        assert filter_obj.pha_low_hz.grad is not None
        print("✅ Kernel rebuilding: Gradients preserved through dynamic kernel creation")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])