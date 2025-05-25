#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for bandpass filtering module.
"""

import pytest
import torch
import numpy as np
from scipy import signal

import sys
sys.path.append('../src')
from gpac._PAC import CombinedBandPassFilter
from gpac._tensorpac_fir1 import design_filter_tensorpac


class TestBandpassFilter:
    """Test bandpass filtering functionality."""
    
    @pytest.fixture
    def setup_filter(self):
        """Setup test filter."""
        fs = 512
        seq_len = 1024
        pha_bands = torch.tensor([[4., 8.], [8., 12.]])
        amp_bands = torch.tensor([[60., 80.], [80., 100.]])
        
        filter_module = CombinedBandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=seq_len,
            cycle_pha=3,
            cycle_amp=6
        )
        return filter_module, fs, seq_len
    
    def test_filter_initialization(self, setup_filter):
        """Test filter initialization."""
        filter_module, fs, seq_len = setup_filter
        
        # Check kernels are created
        assert hasattr(filter_module, 'kernels')
        assert filter_module.kernels.shape[0] == 4  # 2 phase + 2 amplitude
        
    def test_single_frequency_filtering(self, setup_filter):
        """Test filtering of single frequency."""
        filter_module, fs, seq_len = setup_filter
        
        # Create test signal with 6 Hz (should pass through first phase filter)
        t = torch.linspace(0, seq_len/fs, seq_len)
        test_freq = 6.0  # Hz
        signal = torch.sin(2 * np.pi * test_freq * t)
        signal = signal.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Apply filter
        filtered = filter_module(signal)
        
        # Check output shape
        assert filtered.shape == (1, 1, 4, seq_len)
        
        # First phase band (4-8 Hz) should have strong response
        phase_band1_power = filtered[0, 0, 0, :].abs().mean()
        phase_band2_power = filtered[0, 0, 1, :].abs().mean()
        
        assert phase_band1_power > phase_band2_power
        
    def test_filtfilt_mode(self, setup_filter):
        """Test sequential filtfilt mode."""
        filter_module, _, seq_len = setup_filter
        
        # Create filter with filtfilt mode
        filter_module.filtfilt_mode = True
        
        # Create test signal
        signal = torch.randn(1, 1, seq_len)
        
        # Apply filter
        filtered = filter_module(signal)
        
        # Check output shape
        assert filtered.shape == (1, 1, 4, seq_len)
        
    def test_edge_mode(self):
        """Test edge padding modes."""
        fs = 512
        seq_len = 1024
        pha_bands = torch.tensor([[4., 8.]])
        amp_bands = torch.tensor([[60., 80.]])
        
        # Test different edge modes
        for edge_mode in ['reflect', 'replicate', 'circular']:
            filter_module = CombinedBandPassFilter(
                pha_bands=pha_bands,
                amp_bands=amp_bands,
                fs=fs,
                seq_len=seq_len,
                edge_mode=edge_mode
            )
            
            signal = torch.randn(1, 1, seq_len)
            filtered = filter_module(signal)
            
            # Should preserve signal length
            assert filtered.shape[-1] == seq_len
            
    def test_filter_frequency_response(self, setup_filter):
        """Test filter frequency response."""
        filter_module, fs, seq_len = setup_filter
        
        # Get first phase filter kernel
        kernel = filter_module.kernels[0]
        
        # Compute frequency response
        w, h = signal.freqz(kernel.cpu().numpy(), worN=8000)
        freq = w * fs / (2 * np.pi)
        
        # Find peak frequency
        peak_idx = np.argmax(np.abs(h))
        peak_freq = freq[peak_idx]
        
        # Should be within phase band (4-8 Hz)
        assert 4.0 <= peak_freq <= 8.0
        
    def test_batch_processing(self, setup_filter):
        """Test batch processing capability."""
        filter_module, _, seq_len = setup_filter
        
        # Create batch of signals
        batch_size = 16
        signal = torch.randn(batch_size, 1, seq_len)
        
        # Apply filter
        filtered = filter_module(signal)
        
        # Check output shape
        assert filtered.shape == (batch_size, 1, 4, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])