#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Hilbert transform module.
"""

import pytest
import torch
import numpy as np
from scipy.signal import hilbert

import sys
sys.path.append('../src')
from gpac._Hilbert import Hilbert


class TestHilbertTransform:
    """Test Hilbert transform functionality."""
    
    @pytest.fixture
    def setup_hilbert(self):
        """Setup Hilbert transform module."""
        seq_len = 1024
        hilbert_module = Hilbert(seq_len=seq_len, dim=-1)
        return hilbert_module, seq_len
    
    def test_hilbert_initialization(self, setup_hilbert):
        """Test Hilbert transform initialization."""
        hilbert_module, seq_len = setup_hilbert
        
        # Check if initialized
        assert hilbert_module.initial_seq_len == seq_len
        assert hilbert_module.dim == -1
        
    def test_analytic_signal(self, setup_hilbert):
        """Test analytic signal generation."""
        hilbert_module, seq_len = setup_hilbert
        
        # Create test signal
        fs = 512
        t = torch.linspace(0, seq_len/fs, seq_len)
        freq = 10.0  # Hz
        signal = torch.cos(2 * np.pi * freq * t)
        signal = signal.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Apply Hilbert transform
        analytic = hilbert_module(signal)
        
        # Check output shape: (batch, channel, time, 2)
        assert analytic.shape == (1, 1, seq_len, 2)
        
        # Extract phase and amplitude
        phase = analytic[..., 0]
        amplitude = analytic[..., 1]
        
        # Amplitude should be approximately constant for pure sinusoid
        amp_std = amplitude.std()
        amp_mean = amplitude.mean()
        assert amp_std / amp_mean < 0.1  # Less than 10% variation
        
    def test_phase_extraction(self, setup_hilbert):
        """Test phase extraction accuracy."""
        hilbert_module, seq_len = setup_hilbert
        
        # Create signal with known phase
        fs = 512
        t = torch.linspace(0, seq_len/fs, seq_len)
        freq = 5.0  # Hz
        phase_offset = np.pi / 4  # 45 degrees
        signal = torch.cos(2 * np.pi * freq * t + phase_offset)
        signal = signal.unsqueeze(0).unsqueeze(0)
        
        # Apply Hilbert transform
        analytic = hilbert_module(signal)
        phase = analytic[..., 0]
        
        # Check phase unwrapping (using numpy since torch.unwrap may not be available)
        phase_unwrapped = np.unwrap(phase[0, 0].numpy())
        
        # Phase should increase linearly with slope = 2*pi*freq/fs
        expected_slope = 2 * np.pi * freq / fs
        actual_slope = (phase_unwrapped[-1] - phase_unwrapped[0]) / (seq_len - 1)
        
        # Allow 5% error
        assert abs(actual_slope - expected_slope) / expected_slope < 0.05
        
    def test_amplitude_extraction(self, setup_hilbert):
        """Test amplitude extraction accuracy."""
        hilbert_module, seq_len = setup_hilbert
        
        # Create amplitude modulated signal
        fs = 512
        t = torch.linspace(0, seq_len/fs, seq_len)
        carrier_freq = 50.0  # Hz
        mod_freq = 2.0  # Hz
        mod_depth = 0.5
        
        carrier = torch.cos(2 * np.pi * carrier_freq * t)
        modulation = 1 + mod_depth * torch.cos(2 * np.pi * mod_freq * t)
        signal = carrier * modulation
        signal = signal.unsqueeze(0).unsqueeze(0)
        
        # Apply Hilbert transform
        analytic = hilbert_module(signal)
        amplitude = analytic[..., 1]
        
        # Check if amplitude follows modulation
        expected_amp = modulation.unsqueeze(0).unsqueeze(0)
        correlation = torch.corrcoef(
            torch.stack([amplitude.flatten(), expected_amp.flatten()])
        )[0, 1]
        
        assert correlation > 0.95
        
    def test_batch_processing(self, setup_hilbert):
        """Test batch processing."""
        hilbert_module, seq_len = setup_hilbert
        
        # Create batch of signals
        batch_size = 8
        n_channels = 4
        signal = torch.randn(batch_size, n_channels, seq_len)
        
        # Apply Hilbert transform
        analytic = hilbert_module(signal)
        
        # Check output shape
        assert analytic.shape == (batch_size, n_channels, seq_len, 2)
        
    def test_multi_band_input(self, setup_hilbert):
        """Test with multi-band filtered input."""
        hilbert_module, seq_len = setup_hilbert
        
        # Simulate filtered signal with multiple bands
        batch_size = 2
        n_channels = 1
        n_bands = 10
        signal = torch.randn(batch_size, n_channels, n_bands, seq_len)
        
        # Apply Hilbert transform
        analytic = hilbert_module(signal)
        
        # Check output shape
        assert analytic.shape == (batch_size, n_channels, n_bands, seq_len, 2)
        
    def test_compare_with_scipy(self, setup_hilbert):
        """Compare with scipy implementation."""
        hilbert_module, seq_len = setup_hilbert
        
        # Create test signal
        signal_np = np.random.randn(seq_len)
        signal_torch = torch.tensor(signal_np, dtype=torch.float32)
        signal_torch = signal_torch.unsqueeze(0).unsqueeze(0)
        
        # Apply our Hilbert transform
        analytic_torch = hilbert_module(signal_torch)
        amplitude_torch = analytic_torch[0, 0, :, 1].numpy()
        
        # Apply scipy Hilbert transform
        analytic_scipy = hilbert(signal_np)
        amplitude_scipy = np.abs(analytic_scipy)
        
        # Compare amplitudes (phase comparison is tricky due to unwrapping)
        correlation = np.corrcoef(amplitude_torch, amplitude_scipy)[0, 1]
        assert correlation > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])