#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Modulation Index calculation.
"""

import pytest
import torch
import numpy as np

import sys
sys.path.append('../src')
from gpac._ModulationIndex import ModulationIndex


class TestModulationIndex:
    """Test Modulation Index functionality."""
    
    @pytest.fixture
    def setup_mi(self):
        """Setup MI module."""
        n_bins = 18
        mi_module = ModulationIndex(n_bins=n_bins)
        return mi_module, n_bins
    
    def test_mi_initialization(self, setup_mi):
        """Test MI initialization."""
        mi_module, n_bins = setup_mi
        
        assert mi_module.n_bins == n_bins
        assert hasattr(mi_module, 'bin_centers')
        assert len(mi_module.bin_centers) == n_bins
        
    def test_uniform_distribution(self, setup_mi):
        """Test MI for uniform phase distribution."""
        mi_module, n_bins = setup_mi
        
        # Create uniform phase distribution
        n_samples = 10000
        phase = torch.rand(1, 1, 1, 1, n_samples) * 2 * np.pi - np.pi
        amplitude = torch.ones_like(phase)
        
        # Calculate MI
        mi = mi_module(phase, amplitude)
        
        # MI should be close to 0 for uniform distribution
        assert mi.item() < 0.01
        
    def test_perfect_coupling(self, setup_mi):
        """Test MI for perfect phase-amplitude coupling."""
        mi_module, n_bins = setup_mi
        
        # Create perfect coupling: amplitude peaks at phase = 0
        n_samples = 10000
        phase = torch.linspace(-np.pi, np.pi, n_samples)
        phase = phase.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Amplitude follows cosine of phase (peaks at 0)
        amplitude = torch.cos(phase) + 1.0  # Keep positive
        
        # Calculate MI
        mi = mi_module(phase, amplitude)
        
        # MI should be high for perfect coupling
        assert mi.item() > 0.1
        
    def test_batch_processing(self, setup_mi):
        """Test batch processing of MI."""
        mi_module, _ = setup_mi
        
        batch_size = 4
        n_channels = 2
        n_pha_bands = 5
        n_amp_bands = 3
        n_segments = 10
        n_times = 1000
        
        phase = torch.rand(batch_size, n_channels, n_pha_bands, 
                          n_segments, n_times) * 2 * np.pi - np.pi
        amplitude = torch.rand(batch_size, n_channels, n_amp_bands, 
                              n_segments, n_times)
        
        # Calculate MI
        mi = mi_module(phase, amplitude)
        
        # Check output shape
        assert mi.shape == (batch_size, n_channels, n_pha_bands, n_amp_bands)
        
    def test_modulated_signal(self, setup_mi):
        """Test MI with known modulated signal."""
        mi_module, _ = setup_mi
        
        # Create signal with known PAC
        fs = 512
        duration = 4.0
        n_times = int(fs * duration)
        t = torch.linspace(0, duration, n_times)
        
        # Phase signal (6 Hz)
        phase_signal = torch.sin(2 * np.pi * 6 * t)
        
        # Extract phase
        phase = torch.atan2(phase_signal, 
                           torch.roll(phase_signal, 1, dims=-1))
        
        # Modulated amplitude
        modulation_depth = 0.8
        amplitude = 1 + modulation_depth * (torch.cos(phase) + 1) / 2
        
        # Reshape for MI calculation
        phase = phase.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        amplitude = amplitude.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Calculate MI
        mi = mi_module(phase, amplitude)
        
        # Should detect coupling
        assert mi.item() > 0.05
        
    def test_phase_range(self, setup_mi):
        """Test MI with different phase ranges."""
        mi_module, _ = setup_mi
        
        n_samples = 5000
        
        # Test with phase in [-pi, pi]
        phase1 = torch.rand(1, 1, 1, 1, n_samples) * 2 * np.pi - np.pi
        amplitude = torch.rand_like(phase1)
        mi1 = mi_module(phase1, amplitude)
        
        # Test with phase in [0, 2*pi] (should give same result)
        phase2 = torch.rand(1, 1, 1, 1, n_samples) * 2 * np.pi
        mi2 = mi_module(phase2, amplitude)
        
        # Both should be similar (low MI for random coupling)
        assert abs(mi1.item() - mi2.item()) < 0.01
        
    def test_amplitude_probability(self):
        """Test amplitude probability distribution mode."""
        n_bins = 18
        mi_module = ModulationIndex(n_bins=n_bins, amp_prob=True)
        
        # Create test data
        phase = torch.rand(1, 1, 1, 1, 1000) * 2 * np.pi - np.pi
        amplitude = torch.rand_like(phase)
        
        # Calculate amplitude probability
        amp_prob = mi_module(phase, amplitude)
        
        # Check output shape (should have extra bin dimension)
        assert amp_prob.shape == (1, 1, 1, 1, n_bins)
        
        # Probabilities should sum to 1
        assert torch.allclose(amp_prob.sum(dim=-1), torch.ones(1, 1, 1, 1))
        
    def test_surrogate_comparison(self, setup_mi):
        """Test MI calculation for surrogate data."""
        mi_module, _ = setup_mi
        
        # Original coupled signal
        n_times = 5000
        t = torch.linspace(0, 10, n_times)
        phase = torch.sin(2 * np.pi * 1 * t) * np.pi  # 1 Hz phase
        amplitude = 1 + 0.5 * torch.cos(phase)  # Coupled amplitude
        
        phase = phase.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        amplitude = amplitude.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Calculate MI for original
        mi_original = mi_module(phase, amplitude)
        
        # Create surrogate by shuffling amplitude
        amplitude_shuffled = amplitude[..., torch.randperm(n_times)]
        mi_surrogate = mi_module(phase, amplitude_shuffled)
        
        # Original should have higher MI than surrogate
        assert mi_original.item() > mi_surrogate.item() * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])