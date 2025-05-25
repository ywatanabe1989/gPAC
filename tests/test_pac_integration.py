#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for complete PAC pipeline.
"""

import pytest
import torch
import numpy as np

import sys
sys.path.append('../src')
import gpac


class TestPACIntegration:
    """Test complete PAC pipeline."""
    
    @pytest.fixture
    def setup_pac(self):
        """Setup PAC module."""
        fs = 512
        seq_len = 2048
        pac_model = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=10,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=10
        )
        return pac_model, fs, seq_len
    
    def test_pac_initialization(self, setup_pac):
        """Test PAC module initialization."""
        pac_model, fs, seq_len = setup_pac
        
        assert pac_model.seq_len == seq_len
        assert pac_model.fs == fs
        assert pac_model._pha_n_bands == 10
        assert pac_model._amp_n_bands == 10
        
    def test_synthetic_pac_signal(self, setup_pac):
        """Test PAC detection on synthetic signal."""
        pac_model, fs, seq_len = setup_pac
        
        # Create synthetic PAC signal
        t = np.linspace(0, seq_len/fs, seq_len)
        
        # Phase signal (6 Hz)
        phase_freq = 6.0
        phase_signal = np.sin(2 * np.pi * phase_freq * t)
        
        # Amplitude signal (80 Hz)
        amp_freq = 80.0
        amp_signal = np.sin(2 * np.pi * amp_freq * t)
        
        # Modulate amplitude with phase
        modulation_depth = 0.7
        modulated_amp = amp_signal * (1 + modulation_depth * phase_signal)
        
        # Combine signals
        signal = 0.3 * phase_signal + modulated_amp
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, time)
        
        # Calculate PAC
        pac_values = pac_model(signal)
        
        # Check output shape
        assert pac_values.shape == (1, 1, 10, 10)
        
        # Find peak
        pac_values_np = pac_values.squeeze().numpy()
        peak_idx = np.unravel_index(pac_values_np.argmax(), pac_values_np.shape)
        
        # Peak should be near true frequencies
        pha_freqs = pac_model.PHA_MIDS_HZ.numpy()
        amp_freqs = pac_model.AMP_MIDS_HZ.numpy()
        
        peak_pha_freq = pha_freqs[peak_idx[0]]
        peak_amp_freq = amp_freqs[peak_idx[1]]
        
        assert abs(peak_pha_freq - phase_freq) < 5.0  # Within 5 Hz
        assert abs(peak_amp_freq - amp_freq) < 30.0  # Within 30 Hz (due to frequency binning)
        
    def test_noise_signal(self, setup_pac):
        """Test PAC on pure noise (should be low)."""
        pac_model, _, seq_len = setup_pac
        
        # Create noise signal
        signal = torch.randn(1, 1, 1, seq_len)
        
        # Calculate PAC
        pac_values = pac_model(signal)
        
        # PAC should be low for noise
        assert pac_values.max().item() < 0.1
        
    def test_batch_processing(self, setup_pac):
        """Test batch processing capability."""
        pac_model, _, seq_len = setup_pac
        
        batch_size = 4
        n_channels = 2
        n_segments = 3
        
        signal = torch.randn(batch_size, n_channels, n_segments, seq_len)
        
        # Calculate PAC
        pac_values = pac_model(signal)
        
        # Check output shape
        assert pac_values.shape == (batch_size, n_channels, 10, 10)
        
    def test_filtfilt_mode(self):
        """Test PAC with filtfilt mode."""
        fs = 256
        seq_len = 1024
        
        pac_standard = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filtfilt_mode=False
        )
        
        pac_filtfilt = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filtfilt_mode=True
        )
        
        # Test signal
        signal = torch.randn(1, 1, 1, seq_len)
        
        # Calculate PAC with both modes
        pac_values_standard = pac_standard(signal)
        pac_values_filtfilt = pac_filtfilt(signal)
        
        # Both should produce valid results
        assert pac_values_standard.shape == pac_values_filtfilt.shape
        
        # Results should be correlated but not identical
        corr = torch.corrcoef(
            torch.stack([
                pac_values_standard.flatten(),
                pac_values_filtfilt.flatten()
            ])
        )[0, 1]
        
        assert 0.5 < corr < 0.99  # Correlated but different
        
    def test_edge_mode(self):
        """Test PAC with edge padding."""
        fs = 256
        seq_len = 1024
        
        pac_no_edge = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            edge_mode=None
        )
        
        pac_reflect = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            edge_mode='reflect'
        )
        
        signal = torch.randn(1, 1, 1, seq_len)
        
        # Both should work
        pac_values_no_edge = pac_no_edge(signal)
        pac_values_reflect = pac_reflect(signal)
        
        assert pac_values_no_edge.shape == pac_values_reflect.shape
        
    def test_permutation_testing(self):
        """Test PAC with permutation testing."""
        fs = 256
        seq_len = 512
        
        pac_model = gpac.PAC(
            seq_len=seq_len,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            n_perm=20,  # Small number for testing
            return_dist=True
        )
        
        signal = torch.randn(1, 1, 1, seq_len)
        
        # Calculate PAC with permutation
        pac_zscore, surrogate_dist = pac_model(signal)
        
        # Check shapes
        assert pac_zscore.shape == (1, 1, 5, 5)
        assert surrogate_dist.shape == (20, 1, 1, 5, 5)
        
    def test_calculate_pac_function(self):
        """Test the high-level calculate_pac function."""
        fs = 256
        seq_len = 1024
        
        signal = torch.randn(2, 2, seq_len)  # (batch, channel, time)
        
        # Use calculate_pac function
        pac_values = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5
        )
        
        # Check output
        assert pac_values.shape == (2, 2, 5, 5)
        
    def test_device_compatibility(self, setup_pac):
        """Test GPU/CPU compatibility."""
        pac_model, _, seq_len = setup_pac
        
        signal = torch.randn(1, 1, 1, seq_len)
        
        # CPU computation
        pac_cpu = pac_model(signal)
        
        # GPU computation if available
        if torch.cuda.is_available():
            pac_model_gpu = pac_model.cuda()
            signal_gpu = signal.cuda()
            pac_gpu = pac_model_gpu(signal_gpu)
            
            # Results should be very similar
            diff = (pac_cpu - pac_gpu.cpu()).abs().max()
            assert diff < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])