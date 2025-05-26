#!/usr/bin/env python3
"""Test suite for TensorPAC compatibility and frequency band handling."""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gpac import calculate_pac
from scipy.stats import pearsonr

# Try to import TensorPAC
try:
    tensorpac_path = os.path.join(os.path.dirname(__file__), '../../tensorpac_source')
    if tensorpac_path not in sys.path:
        sys.path.insert(0, tensorpac_path)
    from tensorpac import Pac
    from tensorpac.utils import pac_vec
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="TensorPAC not available")


class TestTensorPACFrequencyHandling:
    """Test TensorPAC's frequency band handling quirks."""
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_string_frequency_override(self):
        """Test that TensorPAC string configs override explicit frequencies."""
        # Test string configurations
        f_pha_mres, f_amp_mres = pac_vec('mres', 'mres')
        
        # Check that 'mres' gives 30 bands
        assert len(f_pha_mres) == 30
        assert len(f_amp_mres) == 30
        
        # Check that frequency ranges are NOT the standard 2-20 Hz and 60-160 Hz
        assert f_pha_mres[0, 0] < 2.0  # Should start at 1.5 Hz
        assert f_pha_mres[-1, 1] > 20.0  # Should end at 25 Hz
        assert f_amp_mres[0, 0] < 60.0  # Should start at 52.5 Hz
        assert f_amp_mres[-1, 1] > 160.0  # Should end at 180 Hz
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_overlapping_bands(self):
        """Test that TensorPAC creates overlapping frequency bands."""
        f_pha, f_amp = pac_vec('mres', 'mres')
        
        # Check phase bands overlap (using f - f/4 to f + f/4)
        for i in range(len(f_pha) - 1):
            # Each band should overlap with the next
            assert f_pha[i, 1] > f_pha[i+1, 0], f"Phase bands {i} and {i+1} don't overlap"
        
        # Check amplitude bands overlap (using f - f/8 to f + f/8)
        for i in range(len(f_amp) - 1):
            assert f_amp[i, 1] > f_amp[i+1, 0], f"Amp bands {i} and {i+1} don't overlap"
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_explicit_vs_string_bands(self):
        """Test difference between explicit and string-based band specification."""
        # Create explicit sequential bands (gPAC style)
        n_bands = 10
        pha_vec = np.linspace(2, 20, n_bands + 1)
        amp_vec = np.linspace(60, 160, n_bands + 1)
        f_pha_explicit = np.c_[pha_vec[:-1], pha_vec[1:]]
        f_amp_explicit = np.c_[amp_vec[:-1], amp_vec[1:]]
        
        # Get string-based bands
        f_pha_string, f_amp_string = pac_vec('lres', 'lres')  # 'lres' gives 10 bands
        
        # Both should have 10 bands
        assert len(f_pha_explicit) == len(f_pha_string) == 10
        assert len(f_amp_explicit) == len(f_amp_string) == 10
        
        # But the actual frequency values should be different
        assert not np.allclose(f_pha_explicit, f_pha_string)
        assert not np.allclose(f_amp_explicit, f_amp_string)


class TestGPACTensorPACCompatibility:
    """Test compatibility between gPAC and TensorPAC implementations."""
    
    @pytest.fixture
    def test_signal(self):
        """Generate a test signal with known PAC."""
        fs = 256
        duration = 5
        t = np.linspace(0, duration, int(fs * duration), False)
        
        # Create coupled signal
        phase_freq = 10  # Hz
        amp_freq = 80    # Hz
        phase_signal = np.sin(2 * np.pi * phase_freq * t)
        amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * phase_freq * t)
        amp_signal = amp_mod * np.sin(2 * np.pi * amp_freq * t)
        signal = phase_signal + amp_signal
        
        return {
            'signal': signal,
            'fs': fs,
            'phase_freq': phase_freq,
            'amp_freq': amp_freq,
            't': t
        }
    
    def test_gpac_basic_calculation(self, test_signal):
        """Test basic gPAC calculation."""
        signal = test_signal['signal']
        fs = test_signal['fs']
        
        # Reshape for gPAC (batch, channels, segments, time)
        signal_4d = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        
        # Calculate PAC
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal_4d,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=10,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=10
        )
        
        # Check output shapes
        assert pac_values.shape == (1, 1, 10, 10)
        assert len(pha_freqs) == 10
        assert len(amp_freqs) == 10
        
        # Check that we get non-zero PAC values
        assert pac_values.max() > 0
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_explicit_band_matching(self, test_signal):
        """Test that explicit bands can be matched between implementations."""
        signal = test_signal['signal']
        fs = test_signal['fs']
        
        # Define matching frequency bands
        n_pha = 10
        n_amp = 10
        pha_edges = np.linspace(2, 20, n_pha + 1)
        amp_edges = np.linspace(60, 160, n_amp + 1)
        
        # gPAC calculation
        signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        pac_gpac, _, _ = calculate_pac(
            signal_torch,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=n_pha,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=n_amp
        )
        
        # TensorPAC calculation with explicit bands
        f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]
        f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]
        pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
        pac_tensorpac = pac_obj.filterfit(fs, signal.reshape(1, -1), n_perm=0)
        
        # Both should have same shape
        assert pac_gpac.squeeze().shape == pac_tensorpac.squeeze().shape
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_value_scale_difference(self, test_signal):
        """Test that gPAC values are smaller than TensorPAC values."""
        signal = test_signal['signal']
        fs = test_signal['fs']
        
        # Calculate with both implementations
        signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        pac_gpac, _, _ = calculate_pac(
            signal_torch, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
        )
        
        # TensorPAC with explicit bands
        pha_edges = np.linspace(2, 20, 11)
        amp_edges = np.linspace(60, 160, 11)
        f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]
        f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]
        pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
        pac_tensorpac = pac_obj.filterfit(fs, signal.reshape(1, -1), n_perm=0)
        
        # gPAC values should be smaller (documented ~4-5x difference)
        gpac_max = pac_gpac.cpu().numpy().max()
        tensorpac_max = pac_tensorpac.max()
        
        assert gpac_max < tensorpac_max
        assert 0.1 < gpac_max / tensorpac_max < 0.5  # Roughly 10-50% of TensorPAC values


class TestV01ModeCompatibility:
    """Test v01 mode for potentially better TensorPAC compatibility."""
    
    def test_v01_mode_available(self):
        """Test that v01 mode is available in calculate_pac."""
        # Generate simple test signal
        fs = 256
        signal = torch.randn(1, 1, 1, 1024)
        
        # Should not raise an error
        pac_v01, _, _ = calculate_pac(
            signal, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10,
            v01_mode=True
        )
        
        # Should return valid output
        assert pac_v01.shape == (1, 1, 10, 10)
    
    def test_v01_vs_standard_mode(self):
        """Compare v01 mode with standard mode."""
        fs = 256
        signal = torch.randn(1, 1, 1, 2048)
        
        # Standard mode
        pac_standard, _, _ = calculate_pac(
            signal, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10,
            v01_mode=False
        )
        
        # V01 mode
        pac_v01, _, _ = calculate_pac(
            signal, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10,
            v01_mode=True
        )
        
        # Results should be highly correlated
        corr = pearsonr(pac_standard.cpu().numpy().flatten(), 
                       pac_v01.cpu().numpy().flatten())[0]
        assert corr > 0.9  # Should be highly correlated
        
        # For random noise, they might be identical or very close
        # The main difference appears with structured signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])