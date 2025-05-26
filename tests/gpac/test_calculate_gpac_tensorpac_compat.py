#!/usr/bin/env python3
"""Test suite for gPAC TensorPAC compatibility layer."""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gpac import calculate_pac

# Check if compatibility layer exists
try:
    from src.gpac._calculate_gpac_tensorpac_compat import calculate_pac_tensorpac_compat
    COMPAT_LAYER_AVAILABLE = True
except ImportError:
    COMPAT_LAYER_AVAILABLE = False


class TestTensorPACCompatibilityLayer:
    """Test the TensorPAC compatibility layer functionality."""
    
    @pytest.mark.skipif(not COMPAT_LAYER_AVAILABLE, 
                        reason="Compatibility layer not available")
    def test_compatibility_layer_exists(self):
        """Test that compatibility layer can be imported."""
        assert COMPAT_LAYER_AVAILABLE
    
    @pytest.mark.skipif(not COMPAT_LAYER_AVAILABLE,
                        reason="Compatibility layer not available")
    def test_compatibility_layer_scaling(self):
        """Test that compatibility layer applies proper scaling."""
        # Generate test signal
        fs = 256
        duration = 2
        t = np.linspace(0, duration, int(fs * duration), False)
        
        # Simple coupled signal
        phase_freq = 10
        amp_freq = 80
        signal = (np.sin(2 * np.pi * phase_freq * t) + 
                 0.5 * np.sin(2 * np.pi * amp_freq * t))
        signal_4d = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        
        # Standard gPAC
        pac_standard, _, _ = calculate_pac(
            signal_4d, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
        )
        
        # Compatibility layer
        result = calculate_pac_tensorpac_compat(
            signal_4d, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
        )
        
        # Handle both tuple and tensor return
        if isinstance(result, tuple):
            pac_compat = result[0]
        else:
            pac_compat = result
        
        # Compatibility layer should scale values up
        standard_max = pac_standard.cpu().numpy().max()
        compat_max = pac_compat.cpu().numpy().max()
        
        # Expected scaling factor is ~2.86x based on investigation
        assert compat_max > standard_max
        assert 2.0 < compat_max / standard_max < 4.0
    
    @pytest.mark.skipif(not COMPAT_LAYER_AVAILABLE,
                        reason="Compatibility layer not available")
    def test_compatibility_layer_clipping(self):
        """Test that compatibility layer clips values to reasonable range."""
        # Generate signal with very strong coupling
        fs = 256
        duration = 5
        t = np.linspace(0, duration, int(fs * duration), False)
        
        # Very strong coupling
        phase_freq = 10
        amp_freq = 80
        phase = np.sin(2 * np.pi * phase_freq * t)
        amp = (1 + 0.9 * phase) * np.sin(2 * np.pi * amp_freq * t)
        signal = phase + amp
        signal_4d = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        
        # Calculate with compatibility layer
        result = calculate_pac_tensorpac_compat(
            signal_4d, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
        )
        
        if isinstance(result, tuple):
            pac_compat = result[0]
        else:
            pac_compat = result
        
        # Values should be clipped to [0, 2] range (TensorPAC MI range)
        pac_np = pac_compat.cpu().numpy()
        assert pac_np.min() >= 0.0
        assert pac_np.max() <= 2.0


class TestModulationIndexCompatibility:
    """Test Modulation Index calculation compatibility."""
    
    def test_mi_value_range(self):
        """Test that MI values are in expected range."""
        # Generate test signal
        fs = 512
        duration = 3
        t = np.linspace(0, duration, int(fs * duration), False)
        
        # Create signal with PAC
        phase_freq = 6
        amp_freq = 100
        phase = np.sin(2 * np.pi * phase_freq * t)
        amp = (0.5 + 0.5 * phase) * np.sin(2 * np.pi * amp_freq * t)
        signal = phase + amp
        signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        
        # Calculate PAC
        pac_values, _, _ = calculate_pac(
            signal_torch, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=200, amp_n_bands=10
        )
        
        # Check value range
        pac_np = pac_values.cpu().numpy()
        
        # Standard gPAC should have values in reasonable range
        assert 0 <= pac_np.min()
        assert pac_np.max() < 1.0  # Typically much less than 1
    
    def test_mi_calculation_stability(self):
        """Test MI calculation with edge cases."""
        fs = 256
        
        # Test with different signal types
        test_cases = [
            # No coupling (independent signals)
            lambda t: np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t),
            # Perfect coupling
            lambda t: np.sin(2 * np.pi * 10 * t) * (1 + np.sin(2 * np.pi * 10 * t)),
            # Noisy signal
            lambda t: np.random.randn(len(t)) * 0.5,
        ]
        
        for i, signal_func in enumerate(test_cases):
            t = np.linspace(0, 2, int(fs * 2), False)
            signal = signal_func(t)
            signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
            
            # Should not raise errors
            pac_values, _, _ = calculate_pac(
                signal_torch, fs=fs,
                pha_start_hz=2, pha_end_hz=20, pha_n_bands=5,
                amp_start_hz=60, amp_end_hz=120, amp_n_bands=5
            )
            
            # Should return valid values (no NaN or Inf)
            pac_np = pac_values.cpu().numpy()
            assert not np.isnan(pac_np).any(), f"NaN values in test case {i}"
            assert not np.isinf(pac_np).any(), f"Inf values in test case {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])