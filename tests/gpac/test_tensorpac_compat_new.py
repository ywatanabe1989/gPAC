#!/usr/bin/env python3
"""Test suite for the new TensorPAC compatibility module."""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gpac.tensorpac_compat import (
    calculate_pac_tensorpac_compat,
    TENSORPAC_CONFIGS,
    get_tensorpac_bands,
    compare_with_tensorpac
)


class TestTensorPACCompatModule:
    """Test the TensorPAC compatibility module."""
    
    def test_module_imports(self):
        """Test that all required functions can be imported."""
        assert calculate_pac_tensorpac_compat is not None
        assert TENSORPAC_CONFIGS is not None
        assert get_tensorpac_bands is not None
        assert compare_with_tensorpac is not None
    
    def test_available_configs(self):
        """Test that all expected configurations are available."""
        expected_configs = ['standard', 'medium', 'hres', 'compatible']
        for config in expected_configs:
            assert config in TENSORPAC_CONFIGS
            
        # Check compatible config (50x30)
        assert TENSORPAC_CONFIGS['compatible']['n_pha'] == 50
        assert TENSORPAC_CONFIGS['compatible']['n_amp'] == 30
    
    def test_basic_pac_calculation(self):
        """Test basic PAC calculation with compatibility scaling."""
        # Generate test signal
        fs = 1000.0
        t = np.arange(2000) / fs
        
        # Create PAC signal
        phase = np.sin(2 * np.pi * 5 * t)
        signal = (1 + 0.5 * phase) * np.sin(2 * np.pi * 70 * t)
        
        # Calculate with default config
        pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs)
        
        # Check output shapes
        assert pac.shape == (50, 30)  # Default is 'compatible' config
        assert len(f_pha) == 50
        assert len(f_amp) == 30
        
        # Check frequency ranges
        assert f_pha[0] > 1.0 and f_pha[-1] < 26.0
        assert f_amp[0] > 50.0 and f_amp[-1] < 181.0
        
        # Check that values are scaled up
        assert pac.max() > 0.01  # Should be scaled to TensorPAC range
    
    def test_different_configurations(self):
        """Test different preset configurations."""
        # Generate test signal
        fs = 1000.0
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t)
        
        configs_to_test = ['standard', 'medium', 'hres', 'compatible']
        
        for config in configs_to_test:
            pac, f_pha, f_amp = calculate_pac_tensorpac_compat(
                signal, fs, config=config
            )
            
            cfg = TENSORPAC_CONFIGS[config]
            assert pac.shape == (cfg['n_pha'], cfg['n_amp'])
            assert len(f_pha) == cfg['n_pha']
            assert len(f_amp) == cfg['n_amp']
    
    def test_unscaled_values(self):
        """Test getting both scaled and unscaled values."""
        # Generate test signal
        fs = 1000.0
        t = np.arange(2000) / fs
        signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 70 * t)
        
        # Get both scaled and unscaled
        pac_scaled, f_pha, f_amp, pac_raw = calculate_pac_tensorpac_compat(
            signal, fs, config='compatible', return_unscaled=True
        )
        
        # Check that scaling was applied
        scale_factor = TENSORPAC_CONFIGS['compatible']['scale_factor']
        assert np.allclose(pac_scaled, pac_raw * scale_factor)
        
        # Raw values should be smaller
        assert pac_raw.max() < pac_scaled.max()
    
    def test_custom_scale_factor(self):
        """Test using custom scale factor."""
        # Generate test signal
        fs = 1000.0
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        
        custom_scale = 20.0
        
        pac_custom, _, _, pac_raw = calculate_pac_tensorpac_compat(
            signal, fs, custom_scale=custom_scale, return_unscaled=True
        )
        
        # Check custom scaling
        assert np.allclose(pac_custom, pac_raw * custom_scale)
    
    def test_input_shapes(self):
        """Test various input shapes."""
        fs = 1000.0
        
        # Test 1D signal
        signal_1d = np.random.randn(1000)
        pac_1d, _, _ = calculate_pac_tensorpac_compat(signal_1d, fs, config='standard')
        assert pac_1d.shape == (10, 10)
        
        # Test 2D signal (epochs)
        signal_2d = np.random.randn(5, 1000)
        pac_2d, _, _ = calculate_pac_tensorpac_compat(signal_2d, fs, config='standard')
        assert pac_2d.shape == (10, 10)
        
        # Test torch tensor
        signal_torch = torch.randn(1000)
        pac_torch, _, _ = calculate_pac_tensorpac_compat(signal_torch, fs, config='standard')
        assert pac_torch.shape == (10, 10)
    
    def test_frequency_bands_generation(self):
        """Test frequency band generation."""
        pha_bands, amp_bands = get_tensorpac_bands(
            n_pha=10, n_amp=10,
            pha_range=(2, 20),
            amp_range=(60, 160)
        )
        
        assert pha_bands.shape == (10, 2)
        assert amp_bands.shape == (10, 2)
        
        # Check sequential bands
        for i in range(9):
            assert pha_bands[i, 1] == pha_bands[i+1, 0]
            assert amp_bands[i, 1] == amp_bands[i+1, 0]
    
    @pytest.mark.parametrize("config,expected_shape", [
        ('standard', (10, 10)),
        ('medium', (30, 30)),
        ('hres', (50, 50)),
        ('compatible', (50, 30))
    ])
    def test_config_shapes(self, config, expected_shape):
        """Test that each configuration produces expected output shape."""
        signal = np.random.randn(1000)
        pac, _, _ = calculate_pac_tensorpac_compat(signal, 1000.0, config=config)
        assert pac.shape == expected_shape
    
    def test_value_range(self):
        """Test that PAC values are in reasonable range after scaling."""
        # Generate strong PAC signal
        fs = 1000.0
        t = np.arange(2000) / fs
        phase = np.sin(2 * np.pi * 5 * t)
        signal = (1 + 0.9 * phase) * np.sin(2 * np.pi * 70 * t)
        
        pac, _, _ = calculate_pac_tensorpac_compat(signal, fs)
        
        # After scaling, values should be in TensorPAC-like range
        assert pac.min() >= 0.0
        assert pac.max() < 2.0  # MI theoretical max is 1, but with scaling could approach 2
    
    def test_compare_with_tensorpac(self):
        """Test the comparison function (when TensorPAC not available)."""
        signal = np.random.randn(1000)
        result = compare_with_tensorpac(signal, 1000.0)
        
        # Should either have comparison results or error message
        assert 'error' in result or 'tensorpac_max' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])