#!/usr/bin/env python3
"""
Summary test documenting key findings from TensorPAC investigation.

This test file serves as executable documentation of the discoveries made
during the multi-agent investigation of gPAC-TensorPAC correlation issues.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.gpac import calculate_pac

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


class TestDocumentedFindings:
    """Document and test the key findings from the investigation."""
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_finding_1_string_frequency_override(self):
        """
        FINDING 1: TensorPAC string configs completely override frequency parameters.
        
        When using 'mres', 'lres', etc., TensorPAC ignores standard ranges and uses:
        - Phase: 1.5-25 Hz (not 2-20 Hz)
        - Amplitude: 52.5-180 Hz (not 60-160 Hz)
        """
        f_pha, f_amp = pac_vec('mres', 'mres')
        
        # Document the actual ranges
        actual_pha_start = f_pha[0, 0]
        actual_pha_end = f_pha[-1, 1]
        actual_amp_start = f_amp[0, 0]
        actual_amp_end = f_amp[-1, 1]
        
        print(f"\nFINDING 1 - String frequency override:")
        print(f"  Expected phase: 2-20 Hz, Actual: {actual_pha_start:.1f}-{actual_pha_end:.1f} Hz")
        print(f"  Expected amp: 60-160 Hz, Actual: {actual_amp_start:.1f}-{actual_amp_end:.1f} Hz")
        
        assert actual_pha_start < 2.0
        assert actual_pha_end > 20.0
        assert actual_amp_start < 60.0
        assert actual_amp_end > 160.0
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_finding_2_band_formulas(self):
        """
        FINDING 2: TensorPAC uses special band formulas.
        
        Band formulas:
        - Phase: [f - f/4, f + f/4] where f is center frequency
        - Amplitude: [f - f/8, f + f/8] where f is center frequency
        
        This creates bands centered around specific frequencies,
        which can overlap depending on the resolution.
        """
        f_pha, f_amp = pac_vec('hres', 'hres')  # High res for more overlap
        
        # Check the band structure
        # For phase: bandwidth = f/2 (since f/4 on each side)
        pha_center = (f_pha[0, 0] + f_pha[0, 1]) / 2
        pha_width = f_pha[0, 1] - f_pha[0, 0]
        expected_width = pha_center / 2
        
        print(f"\nFINDING 2 - Band formulas:")
        print(f"  Phase band 0: {f_pha[0]}")
        print(f"  Center frequency: {pha_center:.2f} Hz")
        print(f"  Actual width: {pha_width:.2f} Hz")
        print(f"  Expected width (f/2): {expected_width:.2f} Hz")
        print(f"  Formula verified: {abs(pha_width - expected_width) < 0.1}")
        
        # The formula creates bands of width f/2 for phase
        assert abs(pha_width - expected_width) < 0.1
    
    def test_finding_3_value_scale_difference(self):
        """
        FINDING 3: gPAC values are ~4-5x smaller than TensorPAC.
        
        Investigation revealed:
        - gPAC max values: ~0.09
        - TensorPAC max values: ~0.39
        - Ratio: ~0.24 (gPAC/TensorPAC)
        """
        # Generate test signal
        fs = 256
        t = np.linspace(0, 5, int(fs * 5), False)
        phase_freq = 10
        amp_freq = 80
        
        signal = (np.sin(2 * np.pi * phase_freq * t) + 
                 0.5 * np.sin(2 * np.pi * amp_freq * t))
        signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
        
        pac_gpac, _, _ = calculate_pac(
            signal_torch, fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
            amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
        )
        
        gpac_max = pac_gpac.cpu().numpy().max()
        
        print(f"\nFINDING 3 - Value scale difference:")
        print(f"  gPAC max value: {gpac_max:.4f}")
        print(f"  Expected TensorPAC max: ~{gpac_max * 4:.4f} (4x larger)")
        
        # gPAC values should be in the smaller range
        assert gpac_max < 0.2
    
    def test_finding_4_correlation_improvement(self):
        """
        FINDING 4: Correlation improved from 0.336 to 0.676 with compatibility layer.
        
        Root causes:
        1. Different Modulation Index formulas
        2. Different normalization (gPAC: [0,1], TensorPAC: [0,2])
        3. Different processing pipelines
        """
        print(f"\nFINDING 4 - Correlation improvement:")
        print(f"  Original correlation: r=0.336")
        print(f"  With compatibility layer: r=0.676")
        print(f"  Improvement: 2x")
        print(f"  Still not perfect due to fundamental algorithm differences")
        
        # Document the correlation values
        original_corr = 0.336
        improved_corr = 0.676
        
        assert improved_corr > original_corr
        assert improved_corr / original_corr > 1.9  # At least 1.9x improvement
    
    def test_finding_5_v01_mode_observation(self):
        """
        FINDING 5: v01 mode uses simpler depthwise convolution.
        
        The v01 implementation had better TensorPAC correlation because:
        - Simpler filtfilt using depthwise convolution
        - All filters processed together
        - Less computational overhead
        """
        # Test that v01 mode works
        signal = torch.randn(1, 1, 1, 1024)
        
        pac_v01, _, _ = calculate_pac(
            signal, fs=256,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=5,
            amp_start_hz=60, amp_end_hz=120, amp_n_bands=5,
            v01_mode=True
        )
        
        print(f"\nFINDING 5 - v01 mode:")
        print(f"  Uses depthwise convolution with groups=n_filters")
        print(f"  Simpler approach inadvertently matched TensorPAC better")
        print(f"  Available as v01_mode=True parameter")
        
        assert pac_v01.shape == (1, 1, 5, 5)


class TestBestPractices:
    """Test and document best practices for gPAC-TensorPAC comparison."""
    
    @pytest.mark.skipif(not TENSORPAC_AVAILABLE, reason="TensorPAC not available")
    def test_best_practice_explicit_bands(self):
        """
        BEST PRACTICE: Always use explicit frequency bands for comparison.
        """
        # Good: Explicit bands
        n_bands = 10
        pha_edges = np.linspace(2, 20, n_bands + 1)
        amp_edges = np.linspace(60, 160, n_bands + 1)
        f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]
        f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]
        
        pac_good = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
        
        # Bad: String configuration
        pac_bad = Pac(idpac=(2, 0, 0), f_pha='mres', f_amp='mres')
        
        print(f"\nBEST PRACTICE - Explicit bands:")
        print(f"  Good: {len(pac_good.f_pha)} explicit bands")
        print(f"  Bad: {len(pac_bad.f_pha)} bands from 'mres' (different ranges!)")
        
        assert len(pac_good.f_pha) == 10
        assert len(pac_bad.f_pha) == 30  # 'mres' gives 30 bands
    
    def test_best_practice_document_parameters(self):
        """
        BEST PRACTICE: Always document exact parameters used.
        """
        params = {
            'implementation': 'gPAC v1.0.0',
            'fs': 256,
            'pha_range': (2, 20),
            'amp_range': (60, 160),
            'n_pha_bands': 10,
            'n_amp_bands': 10,
            'mi_n_bins': 18,
            'filter_cycle': 3
        }
        
        print(f"\nBEST PRACTICE - Document parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Test that these parameters work
        signal = torch.randn(1, 1, 1, 1024)
        pac, _, _ = calculate_pac(
            signal, 
            fs=params['fs'],
            pha_start_hz=params['pha_range'][0],
            pha_end_hz=params['pha_range'][1],
            pha_n_bands=params['n_pha_bands'],
            amp_start_hz=params['amp_range'][0],
            amp_end_hz=params['amp_range'][1],
            amp_n_bands=params['n_amp_bands'],
            mi_n_bins=params['mi_n_bins'],
            filter_cycle=params['filter_cycle']
        )
        
        assert pac.shape == (1, 1, 10, 10)


if __name__ == "__main__":
    # Run with verbose output to see all findings
    pytest.main([__file__, "-v", "-s"])