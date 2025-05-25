#!/usr/bin/env python3
import sys
import os

# Add source directory to path
src_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, src_dir)

import pytest
import torch
import numpy as np

# Import directly from module
import gpac
calculate_pac = gpac.calculate_pac

# Try to import mngs for comparison tests
try:
    sys.path.insert(0, '/home/ywatanabe/proj/mngs_repo/src')
    import mngs
    MNGS_AVAILABLE = True
except ImportError:
    MNGS_AVAILABLE = False

# Try to import tensorpac for additional comparison
try:
    import tensorpac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False


class TestCalculatePAC:
    """Test the calculate_pac function for correct shape output."""
    
    def test_shape_output_basic(self):
        """Test that calculate_pac returns correct shape (B, C, F_pha, F_amp)."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test signal
        batch_size = 2
        n_channels = 3
        seq_len = 1000
        fs = 500
        pha_n_bands = 10
        amp_n_bands = 8
        
        signal = np.random.randn(batch_size, n_channels, seq_len)
        
        # Test calculate_pac
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal, fs, 
            pha_n_bands=pha_n_bands, 
            amp_n_bands=amp_n_bands
        )
        
        # Verify shapes
        expected_shape = (batch_size, n_channels, pha_n_bands, amp_n_bands)
        assert pac_values.shape == expected_shape, f"Expected {expected_shape}, got {pac_values.shape}"
        assert pha_freqs.shape == (pha_n_bands,), f"Expected ({pha_n_bands},), got {pha_freqs.shape}"
        assert amp_freqs.shape == (amp_n_bands,), f"Expected ({amp_n_bands},), got {amp_freqs.shape}"
        
    def test_shape_output_4d_input(self):
        """Test with 4D input (batch, channels, segments, time)."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test signal with segments
        batch_size = 1
        n_channels = 2
        n_segments = 5
        seq_len = 512
        fs = 500  # Higher sampling rate to avoid frequency range issues
        pha_n_bands = 8
        amp_n_bands = 6
        
        signal = np.random.randn(batch_size, n_channels, n_segments, seq_len)
        
        # Test calculate_pac
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal, fs,
            pha_n_bands=pha_n_bands,
            amp_n_bands=amp_n_bands
        )
        
        # Verify shapes - segments should be averaged out
        expected_shape = (batch_size, n_channels, pha_n_bands, amp_n_bands)
        assert pac_values.shape == expected_shape, f"Expected {expected_shape}, got {pac_values.shape}"
        assert pha_freqs.shape == (pha_n_bands,), f"Expected ({pha_n_bands},), got {pha_freqs.shape}"
        assert amp_freqs.shape == (amp_n_bands,), f"Expected ({amp_n_bands},), got {amp_freqs.shape}"
        
    def test_shape_consistency_different_params(self):
        """Test shape consistency with different parameter combinations."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        signal = np.random.randn(1, 1, 1024)
        fs = 512
        
        test_cases = [
            {"pha_n_bands": 10, "amp_n_bands": 8},
            {"pha_n_bands": 15, "amp_n_bands": 12},
            {"pha_n_bands": 5, "amp_n_bands": 5},
        ]
        
        for params in test_cases:
            pac_values, pha_freqs, amp_freqs = calculate_pac(
                signal, fs, **params
            )
            
            expected_shape = (1, 1, params["pha_n_bands"], params["amp_n_bands"])
            assert pac_values.shape == expected_shape, f"Failed for {params}: expected {expected_shape}, got {pac_values.shape}"
            assert pha_freqs.shape == (params["pha_n_bands"],)
            assert amp_freqs.shape == (params["amp_n_bands"],)
            
    def test_return_dist_shape(self):
        """Test shapes when return_dist=True."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        signal = np.random.randn(1, 2, 512)
        fs = 256
        pha_n_bands = 8
        amp_n_bands = 6
        n_perm = 10
        
        pac_values, surrogate_dist, pha_freqs, amp_freqs = calculate_pac(
            signal, fs,
            pha_n_bands=pha_n_bands,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            return_dist=True
        )
        
        # Verify PAC values shape
        expected_pac_shape = (1, 2, pha_n_bands, amp_n_bands)
        assert pac_values.shape == expected_pac_shape, f"PAC shape: expected {expected_pac_shape}, got {pac_values.shape}"
        
        # Verify surrogate distribution shape
        expected_surr_shape = (n_perm, 1, 2, pha_n_bands, amp_n_bands)
        assert surrogate_dist.shape == expected_surr_shape, f"Surrogate shape: expected {expected_surr_shape}, got {surrogate_dist.shape}"
        
        # Verify frequency arrays
        assert pha_freqs.shape == (pha_n_bands,)
        assert amp_freqs.shape == (amp_n_bands,)
        
    def test_values_are_finite(self):
        """Test that PAC values are finite (not NaN or infinite)."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        signal = np.random.randn(1, 1, 1000)
        fs = 500
        
        pac_values, _, _ = calculate_pac(signal, fs)
        
        # Convert to numpy if it's a tensor
        if isinstance(pac_values, torch.Tensor):
            pac_values = pac_values.detach().cpu().numpy()
            
        assert np.all(np.isfinite(pac_values)), "PAC values contain NaN or infinite values"
        assert np.all(pac_values >= 0), "PAC values should be non-negative"

    @pytest.mark.skipif(not MNGS_AVAILABLE, reason="mngs not available")
    def test_shape_comparison_with_mngs(self):
        """Test that gpac and mngs return the same shape."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate demo signal using mngs
        xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
        pha_n_bands = 10
        amp_n_bands = 8
        
        # Test with mngs
        pac_mngs, pha_mids_mngs, amp_mids_mngs = mngs.dsp.pac(
            xx[:1, :1], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
        )
        
        # Test with gpac
        pac_gpac, pha_mids_gpac, amp_mids_gpac = calculate_pac(
            xx[:1, :1], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
        )
        
        # Convert gpac result to numpy if it's a tensor
        if isinstance(pac_gpac, torch.Tensor):
            pac_gpac = pac_gpac.detach().cpu().numpy()
        if isinstance(pac_mngs, torch.Tensor):
            pac_mngs = pac_mngs.detach().cpu().numpy()
            
        # Extract the PAC values from gpac to match mngs format
        # gpac returns (1, 1, pha_n_bands, amp_n_bands), mngs returns (pha_n_bands, amp_n_bands)
        pac_gpac_extracted = pac_gpac[0, 0]  # Remove batch and channel dims
        
        # Test shapes match after extraction
        assert pac_mngs.shape == pac_gpac_extracted.shape, f"Shape mismatch: mngs {pac_mngs.shape} vs gpac {pac_gpac_extracted.shape}"
        assert pha_mids_mngs.shape == pha_mids_gpac.shape, f"Phase freqs shape mismatch: mngs {pha_mids_mngs.shape} vs gpac {pha_mids_gpac.shape}"
        assert amp_mids_mngs.shape == amp_mids_gpac.shape, f"Amp freqs shape mismatch: mngs {amp_mids_mngs.shape} vs gpac {amp_mids_gpac.shape}"
        
        # Verify expected shapes
        expected_mngs_shape = (pha_n_bands, amp_n_bands)
        expected_gpac_shape = (1, 1, pha_n_bands, amp_n_bands)
        assert pac_mngs.shape == expected_mngs_shape, f"mngs: Expected {expected_mngs_shape}, got {pac_mngs.shape}"
        assert pac_gpac.shape == expected_gpac_shape, f"gpac: Expected {expected_gpac_shape}, got {pac_gpac.shape}"
        
    @pytest.mark.skipif(not MNGS_AVAILABLE, reason="mngs not available") 
    def test_frequency_ranges_with_mngs(self):
        """Test that frequency ranges are similar between gpac and mngs."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate demo signal using mngs
        xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
        
        # Apply mngs frequency cap
        factor = 0.8
        amp_end_hz_cap = int(min(fs / 2 / (1 + factor) - 1, 160))
        
        pha_n_bands = 10
        amp_n_bands = 8
        
        # Test with mngs (automatically applies cap)
        _, pha_mids_mngs, amp_mids_mngs = mngs.dsp.pac(
            xx[:1, :1], fs, 
            pha_n_bands=pha_n_bands, 
            amp_n_bands=amp_n_bands,
            amp_end_hz=160  # Will be capped internally
        )
        
        # Test with gpac using the same cap
        _, pha_mids_gpac, amp_mids_gpac = calculate_pac(
            xx[:1, :1], fs,
            pha_n_bands=pha_n_bands,
            amp_n_bands=amp_n_bands, 
            amp_end_hz=amp_end_hz_cap
        )
        
        # Test frequency ranges are similar
        assert np.allclose(pha_mids_mngs, pha_mids_gpac, rtol=1e-2), "Phase frequency ranges differ"
        assert np.allclose(amp_mids_mngs, amp_mids_gpac, rtol=1e-2), "Amplitude frequency ranges differ"

    @pytest.mark.skipif(not (MNGS_AVAILABLE and TENSORPAC_AVAILABLE), reason="mngs and tensorpac not available")
    def test_shape_comparison_with_tensorpac(self):
        """Test that gpac, mngs, and tensorpac all return compatible shapes."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate demo signal using mngs
        xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
        t_sec = len(tt) / fs
        i_batch, i_ch = 0, 0
        
        # Test with mngs
        pac_mngs, pha_mids_mngs, amp_mids_mngs = mngs.dsp.pac(
            xx[:1, :1], fs, pha_n_bands=10, amp_n_bands=8
        )
        
        # Test with gpac using same cap as mngs
        factor = 0.8
        amp_end_hz_cap = int(min(fs / 2 / (1 + factor) - 1, 160))
        
        pac_gpac, pha_mids_gpac, amp_mids_gpac = calculate_pac(
            xx[:1, :1], fs, 
            pha_n_bands=10, 
            amp_n_bands=8,
            amp_end_hz=amp_end_hz_cap
        )
        
        # Test with tensorpac
        _, _, pha_mids_tp, amp_mids_tp, pac_tp = mngs.dsp.utils.pac.calc_pac_with_tensorpac(
            xx, fs, t_sec, i_batch=i_batch, i_ch=i_ch
        )
        
        # Convert tensors to numpy
        if isinstance(pac_gpac, torch.Tensor):
            pac_gpac = pac_gpac.detach().cpu().numpy()
        if isinstance(pac_mngs, torch.Tensor):
            pac_mngs = pac_mngs.detach().cpu().numpy()
            
        # Test shapes are compatible (tensorpac might have different exact dimensions)
        print(f"  mngs shape: {pac_mngs.shape}")
        print(f"  gpac shape: {pac_gpac.shape}")  
        print(f"  tensorpac shape: {pac_tp.shape}")
        
        # Test that mngs and gpac have the same shape
        assert pac_mngs.shape == pac_gpac.shape, f"mngs {pac_mngs.shape} vs gpac {pac_gpac.shape}"
        
        # Verify expected shape
        expected_shape = (1, 1, 10, 8)
        assert pac_mngs.shape == expected_shape, f"mngs: Expected {expected_shape}, got {pac_mngs.shape}"
        assert pac_gpac.shape == expected_shape, f"gpac: Expected {expected_shape}, got {pac_gpac.shape}"


if __name__ == "__main__":
    pytest.main([__file__])