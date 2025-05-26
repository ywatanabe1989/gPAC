#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for PAC module.
Tests mirror the source code structure and functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from gpac._PAC import PAC
    from gpac._BandPassFilter import BandPassFilter
except ImportError:
    # Alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.gpac._PAC import PAC
    from src.gpac._BandPassFilter import BandPassFilter


class TestBandPassFilter:
    """Test BandPassFilter class."""
    
    @pytest.fixture
    def basic_filter_params(self):
        """Basic parameters for filter testing."""
        fs = 512
        seq_len = 1024
        pha_bands = torch.tensor([[2., 4.], [4., 8.], [8., 12.]])
        amp_bands = torch.tensor([[60., 80.], [80., 100.]])
        return fs, seq_len, pha_bands, amp_bands
    
    def test_filter_initialization(self, basic_filter_params):
        """Test filter initialization."""
        fs, seq_len, pha_bands, amp_bands = basic_filter_params
        
        filter_module = BandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=seq_len,
            cycle_pha=3,
            cycle_amp=6
        )
        
        # Check kernel creation
        assert hasattr(filter_module, 'kernels')
        assert filter_module.kernels.shape[0] == 5  # 3 phase + 2 amplitude
        
    
    def test_filter_forward_pass(self, basic_filter_params):
        """Test filter forward pass."""
        fs, seq_len, pha_bands, amp_bands = basic_filter_params
        
        filter_module = BandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=seq_len
        )
        
        # Create test signal
        batch_size = 2
        # n_channels = 3
        signal = torch.randn(batch_size, 1, seq_len)
        
        # Apply filters
        filtered_output = filter_module(signal)
        
        # Check output shape (combined phase and amplitude filters)
        expected_total_filters = 3 + 2  # 3 phase + 2 amplitude
        assert filtered_output.shape == (batch_size, 1, expected_total_filters, seq_len)
    
    def test_filtfilt_mode(self, basic_filter_params):
        """Test zero-phase filtering mode."""
        fs, seq_len, pha_bands, amp_bands = basic_filter_params
        
        # Standard mode
        filter_standard = BandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=seq_len,
            filtfilt_mode=False
        )
        
        # Filtfilt mode
        filter_filtfilt = BandPassFilter(
            pha_bands=pha_bands,
            amp_bands=amp_bands,
            fs=fs,
            seq_len=seq_len,
            filtfilt_mode=True
        )
        
        signal = torch.randn(1, 1, seq_len)
        
        # Both should produce valid outputs
        output1 = filter_standard(signal)
        output2 = filter_filtfilt(signal)
        
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()


class TestPAC:
    """Test main PAC module."""
    
    @pytest.fixture
    def basic_pac_params(self):
        """Basic PAC parameters."""
        return {
            'seq_len': 1024,
            'fs': 512,
            'pha_start_hz': 2.0,
            'pha_end_hz': 20.0,
            'pha_n_bands': 5,
            'amp_start_hz': 60.0,
            'amp_end_hz': 120.0,
            'amp_n_bands': 5
        }
    
    def test_pac_initialization(self, basic_pac_params):
        """Test PAC module initialization."""
        pac = PAC(**basic_pac_params)
        
        # Check attributes
        assert pac.seq_len == 1024
        assert pac.fs == 512
        assert pac._pha_n_bands == 5
        assert pac._amp_n_bands == 5
        
        # Check frequency centers
        assert pac.PHA_MIDS_HZ.shape == (5,)
        assert pac.AMP_MIDS_HZ.shape == (5,)
        
        # Check frequency ranges
        assert pac.PHA_MIDS_HZ[0] >= 2.0
        assert pac.PHA_MIDS_HZ[-1] <= 20.0
        assert pac.AMP_MIDS_HZ[0] >= 60.0
        assert pac.AMP_MIDS_HZ[-1] <= 120.0
    
    def test_pac_forward_basic(self, basic_pac_params):
        """Test basic PAC forward pass."""
        pac = PAC(**basic_pac_params)
        
        # Create test signal
        signal = torch.randn(2, 3, 1, 1024)  # (batch, channels, segments, time)
        
        # Compute PAC
        result = pac(signal)
        pac_values = result['mi']
        
        # Check output shape
        assert pac_values.shape == (2, 3, 5, 5)
        assert not torch.isnan(pac_values).any()
    
    def test_pac_with_permutation(self, basic_pac_params):
        """Test PAC with permutation testing."""
        pac_params = basic_pac_params.copy()
        pac_params['n_perm'] = 20
        
        pac = PAC(**pac_params)
        signal = torch.randn(1, 1, 1, 1024)
        
        # Compute PAC with permutation testing
        result = pac(signal)
        pac_values = result['mi']
        
        # Should return z-scored values
        assert pac_values.shape == (1, 1, 5, 5)
        
        # Check that permutation results are available
        assert 'mi_z' in result
        assert 'surrogate_mis' in result
        
        # Create strong PAC signal
        t = torch.linspace(0, 2, 1024)
        phase_signal = torch.sin(2 * np.pi * 6 * t)
        amp_signal = torch.sin(2 * np.pi * 80 * t)
        modulated = phase_signal + (1 + 0.8 * phase_signal) * amp_signal * 0.5
        strong_pac_signal = modulated.view(1, 1, 1, -1)
        
        result_strong = pac(strong_pac_signal)
        pac_strong = result_strong['mi']
        
        # Strong PAC should have higher values
        assert pac_strong.max() > pac_values.max()
    
    def test_pac_return_distribution(self, basic_pac_params):
        """Test returning surrogate distribution."""
        pac_params = basic_pac_params.copy()
        pac_params['n_perm'] = 10
        
        pac = PAC(**pac_params)
        signal = torch.randn(1, 2, 1, 1024)
        
        # Get PAC with distribution
        result = pac(signal)
        
        # Should return dictionary with surrogate data
        assert isinstance(result, dict)
        assert 'mi' in result
        assert 'mi_z' in result
        assert 'surrogate_mis' in result
        
        pac_values = result['mi']
        surrogate_dist = result['surrogate_mis']
        
        assert pac_values.shape == (1, 2, 5, 5)
        assert surrogate_dist.shape == (10, 1, 2, 5, 5)
    
    @pytest.mark.skip(reason="Trainable mode needs DifferenciableBandPassFilter fixes")
    def test_trainable_mode(self, basic_pac_params):
        """Test trainable filter mode."""
        pac_params = basic_pac_params.copy()
        pac_params['trainable'] = True
        
        pac = PAC(**pac_params)
        
        # Check that parameters are trainable
        trainable_params = list(pac.parameters())
        assert len(trainable_params) > 0
        
        # Test gradient flow
        signal = torch.randn(1, 1, 1, 1024, requires_grad=True)
        result = pac(signal)
        pac_values = result['mi']
        
        loss = pac_values.sum()
        loss.backward()
        
        # Gradients should flow
        assert signal.grad is not None
        for param in trainable_params:
            assert param.grad is not None
    
    def test_fp16_mode(self, basic_pac_params):
        """Test half-precision mode."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for fp16 test")
        
        pac_params = basic_pac_params.copy()
        pac_params['fp16'] = True
        
        pac = PAC(**pac_params).cuda()
        signal = torch.randn(1, 1, 1, 1024).cuda()
        
        # Should handle fp16
        result = pac(signal)
        pac_values = result['mi']
        assert pac_values.dtype == torch.float16
        assert not torch.isnan(pac_values).any()
    
    def test_amplitude_probability_mode(self, basic_pac_params):
        """Test amplitude probability mode."""
        pac = PAC(**basic_pac_params)
        signal = torch.randn(1, 1, 1, 1024)
        
        result = pac(signal)
        
        # Check that amplitude probability distribution is returned
        assert 'amp_prob' in result
        amp_prob = result['amp_prob']
        
        # Probabilities should sum to 1 across bins
        prob_sums = amp_prob.sum(dim=-1)  # Sum across bins
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
    
    def test_edge_mode_handling(self, basic_pac_params):
        """Test edge mode for filtering."""
        pac_params = basic_pac_params.copy()
        pac_params['edge_mode'] = 'reflect'
        
        pac = PAC(**pac_params)
        signal = torch.randn(1, 1, 1, 1024)
        
        # Should work with edge mode
        result = pac(signal)
        pac_values = result['mi']
        assert not torch.isnan(pac_values).any()


class TestPACIntegration:
    """Integration tests for PAC module."""
    
    def test_known_pac_signal(self):
        """Test with synthetic signal containing known PAC."""
        fs = 1000
        duration = 2.0
        t = torch.linspace(0, duration, int(fs * duration))
        
        # Create PAC: 5 Hz phase modulates 70 Hz amplitude
        pha_freq = 5.0
        amp_freq = 70.0
        
        phase = torch.sin(2 * np.pi * pha_freq * t)
        amplitude = torch.sin(2 * np.pi * amp_freq * t)
        modulation = 1 + 0.8 * phase
        
        signal = phase * 0.3 + modulation * amplitude * 0.5
        signal = signal.view(1, 1, 1, -1)
        
        # Compute PAC
        pac = PAC(
            seq_len=len(t),
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=10.0,
            pha_n_bands=10,
            amp_start_hz=50.0,
            amp_end_hz=100.0,
            amp_n_bands=10
        )
        
        result = pac(signal)
        pac_values = result['mi']
        
        # Find peak
        pac_np = pac_values.squeeze().numpy()
        peak_idx = np.unravel_index(pac_np.argmax(), pac_np.shape)
        
        pha_freqs = pac.PHA_MIDS_HZ.numpy()
        amp_freqs = pac.AMP_MIDS_HZ.numpy()
        
        peak_pha = pha_freqs[peak_idx[0]]
        peak_amp = amp_freqs[peak_idx[1]]
        
        # Peak should be close to true frequencies
        assert abs(peak_pha - pha_freq) < 2.0
        assert abs(peak_amp - amp_freq) < 10.0
    
    def test_no_pac_signal(self):
        """Test with signal containing no PAC."""
        pac = PAC(
            seq_len=1024,
            fs=512,
            pha_n_bands=5,
            amp_n_bands=5
        )
        
        # Pure noise
        noise = torch.randn(1, 1, 1, 1024)
        result_noise = pac(noise)
        pac_noise = result_noise['mi']
        
        # PAC values should be low
        assert pac_noise.max() < 0.2
        
        # Unmodulated sinusoids
        t = torch.linspace(0, 2, 1024)
        signal = torch.sin(2 * np.pi * 5 * t) + torch.sin(2 * np.pi * 80 * t)
        signal = signal.view(1, 1, 1, -1)
        
        result_unmod = pac(signal)
        pac_unmod = result_unmod['mi']
        
        # Should have low PAC
        assert pac_unmod.max() < 0.3

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_PAC.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-25 22:26:08 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_PAC.py
# # ----------------------------------------
# import os
# 
# __FILE__ = "./src/gpac/_PAC.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# PyTorch Module for calculating Phase-Amplitude Coupling (PAC).
# 
# This implementation uses TensorPAC-compatible filter design for
# better comparability with existing neuroscience tools.
# """
# 
# import warnings
# from typing import Dict, Optional
# 
# import torch
# import torch.nn as nn
# 
# from ._BandPassFilter import BandPassFilter
# from ._DifferenciableBandPassFilter import DifferenciableBandPassFilter
# from ._Hilbert import Hilbert
# from ._ModulationIndex import ModulationIndex
# from ._utils import ensure_4d_input
# 
# 
# class PAC(nn.Module):
#     """
#     PyTorch Module for calculating Phase-Amplitude Coupling (PAC).
#     This implementation uses TensorPAC-compatible filter design with:
#     - Cycle parameters: (3, 6) for phase and amplitude
#     - FIR filter design matching TensorPAC's approach
#     - Optimized for GPU acceleration
# 
#     Parameters
#     ----------
#     seq_len : int
#         Length of the input signal
#     fs : float
#         Sampling frequency in Hz
#     pha_start_hz : float, optional
#         Start frequency for phase bands (default: 2.0)
#     pha_end_hz : float, optional
#         End frequency for phase bands (default: 20.0)
#     pha_n_bands : int, optional
#         Number of phase frequency bands (default: 50)
#     amp_start_hz : float, optional
#         Start frequency for amplitude bands (default: 60.0)
#     amp_end_hz : float, optional
#         End frequency for amplitude bands (default: 160.0)
#     amp_n_bands : int, optional
#         Number of amplitude frequency bands (default: 30)
#     n_perm : int or None, optional
#         Number of permutations for significance testing. When provided,
#         surrogate distributions are always returned (default: None)
#     trainable : bool, optional
#         Whether filters are trainable (default: False, not currently supported)
#     fp16 : bool, optional
#         Use half precision for faster computation (default: False)
#     mi_n_bins : int, optional
#         Number of bins for MI calculation (default: 18)
#     filter_cycle_pha : int, optional
#         Number of cycles for phase filter (default: 3)
#     filter_cycle_amp : int, optional
#         Number of cycles for amplitude filter (default: 6)
#     filtfilt_mode : bool, optional
#         Use sequential zero-phase filtering (like scipy.signal.filtfilt) for exact
#         TensorPAC compatibility (default: False)
#     edge_mode : str or None, optional
#         Edge padding mode for filtering. Options: 'reflect', 'replicate', 'circular', None
#         (default: None). 'reflect' matches scipy.signal.filtfilt behavior.
# 
#     Examples
#     --------
#     Basic PAC calculation:
# 
#     >>> import torch
#     >>> from gpac import PAC
#     >>> 
#     >>> # Create PAC model
#     >>> pac = PAC(seq_len=1024, fs=512, pha_n_bands=10, amp_n_bands=10)
#     >>> 
#     >>> # Generate test signal
#     >>> signal = torch.randn(1, 1, 1, 1024)  # (batch, channels, segments, time)
#     >>> 
#     >>> # Calculate PAC
#     >>> result = pac(signal)
#     >>> pac_values = result['mi']  # Modulation Index values
#     >>> print(f"PAC shape: {pac_values.shape}")  # (1, 1, 10, 10)
# 
#     With permutation testing:
# 
#     >>> pac_perm = PAC(seq_len=1024, fs=512, pha_n_bands=5, amp_n_bands=5, n_perm=100)
#     >>> result = pac_perm(signal)
#     >>> pac_z = result['mi_z']  # Z-scored PAC values
#     >>> surrogate_dist = result['surrogate_mis']  # Surrogate distribution
# 
#     Creating synthetic PAC signal:
# 
#     >>> import numpy as np
#     >>> fs = 512
#     >>> t = torch.linspace(0, 2, 1024)
#     >>> phase_sig = torch.sin(2 * np.pi * 6 * t)  # 6 Hz phase
#     >>> amp_sig = torch.sin(2 * np.pi * 80 * t)   # 80 Hz amplitude
#     >>> modulated = phase_sig + (1 + 0.8 * phase_sig) * amp_sig * 0.5
#     >>> signal = modulated.view(1, 1, 1, -1)
#     >>> 
#     >>> pac = PAC(seq_len=1024, fs=512)
#     >>> result = pac(signal)
#     >>> print(f"Max PAC: {result['mi'].max():.3f}")
#     """
# 
#     def __init__(
#         self,
#         seq_len: int,
#         fs: float,
#         pha_start_hz: float = 2.0,
#         pha_end_hz: float = 20.0,
#         pha_n_bands: int = 50,
#         amp_start_hz: float = 60.0,
#         amp_end_hz: float = 160.0,
#         amp_n_bands: int = 30,
#         n_perm: Optional[int] = None,
#         trainable: bool = False,
#         fp16: bool = False,
#         mi_n_bins: int = 18,
#         filter_cycle_pha: int = 3,
#         filter_cycle_amp: int = 6,
#         filtfilt_mode: bool = False,
#         edge_mode: Optional[str] = None,
#     ):
#         super().__init__()
# 
#         # Store configuration
#         self.seq_len = seq_len
#         self.fs = fs
#         self.fp16 = fp16
#         self.trainable = trainable
#         self.filter_cycle_pha = filter_cycle_pha
#         self.filter_cycle_amp = filter_cycle_amp
#         self.mi_n_bins = mi_n_bins
#         self.filtfilt_mode = filtfilt_mode
#         self.edge_mode = edge_mode
# 
#         # Validate and store permutation setting
#         self.n_perm = None
#         if n_perm is not None:
#             if not isinstance(n_perm, int) or n_perm < 1:
#                 raise ValueError("n_perm must be a positive integer or None.")
#             self.n_perm = n_perm
# 
#         # Initialize core components
#         self.bandpass_filter = self._init_bandpass(
#             seq_len,
#             fs,
#             pha_start_hz,
#             pha_end_hz,
#             pha_n_bands,
#             amp_start_hz,
#             amp_end_hz,
#             amp_n_bands,
#             trainable,
#             fp16,
#         )
#         self.hilbert = Hilbert(seq_len=seq_len, dim=-1, fp16=fp16)
#         self.modulation_index = ModulationIndex(
#             n_bins=mi_n_bins,
#             fp16=fp16,
#         )
# 
#         # Store frequency information
#         self.PHA_MIDS_HZ: torch.Tensor
#         self.AMP_MIDS_HZ: torch.Tensor
#         # Frequency info is set within _init_bandpass
# 
#         # Store band counts
#         self._pha_n_bands = pha_n_bands
#         self._amp_n_bands = amp_n_bands
# 
#     def _init_bandpass(
#         self,
#         seq_len: int,
#         fs: float,
#         pha_start_hz: float,
#         pha_end_hz: float,
#         pha_n_bands: int,
#         amp_start_hz: float,
#         amp_end_hz: float,
#         amp_n_bands: int,
#         trainable: bool,
#         fp16: bool,
#     ) -> nn.Module:
#         """Initialize bandpass filters with TensorPAC compatibility."""
#         # Create frequency bands
#         pha_bands = torch.stack(
#             [
#                 torch.linspace(pha_start_hz, pha_end_hz, pha_n_bands + 1)[:-1],
#                 torch.linspace(pha_start_hz, pha_end_hz, pha_n_bands + 1)[1:],
#             ],
#             dim=1,
#         )
#         amp_bands = torch.stack(
#             [
#                 torch.linspace(amp_start_hz, amp_end_hz, amp_n_bands + 1)[:-1],
#                 torch.linspace(amp_start_hz, amp_end_hz, amp_n_bands + 1)[1:],
#             ],
#             dim=1,
#         )
# 
#         # Store frequency centers
#         pha_mids = pha_bands.mean(dim=1)
#         amp_mids = amp_bands.mean(dim=1)
#         self.register_buffer("PHA_MIDS_HZ", pha_mids, persistent=False)
#         self.register_buffer("AMP_MIDS_HZ", amp_mids, persistent=False)
# 
#         if trainable:
#             if not _DIFFERENTIABLE_AVAILABLE:
#                 raise ImportError(
#                     f"Trainable mode requires DifferenciableBandPassFilter but import failed: {_DIFFERENTIABLE_IMPORT_ERROR}"
#                 )
#             return DifferenciableBandPassFilter(
#                 sig_len=seq_len,
#                 fs=fs,
#                 pha_low_hz=pha_start_hz,
#                 pha_high_hz=pha_end_hz,
#                 pha_n_bands=pha_n_bands,
#                 amp_low_hz=amp_start_hz,
#                 amp_high_hz=amp_end_hz,
#                 amp_n_bands=amp_n_bands,
#                 cycle=self.filter_cycle_pha,
#                 fp16=fp16,
#             )
#         else:
#             return BandPassFilter(
#                 pha_bands=pha_bands,
#                 amp_bands=amp_bands,
#                 fs=fs,
#                 seq_len=seq_len,
#                 fp16=fp16,
#                 cycle_pha=self.filter_cycle_pha,
#                 cycle_amp=self.filter_cycle_amp,
#                 filtfilt_mode=self.filtfilt_mode,
#                 edge_mode=self.edge_mode,
#             )
# 
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Performs the full PAC calculation pipeline.
# 
#         Args:
#             x: Input signal tensor with shape (B, C, Seg, Time)
# 
#         Returns:
#             dict: Dictionary containing:
#                 - 'mi': Modulation Index values (B, C, F_pha, F_amp)
#                 - 'amp_prob': Amplitude probability distribution (B, C, F_pha, F_amp, n_bins)
#                 - 'pha_bin_centers': Phase bin center values
#                 - 'pha_freqs_hz': Phase frequency centers
#                 - 'amp_freqs_hz': Amplitude frequency centers
#                 - 'mi_z': Z-scored MI values (only if n_perm is not None)
#                 - 'surrogate_mis': Surrogate MI distributions (only if n_perm is not None)
#         """
# 
#         # Input preparation
#         x = ensure_4d_input(x)
#         batch_size, n_chs, n_segments, current_seq_len = x.shape
#         device = x.device
#         target_dtype = torch.float16 if self.fp16 else torch.float32
# 
#         if current_seq_len != self.seq_len:
#             warnings.warn(
#                 f"Input length {current_seq_len} != init length {self.seq_len}. Results may be suboptimal."
#             )
# 
#         x = x.to(target_dtype)
# 
#         # Set gradient tracking context
#         grad_context = (
#             torch.enable_grad() if self.trainable else torch.no_grad()
#         )
#         with grad_context:
# 
#             # Bandpass filtering
#             # Reshape: (B, C, Seg, Time) -> (B * C * Seg, 1, Time)
#             x_flat = x.reshape(-1, 1, current_seq_len)
#             # Apply filter: (B * C * Seg, 1, N_filters, Time)
#             x_filt_flat = self.bandpass_filter(x_flat)
#             # Reshape back: (B, C, Seg, N_filters, Time)
#             x_filt = x_filt_flat.view(
#                 batch_size, n_chs, n_segments, -1, current_seq_len
#             )
# 
#             # Hilbert transform
#             # Output: (B, C, Seg, N_filters, Time, 2=[Phase, Amp])
#             x_analytic = self.hilbert(x_filt)
# 
#             # Extract phase and amplitude bands
#             # Phase: (B, C, Seg, n_pha_bands, Time)
#             pha = x_analytic[..., : self._pha_n_bands, :, 0]
#             # Amplitude: (B, C, Seg, n_amp_bands, Time)
#             amp = x_analytic[..., self._pha_n_bands :, :, 1]
# 
#             # Permute for Modulation Index: (B, C, Freq, Seg, Time)
#             pha = pha.permute(0, 1, 3, 2, 4)
#             amp = amp.permute(0, 1, 3, 2, 4)
# 
#             # Calculate observed Modulation index
#             pac_result = self.modulation_index(pha, amp)
# 
#             # Build result dictionary
#             result = pac_result
#             result.update(
#                 {
#                     "pha_freqs_hz": self.PHA_MIDS_HZ,
#                     "amp_freqs_hz": self.AMP_MIDS_HZ,
#                 }
#             )
# 
#             # Permutation test
#             if self.n_perm is not None:
#                 observed_mi = pac_result["mi"]
#                 surrogate_mis = self._generate_surrogates_with_grad(
#                     pha, amp, device, target_dtype
#                 )
#                 mean_surr = surrogate_mis.mean(dim=0)
#                 std_surr = surrogate_mis.std(dim=0)
#                 mi_z = (observed_mi - mean_surr) / (std_surr + 1e-9)
#                 mask = torch.isfinite(mi_z)
#                 mi_z = torch.where(mask, mi_z, torch.zeros_like(mi_z))
#                 result["mi_z"] = mi_z
#                 result["surrogate_mis"] = surrogate_mis
# 
#             return result
# 
#     def _generate_surrogates_with_grad(
#         self, pha: torch.Tensor, amp: torch.Tensor, device, dtype
#     ) -> torch.Tensor:
#         batch_size, n_chs, n_amp_bands, n_segments, time_core = amp.shape
# 
#         if time_core <= 1:
#             warnings.warn("Cannot generate surrogates: sequence length <= 1.")
#             dummy_shape = pha.shape[:2] + (pha.shape[2], amp.shape[2])
#             return torch.zeros(
#                 (self.n_perm,) + dummy_shape, dtype=dtype, device=device
#             )
# 
#         surrogate_results = []
#         indices = torch.arange(time_core, device=device)
# 
#         for _ in range(self.n_perm):
#             shift_shape = (batch_size, n_chs, n_amp_bands, n_segments)
#             shifts = torch.randint(
#                 1, time_core, size=shift_shape, device=device
#             )
#             shifted_indices = (
#                 indices.view(1, 1, 1, 1, -1) - shifts.unsqueeze(-1)
#             ) % time_core
#             amp_shifted = torch.gather(amp, dim=-1, index=shifted_indices)
#             surrogate_pac = self.modulation_index(pha, amp_shifted)["mi"]
#             surrogate_results.append(surrogate_pac)
# 
#         return torch.stack(surrogate_results, dim=0)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_PAC.py
# --------------------------------------------------------------------------------
