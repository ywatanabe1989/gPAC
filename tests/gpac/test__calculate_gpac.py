#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for calculate_pac function.
Tests mirror the source code structure and functionality.
"""

import pytest
import torch
import numpy as np
import warnings
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from gpac import calculate_pac
except ImportError:
    # Alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.gpac import calculate_pac


class TestCalculatePAC:
    """Test suite for calculate_pac function."""
    
    @pytest.fixture
    def basic_signal(self) -> Tuple[np.ndarray, float]:
        """Create a basic test signal with known PAC."""
        fs = 1000  # Hz
        duration = 2.0  # seconds
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create PAC signal: 6 Hz phase modulates 80 Hz amplitude
        pha_freq = 6.0
        amp_freq = 80.0
        
        phase_signal = np.sin(2 * np.pi * pha_freq * t)
        amplitude_mod = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
        carrier = np.sin(2 * np.pi * amp_freq * t)
        
        signal = phase_signal + amplitude_mod * carrier * 0.5
        signal = signal.reshape(1, 1, 1, -1)  # (batch, channels, segments, time)
        
        return signal, fs
    
    @pytest.fixture
    def multichannel_signal(self) -> Tuple[np.ndarray, float]:
        """Create a multichannel test signal."""
        fs = 512  # Hz
        duration = 1.0  # seconds
        n_channels = 4
        t = np.linspace(0, duration, int(fs * duration))
        
        signals = []
        for ch in range(n_channels):
            # Different PAC coupling for each channel
            pha_freq = 4.0 + ch * 2.0  # 4, 6, 8, 10 Hz
            amp_freq = 60.0 + ch * 20.0  # 60, 80, 100, 120 Hz
            
            phase_signal = np.sin(2 * np.pi * pha_freq * t)
            amplitude_mod = (1 + 0.7 * np.cos(2 * np.pi * pha_freq * t)) / 2
            carrier = np.sin(2 * np.pi * amp_freq * t)
            
            signal = phase_signal + amplitude_mod * carrier * 0.5
            signals.append(signal)
        
        signal = np.stack(signals, axis=0)
        signal = signal.reshape(1, n_channels, 1, -1)
        
        return signal, fs
    
    def test_basic_calculation(self, basic_signal):
        """Test basic PAC calculation."""
        signal, fs = basic_signal
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=10,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=10
        )
        
        # Check output shapes
        assert pac_values.shape == (1, 1, 10, 10)
        assert pha_freqs.shape == (10,)
        assert amp_freqs.shape == (10,)
        
        # Check frequency ranges
        assert pha_freqs[0] >= 2.0
        assert pha_freqs[-1] <= 20.0
        assert amp_freqs[0] >= 60.0
        assert amp_freqs[-1] <= 120.0
        
        # Check PAC peak is near expected frequencies (6 Hz, 80 Hz)
        pac_np = pac_values.squeeze().cpu().numpy()
        peak_idx = np.unravel_index(pac_np.argmax(), pac_np.shape)
        peak_pha = pha_freqs[peak_idx[0]]
        peak_amp = amp_freqs[peak_idx[1]]
        
        assert abs(peak_pha - 6.0) < 5.0  # Within 5 Hz
        assert abs(peak_amp - 80.0) < 20.0  # Within 20 Hz
    
    def test_tensor_input(self, basic_signal):
        """Test with PyTorch tensor input."""
        signal_np, fs = basic_signal
        signal_tensor = torch.from_numpy(signal_np)
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal_tensor,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5
        )
        
        assert isinstance(pac_values, torch.Tensor)
        assert isinstance(pha_freqs, np.ndarray)
        assert isinstance(amp_freqs, np.ndarray)
        assert pac_values.shape == (1, 1, 5, 5)
    
    def test_permutation_testing(self, basic_signal):
        """Test PAC with permutation testing."""
        signal, fs = basic_signal
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            n_perm=50  # Small number for testing
        )
        
        # With permutation testing, values should be z-scored
        assert pac_values.shape == (1, 1, 5, 5)
        
        # With permutation testing, values are z-scored
        # They can be positive or negative, but should have reasonable range
        assert abs(pac_values.max().item()) < 10.0  # Reasonable z-score range
    
    def test_return_distribution(self, basic_signal):
        """Test returning surrogate distribution."""
        signal, fs = basic_signal
        
        result = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            n_perm=20,
            return_dist=True
        )
        
        # Should return 4 values with distribution
        assert len(result) == 4
        pac_values, surrogate_dist, pha_freqs, amp_freqs = result
        
        # Check surrogate distribution shape
        assert surrogate_dist.shape == (20, 1, 1, 5, 5)
        
        # Observed PAC should generally be higher than surrogates
        observed_max = pac_values.max().item()
        surrogate_mean = surrogate_dist.mean().item()
        assert observed_max > surrogate_mean
    
    def test_multichannel_processing(self, multichannel_signal):
        """Test multichannel signal processing."""
        signal, fs = multichannel_signal
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=8,
            amp_n_bands=8
        )
        
        # Should preserve channel dimension
        assert pac_values.shape == (1, 4, 8, 8)
        
        # Each channel should have different peak
        for ch in range(4):
            ch_pac = pac_values[0, ch].cpu().numpy()
            peak_idx = np.unravel_index(ch_pac.argmax(), ch_pac.shape)
            peak_pha = pha_freqs[peak_idx[0]]
            
            # Expected phase frequencies: 4, 6, 8, 10 Hz
            expected_pha = 4.0 + ch * 2.0
            assert abs(peak_pha - expected_pha) < 3.0
    
    def test_channel_averaging(self, multichannel_signal):
        """Test channel averaging option."""
        signal, fs = multichannel_signal
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            average_channels=True
        )
        
        # Channel dimension should be averaged out
        assert pac_values.shape == (1, 5, 5)
    
    def test_chunk_processing(self, basic_signal):
        """Test chunked processing for memory efficiency."""
        signal, fs = basic_signal
        # Create larger batch for chunking
        signal_large = np.tile(signal, (10, 1, 1, 1))
        
        pac_values, pha_freqs, amp_freqs = calculate_pac(
            signal_large,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            chunk_size=3  # Process 3 at a time
        )
        
        assert pac_values.shape == (10, 1, 5, 5)
        
        # All batches should have similar PAC patterns
        pac_std = pac_values.std(dim=0).max().item()
        assert pac_std < 0.5  # Low variance across batches
    
    def test_frequency_validation(self, basic_signal):
        """Test frequency range validation and adjustment."""
        signal, fs = basic_signal
        
        # Test warning for frequencies near Nyquist
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            pac_values, pha_freqs, amp_freqs = calculate_pac(
                signal,
                fs=fs,
                amp_end_hz=480.0,  # Close to Nyquist (500 Hz)
                pha_n_bands=5,
                amp_n_bands=5
            )
            
            # Should have warned about frequency adjustment
            assert len(w) > 0
            assert "exceeds safe frequency limit" in str(w[0].message)
    
    def test_invalid_frequency_ranges(self, basic_signal):
        """Test error handling for invalid frequency ranges."""
        signal, fs = basic_signal
        
        # pha_start >= pha_end
        with pytest.raises(ValueError, match="pha_start_hz.*must be < pha_end_hz"):
            calculate_pac(signal, fs=fs, pha_start_hz=20.0, pha_end_hz=10.0)
        
        # amp_start >= amp_end
        with pytest.raises(ValueError, match="amp_start_hz.*must be < amp_end_hz"):
            calculate_pac(signal, fs=fs, amp_start_hz=120.0, amp_end_hz=60.0)
        
        # Negative frequency
        with pytest.raises(ValueError, match="pha_start_hz.*must be > 0"):
            calculate_pac(signal, fs=fs, pha_start_hz=-1.0)
    
    def test_device_handling(self, basic_signal):
        """Test different device specifications."""
        signal, fs = basic_signal
        
        # Test CPU device
        pac_cpu = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=3,
            amp_n_bands=3,
            device="cpu"
        )[0]
        
        assert pac_cpu.device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            pac_cuda = calculate_pac(
                signal,
                fs=fs,
                pha_n_bands=3,
                amp_n_bands=3,
                device="cuda"
            )[0]
            
            assert pac_cuda.device.type == "cuda"
            
            # Results should be similar
            pac_diff = (pac_cpu - pac_cuda.cpu()).abs().max().item()
            assert pac_diff < 0.01
    
    def test_fp16_mode(self, basic_signal):
        """Test half-precision computation."""
        signal, fs = basic_signal
        
        if torch.cuda.is_available():
            pac_fp32 = calculate_pac(
                signal,
                fs=fs,
                pha_n_bands=5,
                amp_n_bands=5,
                fp16=False,
                device="cuda"
            )[0]
            
            pac_fp16 = calculate_pac(
                signal,
                fs=fs,
                pha_n_bands=5,
                amp_n_bands=5,
                fp16=True,
                device="cuda"
            )[0]
            
            # FP16 should give similar results
            diff = (pac_fp32 - pac_fp16).abs().max().item()
            assert diff < 0.1  # Some tolerance for precision difference
    
    def test_trainable_mode(self, basic_signal):
        """Test trainable filter mode."""
        signal_np, fs = basic_signal
        signal = torch.from_numpy(signal_np).requires_grad_(True)
        
        pac_values, _, _ = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=3,
            amp_n_bands=3,
            trainable=True
        )
        
        # Should be able to compute gradients
        loss = pac_values.sum()
        loss.backward()
        
        assert signal.grad is not None
        assert signal.grad.shape == signal.shape
    
    def test_filtfilt_mode(self, basic_signal):
        """Test zero-phase filtering mode."""
        signal, fs = basic_signal
        
        # Standard filtering
        pac_standard = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filtfilt_mode=False
        )[0]
        
        # Zero-phase filtering
        pac_filtfilt = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filtfilt_mode=True
        )[0]
        
        # Both should detect PAC but values might differ
        assert pac_standard.max().item() > 0.1
        assert pac_filtfilt.max().item() > 0.1
    
    def test_amplitude_probability_mode(self, basic_signal):
        """Test amplitude probability mode."""
        signal, fs = basic_signal
        
        pac_values, _, _ = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            amp_prob=True
        )
        
        # In probability mode, values should be between 0 and 1
        assert pac_values.min().item() >= 0.0
        assert pac_values.max().item() <= 1.0
    
    def test_custom_filter_cycles(self, basic_signal):
        """Test custom filter cycle parameters."""
        signal, fs = basic_signal
        
        # Narrow filters
        pac_narrow = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filter_cycle_pha=1,
            filter_cycle_amp=2
        )[0]
        
        # Wide filters
        pac_wide = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5,
            filter_cycle_pha=6,
            filter_cycle_amp=12
        )[0]
        
        # Both should detect PAC
        assert pac_narrow.max().item() > 0.1
        assert pac_wide.max().item() > 0.1
    
    def test_edge_cases(self):
        """Test edge cases and special inputs."""
        fs = 512
        
        # Very short signal
        short_signal = np.random.randn(1, 1, 1, 128)
        pac_values, _, _ = calculate_pac(
            short_signal,
            fs=fs,
            pha_n_bands=2,
            amp_n_bands=2
        )
        assert pac_values.shape == (1, 1, 2, 2)
        
        # Single sample (should handle gracefully)
        single_sample = np.random.randn(1, 1, 1, 1)
        with pytest.raises(Exception):  # Should fail appropriately
            calculate_pac(single_sample, fs=fs)
    
    def test_return_types(self, basic_signal):
        """Test return value types and structure."""
        signal, fs = basic_signal
        
        # Without distribution
        result = calculate_pac(signal, fs=fs, pha_n_bands=3, amp_n_bands=3)
        assert len(result) == 3
        pac_values, pha_freqs, amp_freqs = result
        
        assert isinstance(pac_values, torch.Tensor)
        assert isinstance(pha_freqs, np.ndarray)
        assert isinstance(amp_freqs, np.ndarray)
        
        # With distribution
        result_dist = calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=3,
            amp_n_bands=3,
            n_perm=10,
            return_dist=True
        )
        assert len(result_dist) == 4
        
    def test_warnings_and_validations(self, basic_signal):
        """Test warning messages and input validations."""
        signal, fs = basic_signal
        
        # Test return_dist without n_perm warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            calculate_pac(
                signal,
                fs=fs,
                pha_n_bands=3,
                amp_n_bands=3,
                return_dist=True  # No n_perm
            )
            
            assert any("return_dist=True has no effect" in str(warning.message) for warning in w)
        
        # Test low n_perm warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            calculate_pac(
                signal,
                fs=fs,
                pha_n_bands=3,
                amp_n_bands=3,
                n_perm=5
            )
            
            assert any("very low for permutation testing" in str(warning.message) for warning in w)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_calculate_gpac.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-25 09:44:58 (ywatanabe)"
# # File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/_pac.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/gpac/_pac.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# import warnings
# from typing import Optional, Tuple, Union
# 
# import numpy as np
# import torch
# import torch.nn as nn
# 
# from ._PAC import PAC
# from ._utils import ensure_4d_input
# 
# 
# # --- User-facing Function ---
# def calculate_pac(
#     signal: torch.Tensor | np.ndarray,
#     fs: float,
#     pha_start_hz: float = 2.0,
#     pha_end_hz: float = 20.0,
#     pha_n_bands: int = 50,
#     amp_start_hz: float = 60.0,
#     amp_end_hz: float = 160.0,
#     amp_n_bands: int = 30,
#     n_perm: Optional[int] = None,
#     trainable: bool = False,
#     fp16: bool = False,
#     amp_prob: bool = False,
#     mi_n_bins: int = 18,
#     filter_cycle_pha: int = 3,
#     filter_cycle_amp: int = 6,
#     device: Optional[str | torch.device] = None,
#     chunk_size: Optional[int] = None,
#     average_channels: bool = False,
#     return_dist: bool = False,
#     filtfilt_mode: bool = False,
#     edge_mode: Optional[str] = None,
# ) -> Union[
#     # Standard return (no distribution): pac_values, pha_freqs, amp_freqs
#     Tuple[torch.Tensor, np.ndarray, np.ndarray],
#     # With distribution: pac_values, surrogate_dist, pha_freqs, amp_freqs
#     Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
# ]:
#     """
#     Calculate Phase-Amplitude Coupling (PAC) between neural oscillations.
#     
#     PAC measures how the amplitude of fast brain waves (e.g., 60-160 Hz gamma) is 
#     modulated by the phase of slower waves (e.g., 2-20 Hz theta/alpha). This is 
#     important for understanding cross-frequency interactions in neural networks.
# 
#     Input Signal Format:
#         signal: 4D tensor with shape (batch_size, channels, segments, time_points)
#         - batch_size: Number of different recordings/subjects
#         - channels: EEG recording sites (different electrodes)  
#         - segments: Time windows from same session (e.g., consecutive 2-sec epochs)
#         - time_points: Individual samples (duration × sampling_rate)
# 
#     Args:
#         signal: Input neural signals as PyTorch tensor or numpy array
#         fs: Sampling frequency in Hz (e.g., 256, 512, 1000)
#         pha_start_hz: Start of phase frequency range (default: 2 Hz)
#         pha_end_hz: End of phase frequency range (default: 20 Hz)  
#         pha_n_bands: Number of phase frequency bands to analyze (default: 50)
#         amp_start_hz: Start of amplitude frequency range (default: 60 Hz)
#         amp_end_hz: End of amplitude frequency range (default: 160 Hz)
#         amp_n_bands: Number of amplitude frequency bands to analyze (default: 30)
#         n_perm: Number of permutations for statistical testing (None=no testing)
#         trainable: Use learnable frequency filters (for deep learning integration)
#         fp16: Use half-precision (faster but less precise)
#         amp_prob: Return amplitude probabilities instead of modulation index
#         mi_n_bins: Number of phase bins for modulation index (default: 18)
#         filter_cycle_pha: Filter bandwidth for phase in cycles (default: 3)
#         filter_cycle_amp: Filter bandwidth for amplitude in cycles (default: 6)
#         device: Compute device ("cuda", "cpu", or torch.device)
#         chunk_size: Process in chunks to save memory (None=no chunking)
#         average_channels: Average results across channels (default: False)
#         return_dist: Return full surrogate distribution for custom analysis
#         filtfilt_mode: Use zero-phase filtering for TensorPAC compatibility (default: False)
# 
#     Returns:
#         Standard return (return_dist=False):
#             - pac_values: Shape (batch_size, channels, pha_n_bands, amp_n_bands)
#             - pha_freqs: Phase frequency centers (pha_n_bands,)
#             - amp_freqs: Amplitude frequency centers (amp_n_bands,)
#             
#         With surrogate distribution (return_dist=True and n_perm > 0):
#             - pac_values: Shape (batch_size, channels, pha_n_bands, amp_n_bands)
#             - surrogate_dist: Shape (n_perm, batch_size, channels, pha_n_bands, amp_n_bands)
#             - pha_freqs: Phase frequency centers (pha_n_bands,)
#             - amp_freqs: Amplitude frequency centers (amp_n_bands,)
# 
#     Note:
#         Frequencies exceeding 90% of Nyquist limit (fs/2) are automatically adjusted
#         with warnings to prevent filter errors.
# 
#     Examples:
#         Basic PAC calculation:
# 
#         >>> import torch
#         >>> import numpy as np
#         >>> from gpac import calculate_pac
#         >>>
#         >>> # Generate sample data: 1 channel, 10 second signal at 1000 Hz
#         >>> fs = 1000
#         >>> t = np.arange(0, 10, 1/fs)
#         >>> pha_freq = 5  # Hz
#         >>> amp_freq = 80  # Hz
#         >>>
#         >>> # Create signal with PAC: phase of 5 Hz modulates amplitude of 80 Hz
#         >>> pha = np.sin(2 * np.pi * pha_freq * t)
#         >>> amp = np.sin(2 * np.pi * amp_freq * t)
#         >>> signal = np.sin(2 * np.pi * pha_freq * t) + (1 + 0.8 * pha) * amp * 0.2
#         >>> signal = signal.reshape(1, 1, 1, -1)  # [batch, channel, segment, time]
#         >>>
#         >>> # Standard PAC calculation
#         >>> pac_values, pha_freqs, amp_freqs = calculate_pac(
#         ...     signal,
#         ...     fs=fs,
#         ...     pha_start_hz=2,
#         ...     pha_end_hz=20,
#         ...     pha_n_bands=10,
#         ...     amp_start_hz=60,
#         ...     amp_end_hz=120,
#         ...     amp_n_bands=10,
#         ...     n_perm=200  # Use permutation testing
#         ... )
# 
#         Get the full permutation test distribution:
# 
#         >>> # Get the distribution of surrogate values
#         >>> pac_values, surrogate_dist, pha_freqs, amp_freqs = calculate_pac(
#         ...     signal,
#         ...     fs=fs,
#         ...     pha_start_hz=2,
#         ...     pha_end_hz=20,
#         ...     pha_n_bands=10,
#         ...     amp_start_hz=60,
#         ...     amp_end_hz=120,
#         ...     amp_n_bands=10,
#         ...     n_perm=200,
#         ...     return_dist=True  # Return the full distribution
#         ... )
#         >>>
#         >>> # Visualize the PAC values
#         >>> import matplotlib.pyplot as plt
#         >>>
#         >>> # Plot the PAC values
#         >>> plt.figure(figsize=(10, 4))
#         >>> plt.subplot(121)
#         >>> plt.imshow(pac_values[0, 0], aspect='auto', origin='lower')
#         >>> plt.xlabel('Amplitude Frequency (Hz)')
#         >>> plt.ylabel('Phase Frequency (Hz)')
#         >>> plt.title('PAC Z-Scores')
#         >>> plt.colorbar(label='Z-score')
#         >>>
#         >>> # Get indices of max coupling
#         >>> max_idx = pac_values[0, 0].argmax()
#         >>> max_pha_idx, max_amp_idx = np.unravel_index(max_idx, pac_values[0, 0].shape)
#         >>>
#         >>> # Plot the surrogate distribution for the max coupling
#         >>> plt.subplot(122)
#         >>> surrogate_values = surrogate_dist[:, 0, 0, max_pha_idx, max_amp_idx].numpy()
#         >>> observed_value = pac_values[0, 0, max_pha_idx, max_amp_idx].item()
#         >>> plt.hist(surrogate_values, bins=20, alpha=0.8)
#         >>> plt.axvline(observed_value, color='r', linestyle='--',
#         ...             label=f'Observed: {observed_value:.2f}')
#         >>> plt.xlabel('PAC Value')
#         >>> plt.ylabel('Count')
#         >>> plt.title('Surrogate Distribution')
#         >>> plt.legend()
#         >>> plt.tight_layout()
#         >>> plt.show()
# 
#         Custom statistical analysis using the surrogate distribution:
# 
#         >>> # Calculate p-values manually
#         >>> def calculate_pvalues(observed, surrogates):
#         ...     # One-sided p-value: proportion of surrogates >= observed
#         ...     return ((surrogates >= observed).sum(axis=0) / len(surrogates))
#         >>>
#         >>> # Convert tensors to numpy arrays
#         >>> pac_array = pac_values[0, 0].numpy()
#         >>> surr_array = surrogate_dist[:, 0, 0].numpy()
#         >>>
#         >>> # Calculate p-values
#         >>> pvalues = calculate_pvalues(pac_array, surr_array)
#         >>>
#         >>> # Apply multiple comparison correction (FDR)
#         >>> from statsmodels.stats.multitest import multipletests
#         >>> pvals_flat = pvalues.flatten()
#         >>> significant, pvals_corr, _, _ = multipletests(pvals_flat, method='fdr_bh')
#         >>> pvals_corr = pvals_corr.reshape(pvalues.shape)
#         >>>
#         >>> # Find significant couplings after correction
#         >>> significant_mask = pvals_corr < 0.05
#     """
#     # 1. Frequency Validation and Adjustment
#     nyquist_freq = fs / 2
#     safety_factor = 0.9  # Leave some margin below Nyquist
#     max_safe_freq = nyquist_freq * safety_factor
#     
#     # Adjust amplitude end frequency if it exceeds safe limits
#     if amp_end_hz >= max_safe_freq:
#         amp_end_hz_adjusted = max_safe_freq
#         warnings.warn(
#             f"amp_end_hz ({amp_end_hz} Hz) exceeds safe frequency limit "
#             f"({max_safe_freq:.1f} Hz) for fs={fs} Hz. "
#             f"Adjusting to {amp_end_hz_adjusted:.1f} Hz."
#         )
#         amp_end_hz = amp_end_hz_adjusted
#     
#     # Adjust phase end frequency if it exceeds safe limits
#     if pha_end_hz >= max_safe_freq:
#         pha_end_hz_adjusted = min(max_safe_freq, amp_start_hz * 0.5)  # Keep phase below amplitude
#         warnings.warn(
#             f"pha_end_hz ({pha_end_hz} Hz) exceeds safe frequency limit "
#             f"({max_safe_freq:.1f} Hz) for fs={fs} Hz. "
#             f"Adjusting to {pha_end_hz_adjusted:.1f} Hz."
#         )
#         pha_end_hz = pha_end_hz_adjusted
#     
#     # Ensure phase frequencies are below amplitude frequencies
#     if pha_end_hz >= amp_start_hz:
#         pha_end_hz_adjusted = amp_start_hz * 0.8
#         warnings.warn(
#             f"pha_end_hz ({pha_end_hz} Hz) should be below amp_start_hz ({amp_start_hz} Hz). "
#             f"Adjusting to {pha_end_hz_adjusted:.1f} Hz."
#         )
#         pha_end_hz = pha_end_hz_adjusted
#         
#     # Final validation
#     if pha_start_hz >= pha_end_hz:
#         raise ValueError(f"pha_start_hz ({pha_start_hz}) must be < pha_end_hz ({pha_end_hz})")
#     if amp_start_hz >= amp_end_hz:
#         raise ValueError(f"amp_start_hz ({amp_start_hz}) must be < amp_end_hz ({amp_end_hz})")
#     if pha_start_hz <= 0:
#         raise ValueError(f"pha_start_hz ({pha_start_hz}) must be > 0")
#     
#     # 2. Device Handling
#     if device is None:
#         resolved_device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )
#     elif isinstance(device, str):
#         resolved_device = torch.device(device)
#     else:
#         resolved_device = device
# 
#     # 3. Input Preparation
#     if isinstance(signal, np.ndarray):
#         try:
#             dtype = np.float32 if signal.dtype == np.float64 else signal.dtype
#             signal_tensor = torch.from_numpy(signal.astype(dtype))
#         except TypeError as e:
#             raise TypeError(
#                 f"Could not convert numpy array to tensor: {e}"
#             ) from e
#     elif isinstance(signal, torch.Tensor):
#         signal_tensor = signal
#     else:
#         raise TypeError("Input signal must be a torch.Tensor or numpy.ndarray")
# 
#     signal_tensor = signal_tensor.to(resolved_device)
#     signal_4d = ensure_4d_input(signal_tensor)
#     batch_size, n_chs, n_segments, seq_len = signal_4d.shape
#     target_dtype = torch.float16 if fp16 else torch.float32
#     signal_4d = signal_4d.to(target_dtype)
# 
#     # 4. Model Instantiation
#     model = PAC(
#         seq_len=seq_len,
#         fs=fs,
#         pha_start_hz=pha_start_hz,
#         pha_end_hz=pha_end_hz,
#         pha_n_bands=pha_n_bands,
#         amp_start_hz=amp_start_hz,
#         amp_end_hz=amp_end_hz,
#         amp_n_bands=amp_n_bands,
#         n_perm=n_perm,
#         trainable=trainable,
#         fp16=fp16,
#         amp_prob=amp_prob,
#         mi_n_bins=mi_n_bins,
#         filter_cycle_pha=filter_cycle_pha,
#         filter_cycle_amp=filter_cycle_amp,
#         return_dist=return_dist,
#         filtfilt_mode=filtfilt_mode,
#         edge_mode=edge_mode,
#     ).to(resolved_device)
# 
#     if not trainable:
#         model.eval()
# 
#     # 5. Calculation (Chunked or Full)
#     num_traces = batch_size * n_chs * n_segments
#     process_in_chunks = (
#         chunk_size is not None and chunk_size > 0 and num_traces > chunk_size
#     )
# 
#     # Validate return_dist and n_perm combination
#     if return_dist and n_perm is None:
#         warnings.warn(
#             "return_dist=True has no effect when n_perm is None. "
#             "No distribution will be returned."
#         )
#         return_dist = False
# 
#     # Validate n_perm when it's provided
#     if n_perm is not None and n_perm < 10:
#         warnings.warn(
#             f"Using n_perm={n_perm} which is very low for permutation testing. "
#             "Consider using at least 200 for stable statistical results."
#         )
# 
#     # Additional warning for return_dist with low permutation count
#     if return_dist and n_perm is not None and n_perm < 50:
#         warnings.warn(
#             f"Using n_perm={n_perm} with return_dist=True may not provide "
#             "a reliable distribution for statistical analysis. "
#             "Consider using at least 100-200 permutations."
#         )
# 
#     if not process_in_chunks:
#         grad_context = torch.enable_grad() if trainable else torch.no_grad()
#         with grad_context:
#             result = model(signal_4d)
#             # Extract surrogate distribution if it was returned
#             if isinstance(result, tuple) and len(result) == 2:
#                 pac_results, surrogate_dist = result
#             else:
#                 pac_results = result
#     else:
#         print(
#             f"Processing {num_traces} traces in chunks of size {chunk_size}..."
#         )
#         # Process in chunks while preserving original structure
#         all_pac_results_chunks = []
#         all_surrogate_dist_chunks = (
#             [] if return_dist and n_perm is not None else None
#         )
# 
#         grad_context = torch.enable_grad() if trainable else torch.no_grad()
#         with grad_context:
#             # Process batch chunks to avoid memory issues
#             for batch_start in range(0, batch_size, chunk_size):
#                 batch_end = min(batch_start + chunk_size, batch_size)
#                 current_chunk_batch_size = batch_end - batch_start
# 
#                 # Extract chunk preserving original structure: (ChunkBatch, C, Seg, Time)
#                 signal_chunk = signal_4d[batch_start:batch_end]
#                 chunk_result = model(signal_chunk)
# 
#                 # Handle distribution if returned
#                 if isinstance(chunk_result, tuple) and len(chunk_result) == 2:
#                     chunk_pac, chunk_surrogate = chunk_result
#                     all_pac_results_chunks.append(chunk_pac)
#                     if all_surrogate_dist_chunks is not None:
#                         all_surrogate_dist_chunks.append(chunk_surrogate)
#                 else:
#                     all_pac_results_chunks.append(chunk_result)
# 
#                 if resolved_device.type == "cuda":
#                     torch.cuda.empty_cache()
# 
#         # Concatenate results from chunks
#         pac_results = torch.cat(all_pac_results_chunks, dim=0)
# 
#         # Surrogate distributions handling for chunked processing
#         surrogate_dist = None
#         if (
#             all_surrogate_dist_chunks is not None
#             and len(all_surrogate_dist_chunks) > 0
#         ):
#             try:
#                 # Concatenate surrogate distributions along the batch dimension
#                 surrogate_dist = torch.cat(all_surrogate_dist_chunks, dim=1)
#             except Exception as e:
#                 warnings.warn(
#                     f"Error concatenating surrogate distributions in chunked mode: {e}. "
#                     "Returning PAC values without distribution."
#                 )
#                 surrogate_dist = None
#                 return_dist = False
# 
#         print("Chunk processing complete.")
# 
#     # Average across channels if requested
#     if average_channels and pac_results.shape[1] > 1:
#         pac_results = pac_results.mean(dim=1)
#         if surrogate_dist is not None and surrogate_dist.shape[2] > 1:
#             surrogate_dist = surrogate_dist.mean(dim=2)
# 
#     # 6. Prepare Outputs
#     # Ensure frequencies are numpy arrays on CPU
#     if isinstance(model.PHA_MIDS_HZ, nn.Parameter):
#         freqs_pha_np = model.PHA_MIDS_HZ.detach().cpu().numpy()
#     else:
#         freqs_pha_np = model.PHA_MIDS_HZ.cpu().numpy()
# 
#     if isinstance(model.AMP_MIDS_HZ, nn.Parameter):
#         freqs_amp_np = model.AMP_MIDS_HZ.detach().cpu().numpy()
#     else:
#         freqs_amp_np = model.AMP_MIDS_HZ.cpu().numpy()
# 
#     # Return appropriate output based on whether distribution was requested and available
#     if return_dist and surrogate_dist is not None:
#         return pac_results, surrogate_dist, freqs_pha_np, freqs_amp_np
#     else:
#         return pac_results, freqs_pha_np, freqs_amp_np
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/gPAC/src/gpac/_calculate_gpac.py
# --------------------------------------------------------------------------------
