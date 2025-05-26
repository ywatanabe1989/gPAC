#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

try:
    from torchaudio.prototype.functional import sinc_impulse_response
    TORCHAUDIO_SINC_AVAILABLE = True
except ImportError:
    TORCHAUDIO_SINC_AVAILABLE = False
    sinc_impulse_response = None


def to_even(n):
    """Convert to even number."""
    return int(n + n % 2)


def to_odd(n):
    """Convert to odd number.""" 
    return int(n + (n + 1) % 2)


def build_differentiable_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle=3):
    """
    Build differentiable bandpass filters using sinc impulse response.
    Based on mngs implementation but adapted for gPAC.
    
    Args:
        sig_len: Signal length
        fs: Sampling frequency
        pha_mids: Phase band center frequencies (learnable)
        amp_mids: Amplitude band center frequencies (learnable)
        cycle: Number of cycles for filter design
        
    Returns:
        torch.Tensor: Combined filter kernels
    """
    if not TORCHAUDIO_SINC_AVAILABLE:
        raise ImportError(
            "torchaudio.prototype.functional.sinc_impulse_response is required "
            "for differentiable filter design. Please install torchaudio>=0.12"
        )
    
    def _define_freqs(mids, factor):
        """Define low and high frequencies from center frequencies."""
        lows = mids - mids / factor
        highs = mids + mids / factor
        return lows, highs

    def define_order(low_hz, fs, sig_len, cycle):
        """Define filter order."""
        order = cycle * int((fs // low_hz))
        order = order if 3 * order <= sig_len else (sig_len - 1) // 3
        order = to_even(order)
        return order

    def _calc_filters(lows_hz, highs_hz, fs, order):
        """Calculate bandpass filters using differentiable sinc functions."""
        nyq = fs / 2.0
        order = to_odd(order)
        
        # Normalize frequencies to [0, 1] range
        lows_norm = lows_hz / nyq
        highs_norm = highs_hz / nyq
        
        # Ensure frequencies are in valid range [0, 1]
        lows_norm = torch.clamp(lows_norm, 0.001, 0.999)
        highs_norm = torch.clamp(highs_norm, 0.001, 0.999)
        
        # Create lowpass filters and subtract to get bandpass
        irs_low = sinc_impulse_response(highs_norm, window_size=order)
        irs_high = sinc_impulse_response(lows_norm, window_size=order)
        
        # Bandpass = lowpass(high_cutoff) - lowpass(low_cutoff)
        irs = irs_low - irs_high
        return irs

    # Process phase bands
    pha_lows, pha_highs = _define_freqs(pha_mids, factor=4.0)
    
    # Process amplitude bands  
    amp_lows, amp_highs = _define_freqs(amp_mids, factor=8.0)
    
    # Determine filter order based on lowest frequency
    all_lows = torch.cat([pha_lows, amp_lows])
    lowest = all_lows.min()
    order = define_order(lowest.item(), fs, sig_len, cycle)
    
    # Build filters
    pha_bp_filters = _calc_filters(pha_lows, pha_highs, fs, order)
    amp_bp_filters = _calc_filters(amp_lows, amp_highs, fs, order)
    
    # Combine all filters
    return torch.vstack([pha_bp_filters, amp_bp_filters])


def init_differentiable_bandpass_filters(
    sig_len,
    fs,
    pha_low_hz=2,
    pha_high_hz=20,
    pha_n_bands=30,
    amp_low_hz=60,
    amp_high_hz=160,
    amp_n_bands=50,
    cycle=3,
):
    """
    Initialize differentiable bandpass filters with learnable parameters.
    
    Returns:
        Tuple[torch.Tensor, nn.Parameter, nn.Parameter]: 
            (filters, pha_mids, amp_mids)
    """
    # Create learnable center frequencies
    pha_mids = nn.Parameter(
        torch.linspace(pha_low_hz, pha_high_hz, pha_n_bands)
    )
    amp_mids = nn.Parameter(
        torch.linspace(amp_low_hz, amp_high_hz, amp_n_bands)
    )
    
    # Build initial filters
    filters = build_differentiable_bandpass_filters(
        sig_len, fs, pha_mids, amp_mids, cycle
    )
    
    return filters, pha_mids, amp_mids