#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorPAC Compatibility Module for gPAC

This module provides functions and configurations to achieve better compatibility
with TensorPAC, including appropriate frequency band setup and value scaling.

Based on empirical testing and documentation, gPAC values are typically 4-12x 
smaller than TensorPAC values depending on the configuration.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional
from ._PAC import PAC


# Standard TensorPAC-compatible configurations
TENSORPAC_CONFIGS = {
    'standard': {
        'n_pha': 10,
        'n_amp': 10,
        'pha_range': (2, 20),
        'amp_range': (60, 160),
        'scale_factor': 5.0
    },
    'medium': {
        'n_pha': 30,
        'n_amp': 30,
        'pha_range': (1.5, 25),
        'amp_range': (52.5, 180),
        'scale_factor': 6.5
    },
    'hres': {
        'n_pha': 50,
        'n_amp': 50,
        'pha_range': (1.5, 25),
        'amp_range': (52.5, 180),
        'scale_factor': 8.0
    },
    'compatible': {
        'n_pha': 50,
        'n_amp': 30,
        'pha_range': (1.5, 25),
        'amp_range': (52.5, 180),
        'scale_factor': 12.0
    }
}


def calculate_pac_tensorpac_compat(
    signal: Union[np.ndarray, torch.Tensor],
    fs: float,
    config: str = 'compatible',
    custom_scale: Optional[float] = None,
    return_unscaled: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate PAC with TensorPAC-compatible configuration and scaling.
    
    Parameters
    ----------
    signal : array-like
        Input signal, shape (n_samples,) or (n_epochs, n_samples)
    fs : float
        Sampling frequency in Hz
    config : str
        Configuration preset: 'standard', 'medium', 'hres', or 'compatible'
        Default is 'compatible' which uses 50 phase x 30 amplitude bands
    custom_scale : float, optional
        Custom scaling factor to override the preset value
    return_unscaled : bool
        If True, also return the unscaled gPAC values
    **kwargs : dict
        Additional parameters to pass to PAC module
        
    Returns
    -------
    pac_values : np.ndarray
        PAC values scaled to match TensorPAC, shape (n_pha, n_amp)
    pha_freqs : np.ndarray
        Center frequencies for phase bands
    amp_freqs : np.ndarray
        Center frequencies for amplitude bands
    pac_unscaled : np.ndarray (optional)
        Unscaled gPAC values if return_unscaled=True
        
    Examples
    --------
    >>> # Basic usage with default 50x30 configuration
    >>> pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000)
    
    >>> # Use high-resolution 50x50 configuration
    >>> pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000, config='hres')
    
    >>> # Get both scaled and unscaled values
    >>> pac_scaled, f_pha, f_amp, pac_raw = calculate_pac_tensorpac_compat(
    ...     signal, fs=1000, return_unscaled=True
    ... )
    """
    # Get configuration
    if config not in TENSORPAC_CONFIGS:
        raise ValueError(f"config must be one of {list(TENSORPAC_CONFIGS.keys())}")
    
    cfg = TENSORPAC_CONFIGS[config]
    
    # Extract parameters
    n_pha = cfg['n_pha']
    n_amp = cfg['n_amp']
    pha_range = cfg['pha_range']
    amp_range = cfg['amp_range']
    scale_factor = custom_scale if custom_scale is not None else cfg['scale_factor']
    
    # Handle signal shape
    if isinstance(signal, np.ndarray):
        signal = torch.tensor(signal, dtype=torch.float32)
    
    if signal.ndim == 1:
        signal = signal.reshape(1, 1, 1, -1)
    elif signal.ndim == 2:
        # (n_epochs, n_samples) -> (n_epochs, 1, 1, n_samples)
        signal = signal.reshape(signal.shape[0], 1, 1, signal.shape[1])
    
    # Create PAC module with TensorPAC-compatible settings
    pac_module = PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_start_hz=pha_range[0],
        pha_end_hz=pha_range[1],
        pha_n_bands=n_pha,
        amp_start_hz=amp_range[0],
        amp_end_hz=amp_range[1],
        amp_n_bands=n_amp,
        mi_n_bins=18,  # TensorPAC default
        filtfilt_mode=True,  # For better filter match
        **kwargs
    )
    
    # Calculate PAC
    result = pac_module(signal)
    
    # Extract values
    pac_values = result['mi'].squeeze().numpy()
    pha_freqs = result['pha_freqs_hz'].squeeze().numpy()
    amp_freqs = result['amp_freqs_hz'].squeeze().numpy()
    
    # Apply scaling
    pac_scaled = pac_values * scale_factor
    
    if return_unscaled:
        return pac_scaled, pha_freqs, amp_freqs, pac_values
    else:
        return pac_scaled, pha_freqs, amp_freqs


def get_tensorpac_bands(n_pha: int, n_amp: int,
                       pha_range: Tuple[float, float] = (1.5, 25),
                       amp_range: Tuple[float, float] = (52.5, 180)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate frequency bands matching TensorPAC's approach.
    
    Parameters
    ----------
    n_pha : int
        Number of phase bands
    n_amp : int
        Number of amplitude bands
    pha_range : tuple
        (min, max) frequencies for phase bands
    amp_range : tuple
        (min, max) frequencies for amplitude bands
        
    Returns
    -------
    pha_bands : np.ndarray
        Phase band edges, shape (n_pha, 2)
    amp_bands : np.ndarray
        Amplitude band edges, shape (n_amp, 2)
    """
    # Create linear spacing
    pha_edges = np.linspace(pha_range[0], pha_range[1], n_pha + 1)
    amp_edges = np.linspace(amp_range[0], amp_range[1], n_amp + 1)
    
    # Convert to band pairs
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    return pha_bands, amp_bands


def compare_with_tensorpac(signal: Union[np.ndarray, torch.Tensor],
                          fs: float,
                          config: str = 'compatible') -> dict:
    """
    Helper function to compare gPAC and TensorPAC results.
    
    Parameters
    ----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency
    config : str
        Configuration preset
        
    Returns
    -------
    results : dict
        Dictionary containing comparison results
    """
    try:
        from tensorpac import Pac
        
        # Get configuration
        cfg = TENSORPAC_CONFIGS[config]
        
        # Create bands
        pha_bands, amp_bands = get_tensorpac_bands(
            cfg['n_pha'], cfg['n_amp'], cfg['pha_range'], cfg['amp_range']
        )
        
        # TensorPAC calculation
        pac_tp = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands)
        
        # Ensure numpy array for TensorPAC
        if isinstance(signal, torch.Tensor):
            signal_np = signal.squeeze().numpy()
        else:
            signal_np = signal
            
        if signal_np.ndim == 1:
            signal_np = signal_np.reshape(1, -1)
            
        mi_tp = pac_tp.filterfit(fs, signal_np)
        
        # gPAC calculation
        pac_scaled, pha_freqs, amp_freqs, pac_raw = calculate_pac_tensorpac_compat(
            signal, fs, config=config, return_unscaled=True
        )
        
        # Find peaks
        tp_peak_idx = np.unravel_index(mi_tp.argmax(), mi_tp.shape)
        gp_peak_idx = np.unravel_index(pac_scaled.argmax(), pac_scaled.shape)
        
        results = {
            'tensorpac_max': mi_tp.max(),
            'gpac_max_scaled': pac_scaled.max(),
            'gpac_max_raw': pac_raw.max(),
            'actual_scale_factor': mi_tp.max() / pac_raw.max() if pac_raw.max() > 0 else np.inf,
            'applied_scale_factor': cfg['scale_factor'],
            'tensorpac_peak_pha': pha_bands[tp_peak_idx[0]].mean(),
            'tensorpac_peak_amp': amp_bands[tp_peak_idx[1]].mean(),
            'gpac_peak_pha': pha_freqs[gp_peak_idx[0]],
            'gpac_peak_amp': amp_freqs[gp_peak_idx[1]],
            'peak_location_match': (
                abs(pha_bands[tp_peak_idx[0]].mean() - pha_freqs[gp_peak_idx[0]]) < 2.0
            )
        }
        
        return results
        
    except ImportError:
        return {'error': 'TensorPAC not available for comparison'}


# Convenience function for backward compatibility
def pac_tensorpac_compat(signal, fs, **kwargs):
    """Alias for calculate_pac_tensorpac_compat with default configuration."""
    return calculate_pac_tensorpac_compat(signal, fs, **kwargs)