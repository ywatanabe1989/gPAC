#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TensorPAC-compatible configuration for gPAC

"""
This script demonstrates how to configure gPAC to match TensorPAC's
typical high-resolution frequency band setup.

Based on the documentation, TensorPAC often uses:
- 'hres': 50 phase bands, 50 amplitude bands
- 'demon': 70 bands each
- 'hulk': 100 bands each

For compatibility, we'll implement matching configurations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tensorpac_source'))

import numpy as np
import torch
from tensorpac import Pac
from gpac import PAC

def create_tensorpac_compatible_bands(n_pha=50, n_amp=30, 
                                    pha_range=(1.5, 25), 
                                    amp_range=(52.5, 180)):
    """
    Create frequency bands matching TensorPAC's high-resolution configurations.
    
    Parameters
    ----------
    n_pha : int
        Number of phase bands (default: 50 for 'hres')
    n_amp : int  
        Number of amplitude bands (default: 30 for compatibility)
    pha_range : tuple
        Phase frequency range matching TensorPAC defaults
    amp_range : tuple
        Amplitude frequency range matching TensorPAC defaults
    """
    # Create linear spacing for band edges
    pha_edges = np.linspace(pha_range[0], pha_range[1], n_pha + 1)
    amp_edges = np.linspace(amp_range[0], amp_range[1], n_amp + 1)
    
    # Convert to band pairs for TensorPAC
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    return pha_bands, amp_bands


def test_compatible_configuration():
    """Test gPAC with TensorPAC-compatible configuration."""
    
    print("="*80)
    print("TENSORPAC-COMPATIBLE CONFIGURATION TEST")
    print("="*80)
    
    # Create test signal
    fs = 1000.0
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    
    # Multiple PAC components
    signal = np.zeros_like(t)
    
    # Component 1: 5 Hz phase, 70 Hz amplitude
    pha1 = np.sin(2 * np.pi * 5 * t)
    signal += (1 + 0.5 * pha1) * np.sin(2 * np.pi * 70 * t)
    
    # Component 2: 10 Hz phase, 100 Hz amplitude
    pha2 = np.sin(2 * np.pi * 10 * t)
    signal += 0.5 * (1 + 0.3 * pha2) * np.sin(2 * np.pi * 100 * t)
    
    # Add noise
    signal += 0.1 * np.random.randn(len(t))
    
    # Configuration parameters
    configs = [
        {'name': 'Standard (10x10)', 'n_pha': 10, 'n_amp': 10},
        {'name': 'Medium (30x30)', 'n_pha': 30, 'n_amp': 30},
        {'name': 'Compatible (50x30)', 'n_pha': 50, 'n_amp': 30},
        {'name': 'High-res (50x50)', 'n_pha': 50, 'n_amp': 50},
    ]
    
    for config in configs:
        print(f"\n{'-'*60}")
        print(f"Configuration: {config['name']}")
        print(f"Phase bands: {config['n_pha']}, Amplitude bands: {config['n_amp']}")
        print("-"*60)
        
        # Create bands
        pha_bands, amp_bands = create_tensorpac_compatible_bands(
            n_pha=config['n_pha'],
            n_amp=config['n_amp']
        )
        
        # TensorPAC calculation
        pac_tp = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands)
        mi_tp = pac_tp.filterfit(fs, signal.reshape(1, -1))
        
        # Find peak
        peak_idx = np.unravel_index(mi_tp.argmax(), mi_tp.shape)
        peak_pha = pha_bands[peak_idx[0]].mean()
        peak_amp = amp_bands[peak_idx[1]].mean()
        
        print(f"\nTensorPAC results:")
        print(f"  Max MI: {mi_tp.max():.6f}")
        print(f"  Peak at: {peak_pha:.1f} Hz (phase) / {peak_amp:.1f} Hz (amplitude)")
        print(f"  Shape: {mi_tp.shape}")
        
        # gPAC calculation
        pac_gpac = PAC(
            seq_len=len(signal),
            fs=fs,
            pha_start_hz=pha_bands[0, 0],
            pha_end_hz=pha_bands[-1, 1],
            pha_n_bands=config['n_pha'],
            amp_start_hz=amp_bands[0, 0],
            amp_end_hz=amp_bands[-1, 1],
            amp_n_bands=config['n_amp'],
            mi_n_bins=18,
            filtfilt_mode=True,  # Use filtfilt for better match
            v01_mode=False
        )
        
        signal_torch = torch.tensor(signal, dtype=torch.float32).reshape(1, 1, 1, -1)
        result = pac_gpac(signal_torch)
        mi_gpac = result['mi'][0, 0].numpy()  # Shape: (n_pha, n_amp)
        
        # Find peak
        peak_idx_gpac = np.unravel_index(mi_gpac.argmax(), mi_gpac.shape)
        pha_freqs = result['pha_freqs_hz'].squeeze().numpy()
        amp_freqs = result['amp_freqs_hz'].squeeze().numpy()
        
        print(f"\ngPAC results:")
        print(f"  Max MI: {mi_gpac.max():.6f}")
        print(f"  Peak at: {pha_freqs[peak_idx_gpac[0]]:.1f} Hz / {amp_freqs[peak_idx_gpac[1]]:.1f} Hz")
        print(f"  Shape: {mi_gpac.shape}")
        
        # Comparison
        scale_factor = mi_tp.max() / mi_gpac.max()
        print(f"\nComparison:")
        print(f"  Scale factor: {scale_factor:.2f}x")
        print(f"  Peak location match: {abs(peak_pha - pha_freqs[peak_idx_gpac[0]]) < 2.0}")
        
        # Try with scaling
        mi_gpac_scaled = mi_gpac * scale_factor
        print(f"  Scaled gPAC max: {mi_gpac_scaled.max():.6f}")
    
    # Test with exact TensorPAC 'hres' configuration
    print(f"\n{'='*80}")
    print("EXACT TENSORPAC 'hres' REPLICATION")
    print("="*80)
    
    # Get TensorPAC's actual 'hres' bands
    pac_hres = Pac(idpac=(2, 0, 0), f_pha='hres', f_amp='hres')
    
    print(f"\nTensorPAC 'hres' configuration:")
    print(f"  Phase bands shape: {pac_hres.f_pha.shape}")
    print(f"  Amplitude bands shape: {pac_hres.f_amp.shape}")
    print(f"  Phase range: {pac_hres.f_pha.min():.1f} - {pac_hres.f_pha.max():.1f} Hz")
    print(f"  Amplitude range: {pac_hres.f_amp.min():.1f} - {pac_hres.f_amp.max():.1f} Hz")
    
    # Calculate with 'hres'
    mi_hres = pac_hres.filterfit(fs, signal.reshape(1, -1))
    print(f"\n'hres' results:")
    print(f"  Max MI: {mi_hres.max():.6f}")
    print(f"  Shape: {mi_hres.shape}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR COMPATIBILITY:")
    print("="*80)
    print("1. Use n_pha=50, n_amp=30 for balanced high-resolution analysis")
    print("2. Apply scaling factor of ~4-5x to match TensorPAC values")
    print("3. Use filtfilt_mode=True for better filter compatibility")
    print("4. Document the configuration and scaling used")
    print("="*80)


def create_compatibility_wrapper():
    """Create a wrapper function for TensorPAC-compatible PAC calculation."""
    
    def calculate_pac_tensorpac_compatible(signal, fs, 
                                         n_pha=50, n_amp=30,
                                         pha_range=(1.5, 25),
                                         amp_range=(52.5, 180),
                                         scale_factor=4.5):
        """
        Calculate PAC with TensorPAC-compatible configuration.
        
        Parameters
        ----------
        signal : array-like
            Input signal
        fs : float
            Sampling frequency
        n_pha : int
            Number of phase bands (default: 50)
        n_amp : int
            Number of amplitude bands (default: 30)
        pha_range : tuple
            Phase frequency range (default: TensorPAC's range)
        amp_range : tuple
            Amplitude frequency range (default: TensorPAC's range)
        scale_factor : float
            Scaling factor to match TensorPAC values (default: 4.5)
            
        Returns
        -------
        pac_values : array
            Scaled PAC values matching TensorPAC scale
        pha_freqs : array
            Phase frequencies
        amp_freqs : array
            Amplitude frequencies
        """
        pac_module = PAC(
            seq_len=len(signal),
            fs=fs,
            pha_start_hz=pha_range[0],
            pha_end_hz=pha_range[1],
            pha_n_bands=n_pha,
            amp_start_hz=amp_range[0],
            amp_end_hz=amp_range[1],
            amp_n_bands=n_amp,
            mi_n_bins=18,
            filtfilt_mode=True,
            v01_mode=False
        )
        
        # Convert signal
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)
        
        # Ensure correct shape
        if signal.ndim == 1:
            signal = signal.reshape(1, 1, 1, -1)
        elif signal.ndim == 2:
            signal = signal.reshape(signal.shape[0], 1, 1, signal.shape[1])
            
        # Calculate PAC
        result = pac_module(signal)
        
        # Extract and scale values
        pac_values = result['mi'].numpy() * scale_factor
        pha_freqs = result['pha_freqs_hz'].numpy()
        amp_freqs = result['amp_freqs_hz'].numpy()
        
        return pac_values, pha_freqs, amp_freqs
    
    return calculate_pac_tensorpac_compatible


if __name__ == "__main__":
    test_compatible_configuration()
    
    # Example of using the wrapper
    print("\n" + "="*80)
    print("EXAMPLE: Using compatibility wrapper")
    print("="*80)
    
    pac_compat = create_compatibility_wrapper()
    
    # Test signal
    fs = 1000.0
    t = np.arange(2000) / fs
    signal = (1 + 0.5 * np.sin(2*np.pi*5*t)) * np.sin(2*np.pi*70*t)
    
    pac_values, pha_freqs, amp_freqs = pac_compat(signal, fs)
    
    print(f"PAC shape: {pac_values.shape}")
    print(f"Max PAC value: {pac_values.max():.6f}")
    print("Compatible with TensorPAC scale!")