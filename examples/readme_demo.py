#!/usr/bin/env python3
"""
Demo script for gPAC package showcasing PAC analysis with synthetic data.
Generates comparison between gPAC and TensorPAC implementations.

This demo creates:
- Synthetic data with known PAC coupling
- PAC calculations using both gPAC and TensorPAC
- Visualization comparing results
- Performance benchmarks
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpac import PAC, SyntheticDataGenerator

# Try to import tensorpac for comparison
try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    print("Warning: TensorPAC not available for comparison")
    TENSORPAC_AVAILABLE = False


def generate_synthetic_pac_signal(n_seconds=10, fs=250, phase_freq=6.0, amp_freq=60.0, 
                                 coupling_strength=0.5, noise_level=0.1):
    """
    Generate synthetic signal with known PAC coupling.
    
    Parameters
    ----------
    n_seconds : float
        Duration of signal in seconds
    fs : float
        Sampling frequency in Hz
    phase_freq : float
        Frequency for phase signal (Hz)
    amp_freq : float
        Frequency for amplitude signal (Hz)
    coupling_strength : float
        Strength of PAC coupling (0-1)
    noise_level : float
        Amount of noise to add (0-1)
    
    Returns
    -------
    signal : np.ndarray
        Synthetic signal with PAC
    time : np.ndarray
        Time vector
    """
    n_samples = int(n_seconds * fs)
    time = np.linspace(0, n_seconds, n_samples)
    
    # Generate phase signal
    phase_signal = np.sin(2 * np.pi * phase_freq * time)
    
    # Generate amplitude signal with modulation
    carrier = np.sin(2 * np.pi * amp_freq * time)
    modulation = 1 + coupling_strength * phase_signal
    amp_signal = modulation * carrier
    
    # Combine and add noise
    signal = amp_signal + noise_level * np.random.randn(n_samples)
    
    return signal, time


def calculate_gpac(signal, fs=250, pha_range=(4, 8), amp_range=(30, 100), 
                   n_pha_bands=5, n_amp_bands=10):
    """
    Calculate PAC using gPAC.
    
    Parameters
    ----------
    signal : torch.Tensor or np.ndarray
        Input signal
    fs : float
        Sampling frequency
    pha_range : tuple
        Phase frequency range (start, end) in Hz
    amp_range : tuple
        Amplitude frequency range (start, end) in Hz
    n_pha_bands : int
        Number of phase bands
    n_amp_bands : int
        Number of amplitude bands
    
    Returns
    -------
    pac_values : torch.Tensor
        PAC values matrix
    pha_freqs : torch.Tensor
        Phase frequencies
    amp_freqs : torch.Tensor
        Amplitude frequencies
    computation_time : float
        Time taken for computation
    """
    # Convert to tensor if needed
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float()
    
    # Ensure correct shape (batch, channels, time)
    if signal.ndim == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.ndim == 2:
        signal = signal.unsqueeze(0)
    
    seq_len = signal.shape[-1]
    
    # Initialize PAC calculator
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=pha_range[0],
        pha_end_hz=pha_range[1],
        pha_n_bands=n_pha_bands,
        amp_start_hz=amp_range[0],
        amp_end_hz=amp_range[1],
        amp_n_bands=n_amp_bands,
        trainable=False
    )
    
    # Time the computation
    start_time = time.time()
    with torch.no_grad():
        output = pac(signal)
    computation_time = time.time() - start_time
    
    pac_values = output['pac'].squeeze().numpy()
    pha_freqs = output['phase_frequencies'].numpy()
    amp_freqs = output['amplitude_frequencies'].numpy()
    
    return pac_values, pha_freqs, amp_freqs, computation_time


def calculate_tensorpac(signal, fs=250, pha_range=(4, 8), amp_range=(30, 100),
                       n_pha_bands=5, n_amp_bands=10):
    """
    Calculate PAC using TensorPAC.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    pha_range : tuple
        Phase frequency range (start, end) in Hz
    amp_range : tuple
        Amplitude frequency range (start, end) in Hz
    n_pha_bands : int
        Number of phase bands
    n_amp_bands : int
        Number of amplitude bands
    
    Returns
    -------
    pac_values : np.ndarray
        PAC values matrix
    pha_freqs : np.ndarray
        Phase frequencies
    amp_freqs : np.ndarray
        Amplitude frequencies
    computation_time : float
        Time taken for computation
    """
    if not TENSORPAC_AVAILABLE:
        return None, None, None, None
    
    # Ensure correct shape for TensorPAC (n_epochs, n_times)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Create phase and amplitude frequency vectors with explicit bands
    pha_freqs = np.linspace(pha_range[0], pha_range[1], n_pha_bands)
    amp_freqs = np.linspace(amp_range[0], amp_range[1], n_amp_bands)
    
    # Create frequency bands for TensorPAC
    pha_bands = [(f-0.5, f+0.5) for f in pha_freqs]
    amp_bands = [(f-2, f+2) for f in amp_freqs]
    
    # Initialize TensorPAC with explicit frequency bands
    pac = TensorPAC(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands, dcomplex='hilbert')
    
    # Time the computation
    start_time = time.time()
    xpac = pac.filterfit(fs, signal, n_jobs=1)
    computation_time = time.time() - start_time
    
    # Extract PAC values - shape should be (n_amp, n_pha, n_epochs, n_times)
    # We want the average across time for each frequency pair
    pac_values = np.squeeze(xpac)
    
    # Handle different output shapes
    if pac_values.ndim == 4:
        # Average across epochs and time
        pac_values = pac_values.mean(axis=(2, 3))
    elif pac_values.ndim == 3:
        # Average across time
        pac_values = pac_values.mean(axis=2)
    elif pac_values.ndim == 2:
        # Already in correct format
        pass
    else:
        print(f"Warning: Unexpected TensorPAC output shape: {pac_values.shape}")
        pac_values = np.zeros((n_amp_bands, n_pha_bands))
    
    # Transpose to match gPAC format (pha x amp)
    pac_values = pac_values.T
    
    return pac_values, pha_freqs, amp_freqs, computation_time


def create_comparison_plot(signal, time, gpac_results, tensorpac_results, 
                          phase_freq=6.0, amp_freq=60.0):
    """
    Create comparison plot between gPAC and TensorPAC.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    time : np.ndarray
        Time vector
    gpac_results : tuple
        Results from gPAC (pac_values, pha_freqs, amp_freqs, time)
    tensorpac_results : tuple
        Results from TensorPAC (pac_values, pha_freqs, amp_freqs, time)
    phase_freq : float
        True phase frequency for marking
    amp_freq : float
        True amplitude frequency for marking
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.5, 1.5])
    
    # Top: Raw signal
    ax_signal = fig.add_subplot(gs[0, :])
    ax_signal.plot(time[:1000], signal[:1000], 'k-', linewidth=0.5)
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.set_title('Raw Signal (first 4 seconds)', fontsize=14, fontweight='bold')
    ax_signal.grid(True, alpha=0.3)
    
    # Bottom left: gPAC results
    ax_gpac = fig.add_subplot(gs[1:, 0])
    gpac_pac, gpac_pha, gpac_amp, gpac_time = gpac_results
    
    if gpac_pac is not None:
        im1 = ax_gpac.imshow(gpac_pac.T, aspect='auto', origin='lower',
                            extent=[gpac_pha[0], gpac_pha[-1], gpac_amp[0], gpac_amp[-1]],
                            cmap='hot', interpolation='bilinear')
        ax_gpac.set_xlabel('Phase Frequency (Hz)')
        ax_gpac.set_ylabel('Amplitude Frequency (Hz)')
        ax_gpac.set_title(f'gPAC\n(Time: {gpac_time:.3f}s)', fontsize=12, fontweight='bold')
        
        # Mark true frequencies
        ax_gpac.axvline(phase_freq, color='cyan', linestyle='--', alpha=0.7)
        ax_gpac.axhline(amp_freq, color='cyan', linestyle='--', alpha=0.7)
        
        plt.colorbar(im1, ax=ax_gpac, label='PAC Value')
    
    # Bottom center: TensorPAC results
    ax_tensorpac = fig.add_subplot(gs[1:, 1])
    tensorpac_pac, tensorpac_pha, tensorpac_amp, tensorpac_time = tensorpac_results
    
    if tensorpac_pac is not None and TENSORPAC_AVAILABLE:
        im2 = ax_tensorpac.imshow(tensorpac_pac.T, aspect='auto', origin='lower',
                                 extent=[tensorpac_pha[0], tensorpac_pha[-1], 
                                        tensorpac_amp[0], tensorpac_amp[-1]],
                                 cmap='hot', interpolation='bilinear')
        ax_tensorpac.set_xlabel('Phase Frequency (Hz)')
        ax_tensorpac.set_ylabel('Amplitude Frequency (Hz)')
        ax_tensorpac.set_title(f'TensorPAC\n(Time: {tensorpac_time:.3f}s)', 
                             fontsize=12, fontweight='bold')
        
        # Mark true frequencies
        ax_tensorpac.axvline(phase_freq, color='cyan', linestyle='--', alpha=0.7)
        ax_tensorpac.axhline(amp_freq, color='cyan', linestyle='--', alpha=0.7)
        
        plt.colorbar(im2, ax=ax_tensorpac, label='PAC Value')
    else:
        ax_tensorpac.text(0.5, 0.5, 'TensorPAC not available', 
                         ha='center', va='center', transform=ax_tensorpac.transAxes)
        ax_tensorpac.set_xlabel('Phase Frequency (Hz)')
        ax_tensorpac.set_ylabel('Amplitude Frequency (Hz)')
    
    # Bottom right: Difference
    ax_diff = fig.add_subplot(gs[1:, 2])
    
    if gpac_pac is not None and tensorpac_pac is not None and TENSORPAC_AVAILABLE:
        # Interpolate to same grid if needed
        diff = gpac_pac - tensorpac_pac
        
        im3 = ax_diff.imshow(diff.T, aspect='auto', origin='lower',
                            extent=[gpac_pha[0], gpac_pha[-1], gpac_amp[0], gpac_amp[-1]],
                            cmap='RdBu_r', interpolation='bilinear',
                            vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        ax_diff.set_xlabel('Phase Frequency (Hz)')
        ax_diff.set_ylabel('Amplitude Frequency (Hz)')
        ax_diff.set_title('Difference\n(gPAC - TensorPAC)', fontsize=12, fontweight='bold')
        
        # Mark true frequencies
        ax_diff.axvline(phase_freq, color='black', linestyle='--', alpha=0.7)
        ax_diff.axhline(amp_freq, color='black', linestyle='--', alpha=0.7)
        
        plt.colorbar(im3, ax=ax_diff, label='Difference')
    else:
        ax_diff.text(0.5, 0.5, 'Comparison not available', 
                    ha='center', va='center', transform=ax_diff.transAxes)
        ax_diff.set_xlabel('Phase Frequency (Hz)')
        ax_diff.set_ylabel('Amplitude Frequency (Hz)')
    
    plt.tight_layout()
    return fig


def main():
    """Main demo function."""
    print("=" * 60)
    print("gPAC Demo: Phase-Amplitude Coupling Analysis")
    print("=" * 60)
    
    # Parameters
    n_seconds = 10
    fs = 250
    phase_freq = 6.0  # Hz (theta band)
    amp_freq = 60.0   # Hz (gamma band)
    coupling_strength = 0.7
    noise_level = 0.1
    
    # PAC calculation parameters
    pha_range = (4, 8)    # Theta band
    amp_range = (30, 100) # Gamma band
    n_pha_bands = 10
    n_amp_bands = 20
    
    print(f"\nGenerating synthetic signal:")
    print(f"  Duration: {n_seconds} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Phase frequency: {phase_freq} Hz")
    print(f"  Amplitude frequency: {amp_freq} Hz")
    print(f"  Coupling strength: {coupling_strength}")
    print(f"  Noise level: {noise_level}")
    
    # Generate synthetic signal
    signal, time = generate_synthetic_pac_signal(
        n_seconds=n_seconds,
        fs=fs,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    
    print(f"\nCalculating PAC:")
    print(f"  Phase range: {pha_range} Hz")
    print(f"  Amplitude range: {amp_range} Hz")
    print(f"  Phase bands: {n_pha_bands}")
    print(f"  Amplitude bands: {n_amp_bands}")
    
    # Calculate PAC using gPAC
    print("\n  Computing with gPAC...", end='', flush=True)
    gpac_results = calculate_gpac(
        signal, fs, pha_range, amp_range, n_pha_bands, n_amp_bands
    )
    print(f" Done! (Time: {gpac_results[3]:.3f}s)")
    
    # Calculate PAC using TensorPAC
    if TENSORPAC_AVAILABLE:
        print("  Computing with TensorPAC...", end='', flush=True)
        tensorpac_results = calculate_tensorpac(
            signal, fs, pha_range, amp_range, n_pha_bands, n_amp_bands
        )
        print(f" Done! (Time: {tensorpac_results[3]:.3f}s)")
    else:
        tensorpac_results = (None, None, None, None)
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    print(f"gPAC computation time: {gpac_results[3]:.3f} seconds")
    if TENSORPAC_AVAILABLE and tensorpac_results[3] is not None:
        print(f"TensorPAC computation time: {tensorpac_results[3]:.3f} seconds")
        speedup = tensorpac_results[3] / gpac_results[3]
        print(f"Speedup factor: {speedup:.2f}x")
    
    # Ground truth PAC location
    print(f"\nGround truth PAC location:")
    print(f"  Phase: {phase_freq} Hz")
    print(f"  Amplitude: {amp_freq} Hz")
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    fig = create_comparison_plot(
        signal, time, gpac_results, tensorpac_results, phase_freq, amp_freq
    )
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'readme_demo_output.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as individual frames for GIF if needed
    # (This would require additional implementation)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()