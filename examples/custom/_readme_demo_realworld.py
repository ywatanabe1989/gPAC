#!/usr/bin/env python3
"""
Real-world demo script for gPAC package using publicly available EEG data.
This demo analyzes phase-amplitude coupling in actual neural recordings.

Uses MNE-Python sample dataset which includes MEG/EEG recordings during
auditory and visual stimuli presentation.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpac import PAC

# Try to import MNE for EEG data loading
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    print("Warning: MNE-Python not available. Install with: pip install mne")
    MNE_AVAILABLE = False

# Try to import tensorpac for comparison
try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    print("Warning: TensorPAC not available for comparison")
    TENSORPAC_AVAILABLE = False


def download_sample_eeg_data():
    """
    Download MNE sample dataset if not already present.
    
    Returns
    -------
    data_path : str
        Path to the sample data
    """
    if not MNE_AVAILABLE:
        return None
        
    # This will download the data if not present (~1.5GB)
    print("Checking for MNE sample data...")
    data_path = mne.datasets.sample.data_path()
    return data_path


def load_eeg_segment(data_path, duration=10.0, start_time=60.0):
    """
    Load a segment of EEG data from the MNE sample dataset.
    
    Parameters
    ----------
    data_path : str
        Path to MNE sample data
    duration : float
        Duration of segment to load in seconds
    start_time : float
        Start time of segment in seconds
        
    Returns
    -------
    eeg_data : np.ndarray
        EEG data array (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    """
    if not MNE_AVAILABLE or data_path is None:
        print("Using synthetic data as fallback...")
        # Create synthetic multi-channel data
        sfreq = 250.0
        n_samples = int(duration * sfreq)
        n_channels = 5
        
        # Create synthetic EEG-like signals with some PAC
        time = np.linspace(0, duration, n_samples)
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Add different frequency components to each channel
            noise = 0.5 * np.random.randn(n_samples)
            
            # Theta oscillation (6 Hz)
            theta = np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
            
            # Gamma oscillation (60 Hz) modulated by theta
            gamma_carrier = np.sin(2 * np.pi * 60 * time)
            modulation = 1 + 0.3 * theta * (ch / n_channels)  # Varying coupling
            gamma = modulation * gamma_carrier
            
            # Alpha rhythm (10 Hz)
            alpha = 0.3 * np.sin(2 * np.pi * 10 * time)
            
            eeg_data[ch] = noise + 0.5 * theta + 0.3 * gamma + alpha
            
        ch_names = [f'EEG {i+1:03d}' for i in range(n_channels)]
        return eeg_data, sfreq, ch_names
    
    # Load actual MNE sample data
    raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    
    # Pick EEG channels only
    raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')
    
    # Get data segment
    start_sample = int(start_time * raw.info['sfreq'])
    stop_sample = start_sample + int(duration * raw.info['sfreq'])
    
    eeg_data, times = raw[:, start_sample:stop_sample]
    
    return eeg_data, raw.info['sfreq'], raw.ch_names


def preprocess_eeg_data(eeg_data, sfreq, highpass=1.0, lowpass=100.0):
    """
    Basic preprocessing of EEG data.
    
    Parameters
    ----------
    eeg_data : np.ndarray
        Raw EEG data (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    highpass : float
        High-pass filter cutoff
    lowpass : float
        Low-pass filter cutoff
        
    Returns
    -------
    filtered_data : np.ndarray
        Preprocessed EEG data
    """
    if MNE_AVAILABLE:
        # Create MNE RawArray for filtering
        info = mne.create_info(
            ch_names=[f'EEG{i}' for i in range(eeg_data.shape[0])],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        
        # Apply band-pass filter
        raw.filter(highpass, lowpass, fir_design='firwin', verbose=False)
        filtered_data = raw.get_data()
    else:
        # Simple detrending as fallback
        from scipy import signal
        filtered_data = signal.detrend(eeg_data, axis=1)
    
    return filtered_data


def calculate_pac_multichannel(data, sfreq, channel_pairs=None):
    """
    Calculate PAC for multiple channel pairs.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    channel_pairs : list of tuples
        List of (phase_ch, amp_ch) pairs. If None, use same channel.
        
    Returns
    -------
    pac_results : dict
        Dictionary with PAC results for each channel/pair
    computation_time : float
        Total computation time
    """
    n_channels, n_samples = data.shape
    
    if channel_pairs is None:
        # Use same channel for phase and amplitude
        channel_pairs = [(i, i) for i in range(min(n_channels, 5))]  # Limit to 5 channels
    
    # PAC parameters
    pha_range = (4, 8)    # Theta band
    amp_range = (30, 80)  # Low gamma band
    n_pha_bands = 5
    n_amp_bands = 10
    
    # Initialize PAC calculator
    pac = PAC(
        seq_len=n_samples,
        fs=sfreq,
        pha_start_hz=pha_range[0],
        pha_end_hz=pha_range[1],
        pha_n_bands=n_pha_bands,
        amp_start_hz=amp_range[0],
        amp_end_hz=amp_range[1],
        amp_n_bands=n_amp_bands,
        trainable=False
    )
    
    pac_results = {}
    start_time = time.time()
    
    for phase_ch, amp_ch in channel_pairs:
        # Prepare signal
        if phase_ch == amp_ch:
            signal = torch.from_numpy(data[phase_ch]).float()
            signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            ch_name = f"Ch{phase_ch}"
        else:
            # For cross-channel PAC, would need to modify approach
            continue
        
        # Calculate PAC
        with torch.no_grad():
            output = pac(signal)
        
        pac_results[ch_name] = {
            'pac_values': output['pac'].squeeze().numpy(),
            'phase_freqs': output['phase_frequencies'].numpy(),
            'amp_freqs': output['amplitude_frequencies'].numpy(),
        }
    
    computation_time = time.time() - start_time
    
    return pac_results, computation_time


def create_realworld_plot(eeg_data, sfreq, pac_results, ch_names):
    """
    Create visualization of real-world PAC analysis.
    
    Parameters
    ----------
    eeg_data : np.ndarray
        Raw EEG data
    sfreq : float
        Sampling frequency
    pac_results : dict
        PAC results for each channel
    ch_names : list
        Channel names
    """
    n_channels = min(len(pac_results), 4)  # Show max 4 channels
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, n_channels, figure=fig, height_ratios=[1, 1.5, 0.1],
                  hspace=0.3, wspace=0.3)
    
    # Time vector for plotting
    time = np.arange(eeg_data.shape[1]) / sfreq
    
    # Plot each channel
    for idx, (ch_name, pac_data) in enumerate(list(pac_results.items())[:n_channels]):
        # Top row: Raw EEG signal (first 2 seconds)
        ax_signal = fig.add_subplot(gs[0, idx])
        ch_idx = int(ch_name.replace('Ch', ''))
        signal_segment = eeg_data[ch_idx, :int(2*sfreq)]
        time_segment = time[:int(2*sfreq)]
        
        ax_signal.plot(time_segment, signal_segment, 'k-', linewidth=0.5)
        ax_signal.set_title(f'{ch_names[ch_idx] if ch_idx < len(ch_names) else ch_name}',
                           fontsize=10, fontweight='bold')
        ax_signal.set_xlabel('Time (s)', fontsize=8)
        ax_signal.set_ylabel('Amplitude (µV)', fontsize=8)
        ax_signal.tick_params(labelsize=7)
        ax_signal.grid(True, alpha=0.3)
        
        # Middle row: PAC comodulogram
        ax_pac = fig.add_subplot(gs[1, idx])
        pac_values = pac_data['pac_values']
        phase_freqs = pac_data['phase_freqs']
        amp_freqs = pac_data['amp_freqs']
        
        im = ax_pac.imshow(pac_values.T, aspect='auto', origin='lower',
                          extent=[phase_freqs[0], phase_freqs[-1],
                                 amp_freqs[0], amp_freqs[-1]],
                          cmap='hot', interpolation='bilinear')
        
        ax_pac.set_xlabel('Phase Frequency (Hz)', fontsize=8)
        if idx == 0:
            ax_pac.set_ylabel('Amplitude Frequency (Hz)', fontsize=8)
        ax_pac.tick_params(labelsize=7)
        
        # Find peak PAC
        max_idx = np.unravel_index(pac_values.argmax(), pac_values.shape)
        max_phase_freq = phase_freqs[max_idx[0]]
        max_amp_freq = amp_freqs[max_idx[1]]
        max_pac_value = pac_values[max_idx]
        
        # Mark peak with cross
        ax_pac.plot(max_phase_freq, max_amp_freq, 'w+', markersize=10, markeredgewidth=2)
        ax_pac.text(0.05, 0.95, f'Peak: {max_pac_value:.3f}\n@{max_phase_freq:.1f}-{max_amp_freq:.1f}Hz',
                   transform=ax_pac.transAxes, fontsize=7, color='white',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add shared colorbar
    cbar_ax = fig.add_subplot(gs[2, :])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('PAC Value (MI)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Main title
    fig.suptitle('Phase-Amplitude Coupling in Real EEG Data', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """Main demo function for real-world data."""
    print("=" * 70)
    print("gPAC Real-World Demo: PAC Analysis of EEG Data")
    print("=" * 70)
    
    # Download/load data
    if MNE_AVAILABLE:
        print("\nAttempting to load MNE sample dataset...")
        data_path = download_sample_eeg_data()
    else:
        print("\nMNE not available, using synthetic EEG-like data...")
        data_path = None
    
    # Load EEG segment
    print("Loading EEG data segment...")
    eeg_data, sfreq, ch_names = load_eeg_segment(data_path, duration=10.0)
    print(f"  Loaded {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples")
    print(f"  Sampling frequency: {sfreq} Hz")
    
    # Preprocess data
    print("\nPreprocessing EEG data...")
    print("  Applying band-pass filter (1-100 Hz)")
    filtered_data = preprocess_eeg_data(eeg_data, sfreq)
    
    # Calculate PAC
    print("\nCalculating PAC for multiple channels...")
    print("  Phase frequencies: 4-8 Hz (Theta)")
    print("  Amplitude frequencies: 30-80 Hz (Low Gamma)")
    
    pac_results, computation_time = calculate_pac_multichannel(filtered_data, sfreq)
    
    print(f"\nComputation completed in {computation_time:.3f} seconds")
    print(f"Analyzed {len(pac_results)} channels")
    
    # Report findings
    print("\nPAC Analysis Summary:")
    print("-" * 50)
    for ch_name, pac_data in pac_results.items():
        pac_values = pac_data['pac_values']
        max_pac = pac_values.max()
        max_idx = np.unravel_index(pac_values.argmax(), pac_values.shape)
        peak_phase = pac_data['phase_freqs'][max_idx[0]]
        peak_amp = pac_data['amp_freqs'][max_idx[1]]
        
        print(f"{ch_name}: Peak PAC = {max_pac:.4f} at {peak_phase:.1f}-{peak_amp:.1f} Hz")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = create_realworld_plot(filtered_data, sfreq, pac_results, ch_names)
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'readme_demo_realworld_output.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    print("\nDemo completed successfully!")
    
    # Additional analysis info
    print("\nInterpretation Guide:")
    print("-" * 50)
    print("- PAC values indicate coupling strength between slow and fast rhythms")
    print("- Theta-gamma coupling (4-8 Hz phase, 30-80 Hz amplitude) is common in:")
    print("  * Memory processing")
    print("  * Cognitive control")
    print("  * Sensory processing")
    print("- Higher PAC values suggest stronger functional coupling")


if __name__ == "__main__":
    main()