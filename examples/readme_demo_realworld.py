#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 02:30:00 (ywatanabe)"
# File: readme_demo_realworld.py

__file__ = "readme_demo_realworld.py"

"""
Functionalities:
  - Demonstrates gPAC on real-world EEG data
  - Downloads sample data from MNE-Python
  - Analyzes PAC during cognitive task

Dependencies:
  - scripts: None
  - packages: gpac, mne, torch, numpy, matplotlib

IO:
  - input-files: MNE sample data (auto-downloaded)
  - output-files: ./readme_demo_realworld_out/realworld_pac_analysis.png
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mngs
try:
    import mne
except ImportError:
    print("Installing MNE-Python for EEG data access...")
    os.system("pip install mne")
    import mne

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def load_sample_eeg_data():
    """Load sample EEG data from MNE."""
    # Download sample data if not already present
    sample_data_path = mne.datasets.sample.data_path()
    
    # Load the sample auditory dataset
    raw_fname = os.path.join(sample_data_path, 'MEG', 'sample', 
                            'sample_auditory_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Pick only EEG channels
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False)
    
    # Get events
    events_fname = os.path.join(sample_data_path, 'MEG', 'sample',
                               'sample_auditory_raw-eve.fif')
    events = mne.read_events(events_fname)
    
    return raw, events


def preprocess_for_pac(raw, tmin=-0.2, tmax=0.5, event_id=1):
    """Preprocess EEG data for PAC analysis."""
    # Create epochs around stimulus events
    events = mne.find_events(raw, stim_channel='STI 014')
    
    # Select only left auditory events (event_id=1)
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                       tmin=tmin, tmax=tmax, baseline=None,
                       preload=True, proj=False)
    
    # Apply bandpass filter to remove artifacts
    epochs.filter(l_freq=1., h_freq=200.)
    
    # Get data and info
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    
    return data, sfreq, ch_names, epochs.times


def compute_pac_on_channel(channel_data, sfreq, ch_name, device='cuda'):
    """Compute PAC for a single channel across epochs."""
    import gpac
    
    # Convert to torch tensor
    data_tensor = torch.tensor(channel_data, dtype=torch.float32)
    
    if device == 'cuda' and torch.cuda.is_available():
        data_tensor = data_tensor.cuda()
    
    # Initialize PAC model
    pac_model = gpac.PAC(
        seq_len=data_tensor.shape[-1],
        fs=sfreq,
        pha_start_hz=4.0,    # Theta/alpha
        pha_end_hz=13.0,
        pha_n_bands=6,
        amp_start_hz=30.0,   # Gamma
        amp_end_hz=100.0,
        amp_n_bands=8,
        n_perm=None,
        fp16=False
    )
    
    if device == 'cuda' and torch.cuda.is_available():
        pac_model = pac_model.cuda()
    
    # Compute PAC for each epoch
    pac_values = []
    with torch.no_grad():
        for epoch_data in data_tensor:
            # Add batch and channel dimensions
            epoch_tensor = epoch_data.unsqueeze(0).unsqueeze(0)
            pac_result = pac_model(epoch_tensor)
            pac_values.append(pac_result['pac'].squeeze().cpu().numpy())
    
    # Average across epochs
    pac_mean = np.mean(pac_values, axis=0)
    pac_std = np.std(pac_values, axis=0)
    
    # Get frequency arrays
    pha_freqs = pac_result['phase_frequencies'].numpy()
    amp_freqs = pac_result['amplitude_frequencies'].numpy()
    
    return pac_mean, pac_std, pha_freqs, amp_freqs


def create_realworld_figure(
    raw_segment, times, sfreq,
    pac_results, ch_names,
    save_path
):
    """Create figure showing real-world PAC analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 0.1], 
                          width_ratios=[1, 1, 1])
    
    # Top panel: Raw EEG traces
    ax_raw = fig.add_subplot(gs[0, :])
    n_show = min(5, len(ch_names))  # Show up to 5 channels
    for i in range(n_show):
        offset = i * 50  # µV offset for visualization
        ax_raw.plot(times, raw_segment[i] * 1e6 - offset, 
                   label=ch_names[i], linewidth=0.8)
    ax_raw.set_xlabel('Time (s)')
    ax_raw.set_ylabel('Amplitude (µV)')
    ax_raw.set_title('Sample EEG Data (Auditory Stimulus Response)', fontsize=14)
    ax_raw.legend(loc='upper right', fontsize=8)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus')
    
    # Bottom panels: PAC for different channels
    for idx, (ch_idx, ch_name) in enumerate([(10, 'EEG 011'), 
                                             (30, 'EEG 031'), 
                                             (50, 'EEG 051')]):
        if ch_idx < len(ch_names):
            ax = fig.add_subplot(gs[1, idx])
            pac_mean, pac_std, pha_freqs, amp_freqs = pac_results[ch_idx]
            
            im = ax.imshow(pac_mean.T, aspect='auto', origin='lower',
                          extent=[pha_freqs[0], pha_freqs[-1],
                                 amp_freqs[0], amp_freqs[-1]],
                          cmap='hot', interpolation='bilinear')
            
            ax.set_xlabel('Phase Frequency (Hz)')
            if idx == 0:
                ax.set_ylabel('Amplitude Frequency (Hz)')
            ax.set_title(f'PAC - {ch_names[ch_idx]}', fontsize=12)
            
            # Add contour for significant PAC
            threshold = np.mean(pac_mean) + 2 * np.std(pac_mean)
            ax.contour(pha_freqs, amp_freqs, pac_mean.T, 
                      levels=[threshold], colors='cyan', linewidths=1)
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[2, :])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Modulation Index', fontsize=10)
    
    # Add title
    fig.suptitle('Phase-Amplitude Coupling in Real EEG Data\n' + 
                 'MNE Sample Dataset - Auditory Evoked Response',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}", c='green')
    
    return fig


def main(args):
    """Main function for real-world demo."""
    # Create output directory
    output_dir = './readme_demo_realworld_out'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample EEG data
    print("Loading MNE sample EEG data", c='cyan')
    raw, events = load_sample_eeg_data()
    print(f"Loaded {len(raw.ch_names)} EEG channels")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    
    # Preprocess data
    print("Preprocessing EEG data", c='cyan')
    data, sfreq, ch_names, times = preprocess_for_pac(
        raw, tmin=-0.2, tmax=0.5, event_id=1
    )
    print(f"Epochs shape: {data.shape}")
    print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    
    # Compute PAC for multiple channels
    print("Computing PAC across channels", c='cyan')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    pac_results = {}
    n_channels_to_analyze = min(len(ch_names), 60)  # Analyze subset
    
    for ch_idx in range(0, n_channels_to_analyze, 10):  # Every 10th channel
        ch_name = ch_names[ch_idx]
        print(f"  Processing {ch_name} ({ch_idx+1}/{n_channels_to_analyze})...")
        
        # Get data for this channel
        channel_data = data[:, ch_idx, :]  # (n_epochs, n_times)
        
        # Compute PAC
        pac_mean, pac_std, pha_freqs, amp_freqs = compute_pac_on_channel(
            channel_data, sfreq, ch_name, device
        )
        
        pac_results[ch_idx] = (pac_mean, pac_std, pha_freqs, amp_freqs)
    
    # Create visualization
    print("Creating visualization", c='cyan')
    
    # Get a sample segment for raw data display
    raw_segment = data[0, :, :]  # First epoch, all channels
    
    fig = create_realworld_figure(
        raw_segment, times, sfreq,
        pac_results, ch_names,
        os.path.join(output_dir, 'realworld_pac_analysis.png')
    )
    
    # Summary
    print("Analysis Summary", c='yellow')
    print(f"Analyzed {len(pac_results)} channels")
    print(f"Phase frequencies: {pha_freqs[0]:.1f} - {pha_freqs[-1]:.1f} Hz")
    print(f"Amplitude frequencies: {amp_freqs[0]:.1f} - {amp_freqs[-1]:.1f} Hz")
    
    # Find channel with strongest PAC
    max_pac = 0
    max_ch = None
    for ch_idx, (pac_mean, _, _, _) in pac_results.items():
        if pac_mean.max() > max_pac:
            max_pac = pac_mean.max()
            max_ch = ch_idx
    
    print(f"\nStrongest PAC found in {ch_names[max_ch]}: MI = {max_pac:.3f}")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='Real-world EEG PAC analysis demo')
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix='realworld_demo',
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == '__main__':
    run_main()

# EOF