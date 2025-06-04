#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 05:33:47 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/gPAC/examples/example_bandpass_filter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_bandpass_filter.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""
Example: Bandpass Filtering with gPAC

This example demonstrates:
- Using gPAC's BandPassFilter for PAC analysis preparation
- Filtering signals into phase (low frequency) and amplitude (high frequency) bands
- Visualizing the filtered outputs
- Understanding the multi-band filtering approach

All outputs are saved using mngs framework conventions.
"""

import sys
import argparse
import mngs
import numpy as np
import torch
import matplotlib.pyplot as plt
from gpac import BandPassFilter
from scipy import signal as sp_signal

# Parameters
CONFIG = mngs.gen.load_configs()


def generate_pac_signal(fs=1000, duration=5.0):
    """Generate a synthetic signal with phase-amplitude coupling."""
    t = np.arange(0, duration, 1 / fs)
    
    # Low frequency (phase) component - theta (6 Hz)
    phase_signal = np.sin(2 * np.pi * 6 * t)
    phase = np.angle(sp_signal.hilbert(phase_signal))
    
    # High frequency (amplitude) component - gamma (80 Hz)
    # Modulated by the phase of the low frequency
    modulation = 0.5 * (1 + np.cos(phase))  # Amplitude modulation
    amp_signal = modulation * np.sin(2 * np.pi * 80 * t)
    
    # Combine with additional frequency components
    signal = (
        phase_signal  # 6 Hz (theta) - phase provider
        + 0.3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha)
        + amp_signal  # 80 Hz (gamma) - amplitude modulated
        + 0.2 * np.sin(2 * np.pi * 100 * t)  # 100 Hz (high gamma)
        + 0.1 * np.random.randn(len(t))  # noise
    )
    
    return signal, t


@mngs.plt.subplots(nrows=3, ncols=1, figsize=(12, 10), facecolor="white")
def visualize_bandpass_results(fig, t, original_signal, filtered_output, filter_module):
    """Visualize the BandPassFilter output."""
    axes = fig.axes
    
    # Plot original signal
    ax = axes[0]
    ax.plot(t[:1000], original_signal[:1000], "b-", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original Signal (first second)")
    ax.grid(True, alpha=0.3)
    
    # Extract dimensions
    bs, n_seg, n_bands, seq_len = filtered_output.shape
    n_pha_bands = filter_module.pha_n_bands
    n_amp_bands = filter_module.amp_n_bands
    
    # Plot phase band filtered signals (show 3 bands)
    ax = axes[1]
    phase_bands_to_show = [0, n_pha_bands//2, n_pha_bands-1]
    colors = ['g', 'orange', 'purple']
    for i, band_idx in enumerate(phase_bands_to_show):
        if band_idx < n_pha_bands:
            freq = filter_module.pha_mids[band_idx].item()
            signal_band = filtered_output[0, 0, band_idx, :1000].cpu().numpy()
            ax.plot(t[:1000], signal_band, colors[i], alpha=0.7, 
                   label=f'Phase band {freq:.1f} Hz')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Phase-Filtered Signals (Low Frequency Bands)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot amplitude band filtered signals (show 3 bands)
    ax = axes[2]
    amp_bands_to_show = [0, n_amp_bands//2, n_amp_bands-1]
    colors = ['r', 'brown', 'pink']
    for i, band_idx in enumerate(amp_bands_to_show):
        if band_idx < n_amp_bands:
            freq = filter_module.amp_mids[band_idx].item()
            # Amplitude bands start after phase bands in the output
            signal_band = filtered_output[0, 0, n_pha_bands + band_idx, :1000].cpu().numpy()
            ax.plot(t[:1000], signal_band, colors[i], alpha=0.7,
                   label=f'Amp band {freq:.1f} Hz')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude-Filtered Signals (High Frequency Bands)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


@mngs.plt.subplots(nrows=2, ncols=1, figsize=(12, 8), facecolor="white")
def visualize_frequency_bands(fig, filter_module):
    """Visualize the frequency band configuration."""
    axes = fig.axes
    
    # Get filter bank info
    bands_info = filter_module.get_filter_banks()
    pha_bands = bands_info['pha_bands'].cpu().numpy()
    amp_bands = bands_info['amp_bands'].cpu().numpy()
    
    # Plot phase frequency bands
    ax = axes[0]
    for i in range(len(pha_bands)):
        low, high = pha_bands[i]
        center = (low + high) / 2
        width = high - low
        ax.bar(center, 1, width=width, alpha=0.3, color='green', edgecolor='darkgreen')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Filter Response")
    ax.set_title(f"Phase Frequency Bands ({filter_module.pha_n_bands} bands)")
    ax.set_xlim(0, filter_module.pha_end_hz * 1.2)
    ax.grid(True, alpha=0.3)
    
    # Plot amplitude frequency bands
    ax = axes[1]
    for i in range(len(amp_bands)):
        low, high = amp_bands[i]
        center = (low + high) / 2
        width = high - low
        ax.bar(center, 1, width=width, alpha=0.3, color='red', edgecolor='darkred')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Filter Response")
    ax.set_title(f"Amplitude Frequency Bands ({filter_module.amp_n_bands} bands)")
    ax.set_xlim(0, filter_module.amp_end_hz * 1.2)
    ax.grid(True, alpha=0.3)
    
    return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bandpass filtering example with gPAC")
    parser.add_argument("--fs", type=int, default=1000, help="Sampling frequency")
    parser.add_argument("--duration", type=float, default=5.0, help="Signal duration in seconds")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', or 'auto'")
    return parser.parse_args()

def run_main(args):
    """Main example function."""
    # Start
    CONFIG, CC, sdir = mngs.gen.start(__file__, args=args, pyplot_backend="Agg")
    
    # Set random seed
    mngs.gen.fix_seeds(42)
    
    # Parameters
    fs = args.fs
    duration = args.duration
    seq_len = int(fs * duration)  # Sequence length
    
    # Generate test signal with PAC
    print("Generating synthetic PAC signal...")
    signal, t = generate_pac_signal(fs=fs, duration=duration)
    
    # Initialize BandPassFilter
    print("\nInitializing BandPassFilter...")
    print("  Phase bands: 2-20 Hz (10 bands)")
    print("  Amplitude bands: 60-160 Hz (10 bands)")
    
    filter_module = BandPassFilter(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=10,  # 10 phase bands
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=10,  # 10 amplitude bands
        fp16=False,
        trainable=False  # Use static filters for this example
    )
    
    # Convert signal to tensor with correct shape
    # BandPassFilter expects: (batch_size, n_segments, seq_len)
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    
    # Apply filtering
    print("\nApplying bandpass filtering...")
    with torch.no_grad():
        # Output shape: (batch_size, n_segments, n_bands, seq_len)
        # where n_bands = pha_n_bands + amp_n_bands
        filtered_output = filter_module(signal_tensor)
    
    print(f"  Input shape: {signal_tensor.shape}")
    print(f"  Output shape: {filtered_output.shape}")
    print(f"  Total bands: {filtered_output.shape[2]} ({filter_module.pha_n_bands} phase + {filter_module.amp_n_bands} amplitude)")
    
    # Output directory already created by mngs.gen.start
    
    # Visualize frequency bands
    print("\nVisualizing frequency band configuration...")
    fig_bands = visualize_frequency_bands(filter_module)
    spath = sdir / "frequency_bands.png"
    mngs.io.save(fig_bands, spath)
    print(f"  Frequency bands saved to: {spath}")
    
    # Visualize filtering results
    print("\nVisualizing filtering results...")
    fig_results = visualize_bandpass_results(t, signal, filtered_output, filter_module)
    spath = sdir / "bandpass_filtering_results.png"
    mngs.io.save(fig_results, spath)
    print(f"  Filtering results saved to: {spath}")
    
    # Save filter information
    filter_info = {
        "sampling_frequency": fs,
        "sequence_length": seq_len,
        "phase_bands": {
            "start_hz": filter_module.pha_start_hz,
            "end_hz": filter_module.pha_end_hz,
            "n_bands": filter_module.pha_n_bands,
            "center_frequencies": filter_module.pha_mids.tolist()
        },
        "amplitude_bands": {
            "start_hz": filter_module.amp_start_hz,
            "end_hz": filter_module.amp_end_hz,
            "n_bands": filter_module.amp_n_bands,
            "center_frequencies": filter_module.amp_mids.tolist()
        },
        "trainable": filter_module.trainable,
        "output_shape": f"(batch_size, n_segments, {filter_module.pha_n_bands + filter_module.amp_n_bands}, {seq_len})"
    }
    spath = sdir / "filter_info.yaml"
    mngs.io.save(filter_info, spath)
    print(f"  Filter info saved to: {spath}")
    
    # Demonstrate accessing individual bands
    print("\nAccessing individual filtered bands:")
    print(f"  Phase bands (0 to {filter_module.pha_n_bands-1}): Low frequency components")
    print(f"  Amplitude bands ({filter_module.pha_n_bands} to {filter_module.pha_n_bands + filter_module.amp_n_bands-1}): High frequency components")
    
    # Example: Extract one phase band and one amplitude band
    phase_band_5hz = filtered_output[0, 0, 1, :].cpu().numpy()  # ~5 Hz band
    amp_band_80hz = filtered_output[0, 0, filter_module.pha_n_bands + 2, :].cpu().numpy()  # ~80 Hz band
    
    print(f"\n  Example phase band power: {np.var(phase_band_5hz):.3f}")
    print(f"  Example amplitude band power: {np.var(amp_band_80hz):.3f}")
    
    print("\nExample completed successfully!")
    print("\nNote: This BandPassFilter is designed for PAC analysis.")
    print("It filters the input into multiple phase and amplitude bands simultaneously.")
    print("The output can be fed to PAC computation modules to analyze cross-frequency coupling.")
    
    # Close
    mngs.gen.close()


if __name__ == "__main__":
    args = parse_args()
    run_main(args)

# EOF
