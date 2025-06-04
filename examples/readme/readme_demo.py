#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 02:00:00 (ywatanabe)"
# File: readme_demo.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/readme/readme_demo.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates gPAC vs TensorPAC comparison
  - Creates publication-quality figure for README
  - Shows performance benchmarks

Dependencies:
  - scripts: None
  - packages: gpac, tensorpac, torch, numpy, matplotlib

IO:
  - input-files: None
  - output-files: ./readme_demo_out/comparison_figure.png
"""

"""Imports"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import mngs

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def generate_synthetic_pac_signal(
    n_seconds=2,
    fs=512,
    f_phase=6,
    f_amp=80,
    pac_strength=0.5,
    noise_level=0.1
):
    """Generate synthetic signal with known PAC."""
    t = np.linspace(0, n_seconds, int(n_seconds * fs))
    
    # Phase signal (low frequency)
    phase_signal = np.sin(2 * np.pi * f_phase * t)
    
    # Amplitude signal (high frequency)
    amp_signal = np.sin(2 * np.pi * f_amp * t)
    
    # Modulate amplitude by phase
    modulation = 1 + pac_strength * (1 + np.sin(2 * np.pi * f_phase * t)) / 2
    modulated_signal = amp_signal * modulation
    
    # Combine signals
    signal = phase_signal + modulated_signal
    
    # Add noise
    signal += noise_level * np.random.randn(len(signal))
    
    return signal, t


def compute_pac_gpac(signal, fs, device='cuda'):
    """Compute PAC using gPAC."""
    import gpac
    
    # Convert to torch tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    if device == 'cuda' and torch.cuda.is_available():
        signal_tensor = signal_tensor.cuda()
    
    # Initialize PAC with moderate resolution for demo
    # Match TensorPAC's frequency bands for fair comparison
    pac_model = gpac.PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=18.0,  # To get 9 bands like TensorPAC
        pha_n_bands=9,  # Match TensorPAC
        amp_start_hz=40.0,
        amp_end_hz=148.0,  # To get 10 bands like TensorPAC
        amp_n_bands=10,  # Match TensorPAC
        n_perm=None,  # No surrogates for speed
        fp16=False
    )
    
    if device == 'cuda' and torch.cuda.is_available():
        pac_model = pac_model.cuda()
    
    # Warm-up run (important for fair timing)
    with torch.no_grad():
        _ = pac_model(signal_tensor)
    
    # Time the computation with proper synchronization
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        pac_results = pac_model(signal_tensor)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    pac_values = pac_results['pac'].squeeze().cpu().numpy()
    comp_time = time.time() - start_time
    
    # Get frequency arrays
    pha_freqs = pac_results['phase_frequencies'].numpy()
    amp_freqs = pac_results['amplitude_frequencies'].numpy()
    
    return pac_values, pha_freqs, amp_freqs, comp_time


def compute_pac_tensorpac(signal, fs):
    """Compute PAC using TensorPAC."""
    from tensorpac import Pac
    
    # Reshape signal for tensorpac (needs 2D: n_epochs x n_times)
    signal_2d = signal.reshape(1, -1)
    
    # Define frequency vectors (reduced resolution for speed)
    f_pha = np.arange(2, 20, 2)  # 9 bands: [2, 4, 6, 8, 10, 12, 14, 16, 18]
    f_amp = np.arange(40, 160, 12)  # 10 bands: [40, 52, 64, 76, 88, 100, 112, 124, 136, 148]
    
    # Initialize with frequency vectors
    pac = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert')  # MI method
    
    # Time the computation
    start_time = time.time()
    pac_values = pac.filterfit(fs, signal_2d, n_perm=0)  # No permutations for speed
    comp_time = time.time() - start_time
    
    # Remove the epoch dimension
    pac_values = pac_values.squeeze()
    
    # Get actual frequency arrays from TensorPAC (it might modify them)
    actual_f_pha = pac.f_pha
    actual_f_amp = pac.f_amp
    
    # Handle TensorPAC returning 2D arrays for frequency bands
    if actual_f_pha.ndim == 2:
        # Take the center frequency of each band
        actual_f_pha = actual_f_pha.mean(axis=1)
    if actual_f_amp.ndim == 2:
        # Take the center frequency of each band
        actual_f_amp = actual_f_amp.mean(axis=1)
    
    return pac_values, actual_f_pha, actual_f_amp, comp_time


def create_comparison_figure(
    signal, t, pac_gpac, pac_tensorpac, 
    pha_freqs_gpac, amp_freqs_gpac,
    pha_freqs_tp, amp_freqs_tp,
    time_gpac, time_tensorpac,
    save_path
):
    """Create 4-panel comparison figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 0.1], width_ratios=[1, 1, 1])
    
    # Top panel: Raw signal
    ax_signal = fig.add_subplot(gs[0, :])
    ax_signal.plot(t[:1024], signal[:1024], 'k-', linewidth=0.8)
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.set_title('Synthetic PAC Signal (Phase: 6 Hz, Amplitude: 80 Hz)', fontsize=14)
    ax_signal.grid(True, alpha=0.3)
    
    # Bottom left: gPAC
    ax_gpac = fig.add_subplot(gs[1, 0])
    im1 = ax_gpac.imshow(pac_gpac.T, aspect='auto', origin='lower',
                         extent=[pha_freqs_gpac[0], pha_freqs_gpac[-1], 
                                amp_freqs_gpac[0], amp_freqs_gpac[-1]],
                         cmap='hot', interpolation='bilinear')
    ax_gpac.set_xlabel('Phase Frequency (Hz)')
    ax_gpac.set_ylabel('Amplitude Frequency (Hz)')
    ax_gpac.set_title(f'gPAC (Time: {time_gpac:.3f}s)', fontsize=12)
    ax_gpac.axvline(6, color='cyan', linestyle='--', alpha=0.5)
    ax_gpac.axhline(80, color='cyan', linestyle='--', alpha=0.5)
    
    # Bottom center: TensorPAC
    ax_tensorpac = fig.add_subplot(gs[1, 1])
    im2 = ax_tensorpac.imshow(pac_tensorpac.T, aspect='auto', origin='lower',
                              extent=[pha_freqs_tp[0], pha_freqs_tp[-1], 
                                     amp_freqs_tp[0], amp_freqs_tp[-1]],
                              cmap='hot', interpolation='bilinear')
    ax_tensorpac.set_xlabel('Phase Frequency (Hz)')
    ax_tensorpac.set_ylabel('')
    ax_tensorpac.set_title(f'TensorPAC (Time: {time_tensorpac:.3f}s)', fontsize=12)
    ax_tensorpac.axvline(6, color='cyan', linestyle='--', alpha=0.5)
    ax_tensorpac.axhline(80, color='cyan', linestyle='--', alpha=0.5)
    
    # Bottom right: Difference
    ax_diff = fig.add_subplot(gs[1, 2])
    
    # Interpolate to common grid for difference
    from scipy.interpolate import RegularGridInterpolator
    
    # Note: gPAC returns (n_pha, n_amp) while TensorPAC returns (n_amp, n_pha)
    # We need to handle the different shapes and orientations
    
    # Create interpolator for gPAC data
    interpolator = RegularGridInterpolator(
        (pha_freqs_gpac, amp_freqs_gpac), 
        pac_gpac,  # (n_pha, n_amp)
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    # Create meshgrid for target frequencies
    # TensorPAC shape is (n_amp, n_pha), so we need to match that
    # Get the actual shape from TensorPAC
    n_amp_tp, n_pha_tp = pac_tensorpac.shape
    
    # Use only the frequencies that TensorPAC actually uses
    pha_freqs_tp_used = pha_freqs_tp[:n_pha_tp]
    amp_freqs_tp_used = amp_freqs_tp[:n_amp_tp]
    
    pha_grid, amp_grid = np.meshgrid(pha_freqs_tp_used, amp_freqs_tp_used, indexing='ij')
    points = np.column_stack([pha_grid.ravel(), amp_grid.ravel()])
    
    # Interpolate and reshape
    pac_gpac_interp = interpolator(points).reshape(pha_grid.shape)  # (n_pha_tp, n_amp_tp)
    
    # Transpose to match TensorPAC's (n_amp, n_pha) shape
    pac_gpac_interp = pac_gpac_interp.T
    
    diff = pac_gpac_interp - pac_tensorpac
    im3 = ax_diff.imshow(diff, aspect='auto', origin='lower',
                         extent=[pha_freqs_tp_used[0], pha_freqs_tp_used[-1], 
                                amp_freqs_tp_used[0], amp_freqs_tp_used[-1]],
                         cmap='RdBu_r', interpolation='bilinear',
                         vmin=-0.2, vmax=0.2)
    ax_diff.set_xlabel('Phase Frequency (Hz)')
    ax_diff.set_ylabel('')
    ax_diff.set_title('Difference (gPAC - TensorPAC)', fontsize=12)
    ax_diff.axvline(6, color='black', linestyle='--', alpha=0.5)
    ax_diff.axhline(80, color='black', linestyle='--', alpha=0.5)
    
    # Add colorbars
    cbar_ax1 = fig.add_subplot(gs[2, :2])
    cbar1 = plt.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Modulation Index', fontsize=10)
    
    cbar_ax2 = fig.add_subplot(gs[2, 2])
    cbar2 = plt.colorbar(im3, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Difference', fontsize=10)
    
    # Add performance comparison text
    speedup = time_tensorpac / time_gpac
    fig.text(0.5, 0.98, f'Performance: gPAC {speedup:.1f}x faster than TensorPAC', 
             ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Add correlation between gPAC and TensorPAC
    corr = np.corrcoef(pac_gpac_interp.flatten(), pac_tensorpac.flatten())[0, 1]
    fig.text(0.5, 0.96, f'Correlation: r = {corr:.3f}', 
             ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    mngs.io.save(fig, save_path)
    print(f"\n[GREEN] Figure saved to: {save_path} [/GREEN]\n")
    
    return fig


def main(args):
    """Main demo function."""
    # Create output directory
    output_dir = './readme_demo_out'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic PAC signal
    print("\n==== Generating synthetic PAC signal ====\n")
    signal, t = generate_synthetic_pac_signal(
        n_seconds=2,
        fs=512,
        f_phase=6,
        f_amp=80,
        pac_strength=0.5,
        noise_level=0.1
    )
    
    # Compute PAC with gPAC
    print("\n==== Computing PAC with gPAC ====\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pac_gpac, pha_freqs_gpac, amp_freqs_gpac, time_gpac = compute_pac_gpac(signal, 512, device)
    print(f"gPAC computation time: {time_gpac:.3f}s")
    print(f"PAC matrix shape: {pac_gpac.shape}")
    print(f"Phase frequencies ({len(pha_freqs_gpac)}): {pha_freqs_gpac}")
    print(f"Amplitude frequencies ({len(amp_freqs_gpac)}): {amp_freqs_gpac}")
    
    # Compute PAC with TensorPAC
    print("\n==== Computing PAC with TensorPAC ====\n")
    pac_tensorpac, pha_freqs_tp, amp_freqs_tp, time_tensorpac = compute_pac_tensorpac(signal, 512)
    print(f"TensorPAC computation time: {time_tensorpac:.3f}s")
    print(f"PAC matrix shape: {pac_tensorpac.shape}")
    print(f"Phase frequencies ({len(pha_freqs_tp)}): {pha_freqs_tp}")
    print(f"Amplitude frequencies ({len(amp_freqs_tp)}): {amp_freqs_tp}")
    
    # Create comparison figure
    print("\n==== Creating comparison figure ====\n")
    fig = create_comparison_figure(
        signal, t, pac_gpac, pac_tensorpac,
        pha_freqs_gpac, amp_freqs_gpac,
        pha_freqs_tp, amp_freqs_tp,
        time_gpac, time_tensorpac,
        os.path.join(output_dir, 'comparison_figure.png')
    )
    
    # Performance summary
    print("\n==== Performance Summary ====\n")
    print(f"gPAC time: {time_gpac:.3f}s")
    print(f"TensorPAC time: {time_tensorpac:.3f}s")
    print(f"Speedup: {time_tensorpac/time_gpac:.1f}x")
    print(f"Device: {device}")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='README demo comparing gPAC vs TensorPAC')
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