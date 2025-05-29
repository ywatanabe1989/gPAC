#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 06:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/comparison_with_tensorpac/example_pac_comparison_practical_mngs.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/comparison_with_tensorpac/example_pac_comparison_practical_mngs.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Practical PAC comparison with TensorPAC using mngs framework

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gpac import PAC, SyntheticDataGenerator
import mngs

try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False


def main():
    """Run practical PAC comparison with proper mngs output management."""
    mngs.str.printc("="*80, "blue")
    mngs.str.printc("gPAC vs TensorPAC: Practical PAC Comparison", "blue")
    mngs.str.printc("="*80, "blue")
    
    # Parameters
    fs = 512
    duration = 5
    phase_freq = 6.0  # Theta
    amp_freq = 80.0   # Gamma
    
    # Generate signal
    mngs.str.printc("\nGenerating synthetic PAC signal...", "yellow")
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=0.8,
        noise_level=0.1
    )
    time_vec = np.linspace(0, duration, len(signal))
    
    # Prepare for gPAC
    signal_torch = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = signal_torch.to(device)
    
    # Initialize gPAC
    mngs.str.printc("\nInitializing gPAC...", "yellow")
    pac_gpac = PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=20,
        amp_start_hz=30,
        amp_end_hz=120,
        amp_n_bands=20,
        trainable=False
    ).to(device)
    
    # Compute with gPAC
    mngs.str.printc("Computing PAC with gPAC...", "yellow")
    torch.cuda.synchronize() if device == 'cuda' else None
    start_gpac = time.time()
    
    with torch.no_grad():
        output_gpac = pac_gpac(signal_torch)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    time_gpac = time.time() - start_gpac
    
    pac_matrix_gpac = output_gpac['pac'].squeeze().cpu().numpy()
    pha_freqs = output_gpac['phase_frequencies'].cpu().numpy()
    amp_freqs = output_gpac['amplitude_frequencies'].cpu().numpy()
    
    mngs.str.printc(f"  ✓ gPAC completed in {time_gpac:.4f}s on {device}", "green")
    
    # Find peak in gPAC
    peak_idx = np.unravel_index(pac_matrix_gpac.argmax(), pac_matrix_gpac.shape)
    peak_pha_gpac = pha_freqs[peak_idx[0]]
    peak_amp_gpac = amp_freqs[peak_idx[1]]
    
    # Compute with TensorPAC if available
    pac_matrix_tp = None
    time_tp = None
    peak_pha_tp = None
    peak_amp_tp = None
    
    if TENSORPAC_AVAILABLE:
        mngs.str.printc("\nComputing PAC with TensorPAC...", "yellow")
        # Create frequency bands for TensorPAC
        pha_bands = [(f-0.5, f+0.5) for f in pha_freqs]
        amp_bands = [(f-2, f+2) for f in amp_freqs]
        
        pac_tp = TensorPAC(
            idpac=(2, 0, 0),  # MI method
            f_pha=pha_bands,
            f_amp=amp_bands,
            dcomplex='hilbert',
            n_bins=18
        )
        
        start_tp = time.time()
        xpac = pac_tp.filterfit(fs, signal.reshape(1, -1), n_jobs=1)
        time_tp = time.time() - start_tp
        
        pac_matrix_tp = np.squeeze(xpac)
        if pac_matrix_tp.ndim > 2:
            pac_matrix_tp = pac_matrix_tp.mean(axis=tuple(range(2, pac_matrix_tp.ndim)))
        pac_matrix_tp = pac_matrix_tp.T  # Transpose to match gPAC
        
        mngs.str.printc(f"  ✓ TensorPAC completed in {time_tp:.4f}s on CPU", "green")
        
        # Find peak in TensorPAC
        peak_idx_tp = np.unravel_index(pac_matrix_tp.argmax(), pac_matrix_tp.shape)
        peak_pha_tp = pha_freqs[peak_idx_tp[0]]
        peak_amp_tp = amp_freqs[peak_idx_tp[1]]
    
    # Create visualization
    mngs.str.printc("\nCreating comparison visualization...", "yellow")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2], hspace=0.3, wspace=0.3)
    
    # Top: Raw signal
    ax_signal = fig.add_subplot(gs[0, :])
    ax_signal.plot(time_vec[:1024], signal[:1024], 'b-', linewidth=0.8, alpha=0.8)
    ax_signal.set_xlabel('Time (s)', fontsize=12)
    ax_signal.set_ylabel('Amplitude', fontsize=12)
    ax_signal.set_title('Raw Synthetic Signal', fontsize=14, fontweight='bold')
    ax_signal.grid(True, alpha=0.3)
    ax_signal.text(0.02, 0.95, f'Ground Truth PAC: θ={phase_freq} Hz → γ={amp_freq} Hz',
                  transform=ax_signal.transAxes, fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Bottom left: gPAC
    ax_gpac = fig.add_subplot(gs[1, 0])
    im1 = ax_gpac.imshow(pac_matrix_gpac.T, aspect='auto', origin='lower',
                        extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
                        cmap='hot', interpolation='bilinear')
    ax_gpac.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax_gpac.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    ax_gpac.set_title(f'PAC calculated by gPAC\nTime: {time_gpac:.4f}s', 
                     fontsize=12, fontweight='bold')
    ax_gpac.scatter(phase_freq, amp_freq, c='cyan', s=200, marker='*',
                   edgecolors='white', linewidth=2, label='Truth')
    ax_gpac.scatter(peak_pha_gpac, peak_amp_gpac, c='yellow', s=100, marker='o',
                   edgecolors='black', linewidth=2, label='Peak')
    ax_gpac.legend(loc='upper right')
    plt.colorbar(im1, ax=ax_gpac, label='MI')
    
    # Bottom center: TensorPAC
    ax_tp = fig.add_subplot(gs[1, 1])
    if pac_matrix_tp is not None:
        im2 = ax_tp.imshow(pac_matrix_tp.T, aspect='auto', origin='lower',
                          extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
                          cmap='hot', interpolation='bilinear')
        ax_tp.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax_tp.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax_tp.set_title(f'PAC calculated by TensorPAC\nTime: {time_tp:.4f}s',
                       fontsize=12, fontweight='bold')
        ax_tp.scatter(phase_freq, amp_freq, c='cyan', s=200, marker='*',
                     edgecolors='white', linewidth=2, label='Truth')
        ax_tp.scatter(peak_pha_tp, peak_amp_tp, c='yellow', s=100, marker='o',
                     edgecolors='black', linewidth=2, label='Peak')
        ax_tp.legend(loc='upper right')
        plt.colorbar(im2, ax=ax_tp, label='MI')
    else:
        ax_tp.text(0.5, 0.5, 'TensorPAC not available', ha='center', va='center',
                  transform=ax_tp.transAxes, fontsize=14)
        ax_tp.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax_tp.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    
    # Bottom right: Difference
    ax_diff = fig.add_subplot(gs[1, 2])
    if pac_matrix_tp is not None:
        diff = pac_matrix_gpac - pac_matrix_tp
        max_diff = np.abs(diff).max()
        im3 = ax_diff.imshow(diff.T, aspect='auto', origin='lower',
                           extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
                           cmap='RdBu_r', interpolation='bilinear',
                           vmin=-max_diff, vmax=max_diff)
        ax_diff.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax_diff.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax_diff.set_title('Difference\n(gPAC - TensorPAC)', fontsize=12, fontweight='bold')
        ax_diff.scatter(phase_freq, amp_freq, c='black', s=200, marker='*',
                       edgecolors='white', linewidth=2)
        plt.colorbar(im3, ax=ax_diff, label='Difference')
        
        # Add correlation text
        correlation = np.corrcoef(pac_matrix_gpac.flatten(), pac_matrix_tp.flatten())[0, 1]
        ax_diff.text(0.95, 0.05, f'Correlation: {correlation:.3f}',
                    transform=ax_diff.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='bottom', horizontalalignment='right')
    else:
        ax_diff.text(0.5, 0.5, 'Comparison not available', ha='center', va='center',
                    transform=ax_diff.transAxes, fontsize=14)
    
    # Add calculation speed comparison
    speed_text = f"Calculation Speed Comparison:\n"
    speed_text += f"gPAC ({device.upper()}): {time_gpac:.4f} seconds\n"
    if time_tp:
        speed_text += f"TensorPAC (CPU): {time_tp:.4f} seconds\n"
        speedup = time_tp / time_gpac
        speed_text += f"Speedup: {speedup:.2f}x faster with gPAC"
    
    fig.text(0.5, 0.02, speed_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('gPAC vs TensorPAC: PAC Comparison', 
                fontsize=16, fontweight='bold')
    
    # Save using mngs
    mngs.io.save(fig, "pac_comparison_practical.png", dpi=150)
    mngs.str.printc("\n✓ Saved comparison figure", "green")
    
    # Save numerical results
    results = {
        "parameters": {
            "fs": fs,
            "duration": duration,
            "phase_freq": phase_freq,
            "amp_freq": amp_freq
        },
        "gpac": {
            "time": time_gpac,
            "device": device,
            "peak_phase": peak_pha_gpac,
            "peak_amp": peak_amp_gpac,
            "max_mi": pac_matrix_gpac.max()
        }
    }
    
    if time_tp:
        results["tensorpac"] = {
            "time": time_tp,
            "device": "cpu",
            "peak_phase": peak_pha_tp,
            "peak_amp": peak_amp_tp,
            "max_mi": pac_matrix_tp.max(),
            "correlation": correlation
        }
        results["comparison"] = {
            "speedup": speedup,
            "correlation": correlation
        }
    
    mngs.io.save(results, "pac_comparison_results.json")
    mngs.str.printc("✓ Saved numerical results", "green")
    
    # Print summary
    mngs.str.printc("\n" + "="*80, "magenta")
    mngs.str.printc("SUMMARY", "magenta")
    mngs.str.printc("="*80, "magenta")
    mngs.str.printc(f"Ground Truth PAC: θ={phase_freq} Hz → γ={amp_freq} Hz", "white")
    mngs.str.printc(f"gPAC detected: θ={peak_pha_gpac:.1f} Hz → γ={peak_amp_gpac:.1f} Hz", "white")
    if peak_pha_tp is not None:
        mngs.str.printc(f"TensorPAC detected: θ={peak_pha_tp:.1f} Hz → γ={peak_amp_tp:.1f} Hz", "white")
    mngs.str.printc(f"\ngPAC time: {time_gpac:.4f}s on {device}", "white")
    if time_tp:
        mngs.str.printc(f"TensorPAC time: {time_tp:.4f}s on CPU", "white")
        mngs.str.printc(f"Speedup: {speedup:.2f}x", "white")
        mngs.str.printc(f"Correlation: {correlation:.3f}", "white")
    
    mngs.str.printc("\n✓ Comparison completed successfully!", "green")


if __name__ == "__main__":
    main()

# EOF