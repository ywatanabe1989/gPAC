#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 06:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/comparison_with_tensorpac/example_pac_comparison_timing.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/comparison_with_tensorpac/example_pac_comparison_timing.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# PAC comparison with detailed timing breakdown

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


def time_gpac_detailed(signal, fs, device='cuda'):
    """Time gPAC with separate initialization and computation timing."""
    signal_torch = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    
    # Time initialization
    mngs.str.printc("\n  Timing gPAC initialization...", "cyan")
    torch.cuda.synchronize() if device == 'cuda' else None
    init_start = time.time()
    
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
    
    torch.cuda.synchronize() if device == 'cuda' else None
    init_time = time.time() - init_start
    
    # Move signal to device
    signal_torch = signal_torch.to(device)
    
    # Warm-up run (important for GPU)
    if device == 'cuda':
        mngs.str.printc("  Running GPU warm-up...", "cyan")
        with torch.no_grad():
            _ = pac_gpac(signal_torch)
        torch.cuda.synchronize()
    
    # Time computation
    mngs.str.printc("  Timing gPAC computation...", "cyan")
    torch.cuda.synchronize() if device == 'cuda' else None
    comp_start = time.time()
    
    with torch.no_grad():
        output = pac_gpac(signal_torch)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    comp_time = time.time() - comp_start
    
    # Get results
    pac_matrix = output['pac'].squeeze().cpu().numpy()
    pha_freqs = output['phase_frequencies'].cpu().numpy()
    amp_freqs = output['amplitude_frequencies'].cpu().numpy()
    
    return {
        'init_time': init_time,
        'comp_time': comp_time,
        'total_time': init_time + comp_time,
        'pac_matrix': pac_matrix,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs
    }


def time_tensorpac_detailed(signal, fs, pha_freqs, amp_freqs):
    """Time TensorPAC with separate initialization and computation timing."""
    # Create frequency bands
    pha_bands = [(f-0.5, f+0.5) for f in pha_freqs]
    amp_bands = [(f-2, f+2) for f in amp_freqs]
    
    # Time initialization
    mngs.str.printc("\n  Timing TensorPAC initialization...", "cyan")
    init_start = time.time()
    
    pac_tp = TensorPAC(
        idpac=(2, 0, 0),  # MI method
        f_pha=pha_bands,
        f_amp=amp_bands,
        dcomplex='hilbert',
        n_bins=18
    )
    
    init_time = time.time() - init_start
    
    # Time computation
    mngs.str.printc("  Timing TensorPAC computation...", "cyan")
    comp_start = time.time()
    
    xpac = pac_tp.filterfit(fs, signal.reshape(1, -1), n_jobs=1)
    
    comp_time = time.time() - comp_start
    
    # Process results
    pac_matrix = np.squeeze(xpac)
    if pac_matrix.ndim > 2:
        pac_matrix = pac_matrix.mean(axis=tuple(range(2, pac_matrix.ndim)))
    pac_matrix = pac_matrix.T  # Transpose to match gPAC
    
    return {
        'init_time': init_time,
        'comp_time': comp_time,
        'total_time': init_time + comp_time,
        'pac_matrix': pac_matrix
    }


def main():
    """Run detailed timing comparison."""
    mngs.str.printc("="*80, "blue")
    mngs.str.printc("gPAC vs TensorPAC: Detailed Timing Comparison", "blue")
    mngs.str.printc("="*80, "blue")
    
    # Parameters
    fs = 512
    duration = 5
    phase_freq = 6.0
    amp_freq = 80.0
    
    # Generate signal
    mngs.str.printc("\nGenerating synthetic PAC signal...", "yellow")
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=0.8,
        noise_level=0.1
    )
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mngs.str.printc(f"\nUsing device: {device}", "yellow")
    
    # Time gPAC
    mngs.str.printc("\nTiming gPAC...", "yellow")
    gpac_results = time_gpac_detailed(signal, fs, device)
    
    # Time TensorPAC
    tensorpac_results = None
    if TENSORPAC_AVAILABLE:
        mngs.str.printc("\nTiming TensorPAC...", "yellow")
        tensorpac_results = time_tensorpac_detailed(
            signal, fs, 
            gpac_results['pha_freqs'], 
            gpac_results['amp_freqs']
        )
    
    # Create visualization
    mngs.str.printc("\nCreating timing comparison visualization...", "yellow")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Timing comparison bar chart
    ax = axes[0, 0]
    categories = ['gPAC\nInit', 'gPAC\nComp', 'gPAC\nTotal']
    gpac_times = [
        gpac_results['init_time'],
        gpac_results['comp_time'],
        gpac_results['total_time']
    ]
    
    if tensorpac_results:
        categories.extend(['TP\nInit', 'TP\nComp', 'TP\nTotal'])
        tp_times = [
            tensorpac_results['init_time'],
            tensorpac_results['comp_time'],
            tensorpac_results['total_time']
        ]
        all_times = gpac_times + tp_times
        colors = ['lightblue', 'blue', 'darkblue', 'lightcoral', 'red', 'darkred']
    else:
        all_times = gpac_times
        colors = ['lightblue', 'blue', 'darkblue']
    
    bars = ax.bar(categories, all_times, color=colors)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Timing Breakdown', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, all_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # Speedup comparison
    ax = axes[0, 1]
    if tensorpac_results:
        speedup_comp = tensorpac_results['comp_time'] / gpac_results['comp_time']
        speedup_total = tensorpac_results['total_time'] / gpac_results['total_time']
        
        bars = ax.bar(['Computation\nOnly', 'Including\nInitialization'], 
                      [speedup_comp, speedup_total],
                      color=['green', 'orange'])
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('gPAC Speedup vs TensorPAC', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, [speedup_comp, speedup_total]):
            height = bar.get_height()
            label = f'{val:.2f}x' if val >= 1 else f'{1/val:.2f}x slower'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'TensorPAC not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    
    # gPAC PAC result
    ax = axes[1, 0]
    im = ax.imshow(gpac_results['pac_matrix'].T, aspect='auto', origin='lower',
                   extent=[gpac_results['pha_freqs'][0], gpac_results['pha_freqs'][-1],
                          gpac_results['amp_freqs'][0], gpac_results['amp_freqs'][-1]],
                   cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    ax.set_title(f'gPAC Result ({device.upper()})', fontsize=12, fontweight='bold')
    ax.scatter(phase_freq, amp_freq, c='cyan', s=200, marker='*',
               edgecolors='white', linewidth=2)
    plt.colorbar(im, ax=ax, label='MI')
    
    # TensorPAC PAC result
    ax = axes[1, 1]
    if tensorpac_results:
        im = ax.imshow(tensorpac_results['pac_matrix'].T, aspect='auto', origin='lower',
                       extent=[gpac_results['pha_freqs'][0], gpac_results['pha_freqs'][-1],
                              gpac_results['amp_freqs'][0], gpac_results['amp_freqs'][-1]],
                       cmap='hot', interpolation='bilinear')
        ax.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax.set_title('TensorPAC Result (CPU)', fontsize=12, fontweight='bold')
        ax.scatter(phase_freq, amp_freq, c='cyan', s=200, marker='*',
                   edgecolors='white', linewidth=2)
        plt.colorbar(im, ax=ax, label='MI')
    else:
        ax.text(0.5, 0.5, 'TensorPAC not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    
    plt.suptitle('gPAC vs TensorPAC: Detailed Timing Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    mngs.io.save(fig, "pac_timing_comparison.png", dpi=150)
    mngs.str.printc("\n✓ Saved timing comparison figure", "green")
    
    # Save timing results
    timing_results = {
        "gpac": {
            "device": device,
            "init_time": gpac_results['init_time'],
            "comp_time": gpac_results['comp_time'],
            "total_time": gpac_results['total_time']
        }
    }
    
    if tensorpac_results:
        timing_results["tensorpac"] = {
            "device": "cpu",
            "init_time": tensorpac_results['init_time'],
            "comp_time": tensorpac_results['comp_time'],
            "total_time": tensorpac_results['total_time']
        }
        timing_results["comparison"] = {
            "computation_speedup": speedup_comp,
            "total_speedup": speedup_total
        }
    
    mngs.io.save(timing_results, "pac_timing_results.yaml")
    mngs.str.printc("✓ Saved timing results", "green")
    
    # Print summary
    mngs.str.printc("\n" + "="*80, "magenta")
    mngs.str.printc("TIMING SUMMARY", "magenta")
    mngs.str.printc("="*80, "magenta")
    
    mngs.str.printc(f"\ngPAC ({device.upper()}):", "white")
    mngs.str.printc(f"  Initialization: {gpac_results['init_time']:.4f}s", "white")
    mngs.str.printc(f"  Computation:    {gpac_results['comp_time']:.4f}s", "white")
    mngs.str.printc(f"  Total:          {gpac_results['total_time']:.4f}s", "white")
    
    if tensorpac_results:
        mngs.str.printc(f"\nTensorPAC (CPU):", "white")
        mngs.str.printc(f"  Initialization: {tensorpac_results['init_time']:.4f}s", "white")
        mngs.str.printc(f"  Computation:    {tensorpac_results['comp_time']:.4f}s", "white")
        mngs.str.printc(f"  Total:          {tensorpac_results['total_time']:.4f}s", "white")
        
        mngs.str.printc(f"\nSpeedup Analysis:", "white")
        if speedup_comp >= 1:
            mngs.str.printc(f"  Computation only: {speedup_comp:.2f}x faster with gPAC", "green")
        else:
            mngs.str.printc(f"  Computation only: {1/speedup_comp:.2f}x slower with gPAC", "yellow")
        
        if speedup_total >= 1:
            mngs.str.printc(f"  Including init:   {speedup_total:.2f}x faster with gPAC", "green")
        else:
            mngs.str.printc(f"  Including init:   {1/speedup_total:.2f}x slower with gPAC", "yellow")
    
    mngs.str.printc("\n✓ Timing analysis completed!", "green")


if __name__ == "__main__":
    main()

# EOF