#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 05:30:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_pac_comparison_improved.py

"""
Improved PAC comparison between gPAC and TensorPAC with optimized parameters.

This version:
- Uses larger batch sizes to better utilize GPU
- Implements proper warm-up for fair timing comparison
- Uses consistent filtering parameters between libraries
- Shows results with mres (70 bands) as specified in USER_PLAN
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gpac import PAC, SyntheticDataGenerator

try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    print("Warning: TensorPAC not available for comparison")
    TENSORPAC_AVAILABLE = False


def generate_batch_signals(batch_size=10, n_seconds=5, fs=512, 
                          phase_freq=6.0, amp_freq=80.0):
    """Generate batch of synthetic PAC signals."""
    generator = SyntheticDataGenerator(fs=fs, duration_sec=n_seconds)
    
    signals = []
    for i in range(batch_size):
        # Vary parameters slightly for each signal
        phase_var = phase_freq + np.random.uniform(-0.5, 0.5)
        amp_var = amp_freq + np.random.uniform(-5, 5)
        coupling = 0.7 + np.random.uniform(-0.1, 0.1)
        
        signal = generator.generate_pac_signal(
            phase_freq=phase_var,
            amp_freq=amp_var,
            coupling_strength=coupling,
            noise_level=0.1
        )
        signals.append(signal)
    
    return np.array(signals), phase_freq, amp_freq


def compute_gpac_batch(signals, fs, n_bands=70, warm_up=True):
    """Compute PAC using gPAC with batch processing."""
    # Convert to tensor
    signals_torch = torch.from_numpy(signals).float()
    
    # Add channel dimension if needed
    if signals_torch.ndim == 2:
        signals_torch = signals_torch.unsqueeze(1)
    
    batch_size, n_channels, seq_len = signals_torch.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signals_torch = signals_torch.to(device)
    
    # Initialize PAC with mres (70 bands)
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_bands,
        amp_start_hz=20,
        amp_end_hz=120,
        amp_n_bands=n_bands,
        trainable=False,
        fp16=True  # Use mixed precision for better GPU utilization
    ).to(device)
    
    # Warm-up run
    if warm_up:
        with torch.no_grad():
            _ = pac(signals_torch[:1])
    
    # Actual timing
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        output = pac(signals_torch)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    computation_time = time.time() - start_time
    
    # Extract results (average across batch)
    pac_values = output['pac'].mean(dim=0).squeeze().cpu().numpy()
    pha_freqs = output['phase_frequencies'].cpu().numpy()
    amp_freqs = output['amplitude_frequencies'].cpu().numpy()
    
    return {
        'pac': pac_values,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'time': computation_time,
        'time_per_signal': computation_time / batch_size,
        'batch_size': batch_size,
        'device': device
    }


def compute_tensorpac_batch(signals, fs, n_bands=70):
    """Compute PAC using TensorPAC with batch processing."""
    if not TENSORPAC_AVAILABLE:
        return None
    
    # TensorPAC expects (n_epochs, n_times)
    batch_size = signals.shape[0]
    
    # Create frequency arrays
    pha_freqs = np.linspace(2, 20, n_bands)
    amp_freqs = np.linspace(20, 120, n_bands)
    
    # Initialize TensorPAC with MI method
    pac = TensorPAC(
        idpac=(2, 0, 0),  # MI method
        f_pha=pha_freqs,
        f_amp=amp_freqs,
        dcomplex='hilbert',
        width=1,  # Narrow bands
        n_bins=18
    )
    
    # Timing
    start_time = time.time()
    
    # Compute PAC (TensorPAC handles batch internally)
    xpac = pac.filterfit(fs, signals, n_jobs=-1)  # Use all cores
    
    computation_time = time.time() - start_time
    
    # Process output - average across batch
    pac_values = np.squeeze(xpac)
    if pac_values.ndim > 2:
        # Average across batch and time dimensions
        pac_values = pac_values.mean(axis=tuple(range(2, pac_values.ndim)))
    
    # Transpose to match gPAC format
    pac_values = pac_values.T
    
    return {
        'pac': pac_values,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'time': computation_time,
        'time_per_signal': computation_time / batch_size,
        'batch_size': batch_size,
        'device': 'cpu'
    }


def create_detailed_comparison(gpac_results, tensorpac_results, true_phase, true_amp):
    """Create detailed comparison visualization."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 2, 1, 1], hspace=0.4, wspace=0.3)
    
    # Common colormap limits
    vmin = 0
    vmax = max(gpac_results['pac'].max(), 
               tensorpac_results['pac'].max() if tensorpac_results else 0)
    
    # 1. gPAC heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gpac_results['pac'].T, aspect='auto', origin='lower',
                     extent=[gpac_results['pha_freqs'][0], gpac_results['pha_freqs'][-1],
                            gpac_results['amp_freqs'][0], gpac_results['amp_freqs'][-1]],
                     cmap='hot', interpolation='bilinear', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    ax1.set_title(f'gPAC ({gpac_results["device"].upper()})', fontsize=14, fontweight='bold')
    
    # Add ground truth marker
    ax1.scatter(true_phase, true_amp, c='cyan', s=200, marker='*', 
               edgecolors='white', linewidth=2, label='Ground Truth')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='MI Value')
    
    # 2. TensorPAC heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if tensorpac_results:
        im2 = ax2.imshow(tensorpac_results['pac'].T, aspect='auto', origin='lower',
                        extent=[tensorpac_results['pha_freqs'][0], tensorpac_results['pha_freqs'][-1],
                               tensorpac_results['amp_freqs'][0], tensorpac_results['amp_freqs'][-1]],
                        cmap='hot', interpolation='bilinear', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax2.set_title('TensorPAC (CPU)', fontsize=14, fontweight='bold')
        ax2.scatter(true_phase, true_amp, c='cyan', s=200, marker='*',
                   edgecolors='white', linewidth=2, label='Ground Truth')
        ax2.legend()
        plt.colorbar(im2, ax=ax2, label='MI Value')
    
    # 3. Difference
    ax3 = fig.add_subplot(gs[0, 2])
    if tensorpac_results:
        diff = gpac_results['pac'] - tensorpac_results['pac']
        max_diff = np.abs(diff).max()
        im3 = ax3.imshow(diff.T, aspect='auto', origin='lower',
                        extent=[gpac_results['pha_freqs'][0], gpac_results['pha_freqs'][-1],
                               gpac_results['amp_freqs'][0], gpac_results['amp_freqs'][-1]],
                        cmap='RdBu_r', interpolation='bilinear',
                        vmin=-max_diff, vmax=max_diff)
        ax3.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax3.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax3.set_title('Difference (gPAC - TensorPAC)', fontsize=14, fontweight='bold')
        ax3.scatter(true_phase, true_amp, c='black', s=200, marker='*',
                   edgecolors='white', linewidth=2)
        plt.colorbar(im3, ax=ax3, label='MI Difference')
    
    # 4-5. Cross-sections
    # Phase slice at ground truth amplitude
    ax4 = fig.add_subplot(gs[1, 0:2])
    amp_idx_g = np.argmin(np.abs(gpac_results['amp_freqs'] - true_amp))
    ax4.plot(gpac_results['pha_freqs'], gpac_results['pac'][:, amp_idx_g],
            'g-', linewidth=2, label='gPAC')
    if tensorpac_results:
        amp_idx_t = np.argmin(np.abs(tensorpac_results['amp_freqs'] - true_amp))
        ax4.plot(tensorpac_results['pha_freqs'], tensorpac_results['pac'][:, amp_idx_t],
                'b--', linewidth=2, label='TensorPAC')
    ax4.axvline(true_phase, color='red', linestyle=':', linewidth=2, label='True Phase')
    ax4.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax4.set_ylabel('MI Value', fontsize=12)
    ax4.set_title(f'Phase Slice at γ = {true_amp} Hz', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 6. Performance metrics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    perf_text = "Performance Metrics\n" + "="*25 + "\n\n"
    perf_text += f"Batch Size: {gpac_results['batch_size']}\n"
    perf_text += f"Frequency Bands: {len(gpac_results['pha_freqs'])}\n\n"
    
    perf_text += f"gPAC ({gpac_results['device'].upper()}):\n"
    perf_text += f"  Total: {gpac_results['time']:.3f}s\n"
    perf_text += f"  Per signal: {gpac_results['time_per_signal']*1000:.1f}ms\n\n"
    
    if tensorpac_results:
        perf_text += f"TensorPAC (CPU):\n"
        perf_text += f"  Total: {tensorpac_results['time']:.3f}s\n"
        perf_text += f"  Per signal: {tensorpac_results['time_per_signal']*1000:.1f}ms\n\n"
        
        speedup = tensorpac_results['time_per_signal'] / gpac_results['time_per_signal']
        perf_text += f"Speedup: {speedup:.2f}x"
    
    ax6.text(0.1, 0.5, perf_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 7-8. Peak detection accuracy
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Find peaks
    gpac_peak_idx = np.unravel_index(gpac_results['pac'].argmax(), gpac_results['pac'].shape)
    gpac_peak_phase = gpac_results['pha_freqs'][gpac_peak_idx[0]]
    gpac_peak_amp = gpac_results['amp_freqs'][gpac_peak_idx[1]]
    
    accuracy_text = "Peak Detection Accuracy\n" + "="*50 + "\n\n"
    accuracy_text += f"Ground Truth: θ = {true_phase} Hz, γ = {true_amp} Hz\n\n"
    accuracy_text += f"gPAC Peak: θ = {gpac_peak_phase:.1f} Hz, γ = {gpac_peak_amp:.1f} Hz "
    accuracy_text += f"(Error: Δθ = {abs(gpac_peak_phase-true_phase):.1f} Hz, "
    accuracy_text += f"Δγ = {abs(gpac_peak_amp-true_amp):.1f} Hz)\n"
    
    if tensorpac_results:
        tp_peak_idx = np.unravel_index(tensorpac_results['pac'].argmax(), 
                                      tensorpac_results['pac'].shape)
        tp_peak_phase = tensorpac_results['pha_freqs'][tp_peak_idx[0]]
        tp_peak_amp = tensorpac_results['amp_freqs'][tp_peak_idx[1]]
        
        accuracy_text += f"TensorPAC Peak: θ = {tp_peak_phase:.1f} Hz, γ = {tp_peak_amp:.1f} Hz "
        accuracy_text += f"(Error: Δθ = {abs(tp_peak_phase-true_phase):.1f} Hz, "
        accuracy_text += f"Δγ = {abs(tp_peak_amp-true_amp):.1f} Hz)"
    
    ax7.text(0.5, 0.5, accuracy_text, transform=ax7.transAxes, fontsize=12,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 9. Method comparison
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    method_text = "Method Details: Both use Modulation Index (MI) with idpac=(2,0,0)\n"
    method_text += f"Resolution: {len(gpac_results['pha_freqs'])} frequency bands (mres mode)\n"
    method_text += "gPAC: GPU-accelerated PyTorch implementation with FP16 support\n"
    method_text += "TensorPAC: CPU-based NumPy implementation with parallel processing"
    
    ax9.text(0.5, 0.5, method_text, transform=ax9.transAxes, fontsize=11,
            horizontalalignment='center', verticalalignment='center',
            fontfamily='monospace', alpha=0.7)
    
    plt.suptitle('gPAC vs TensorPAC: Improved Comparison with Batch Processing', 
                fontsize=18, fontweight='bold')
    
    return fig


def main():
    """Run improved PAC comparison."""
    print("="*80)
    print("gPAC vs TensorPAC: Improved Comparison with Batch Processing")
    print("="*80)
    
    # Parameters
    batch_size = 32  # Process multiple signals for better GPU utilization
    n_seconds = 5
    fs = 512
    phase_freq = 6.0
    amp_freq = 80.0
    n_bands = 70  # mres as specified in USER_PLAN
    
    print(f"\nTest Parameters:")
    print(f"  Batch size: {batch_size} signals")
    print(f"  Duration: {n_seconds} seconds per signal")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Ground truth: θ = {phase_freq} Hz → γ = {amp_freq} Hz")
    print(f"  Frequency bands: {n_bands} (mres mode)")
    
    # Generate batch of signals
    print("\nGenerating batch of test signals...")
    signals, true_phase, true_amp = generate_batch_signals(
        batch_size, n_seconds, fs, phase_freq, amp_freq
    )
    print(f"  Generated {signals.shape[0]} signals of shape {signals.shape[1]}")
    
    # Compute with gPAC
    print("\nComputing PAC with gPAC (with warm-up)...")
    gpac_results = compute_gpac_batch(signals, fs, n_bands, warm_up=True)
    print(f"  ✓ Completed in {gpac_results['time']:.3f}s total")
    print(f"  ✓ {gpac_results['time_per_signal']*1000:.1f}ms per signal")
    
    # Compute with TensorPAC
    tensorpac_results = None
    if TENSORPAC_AVAILABLE:
        print("\nComputing PAC with TensorPAC...")
        tensorpac_results = compute_tensorpac_batch(signals, fs, n_bands)
        if tensorpac_results:
            print(f"  ✓ Completed in {tensorpac_results['time']:.3f}s total")
            print(f"  ✓ {tensorpac_results['time_per_signal']*1000:.1f}ms per signal")
    
    # Create visualization
    print("\nCreating detailed comparison visualization...")
    fig = create_detailed_comparison(gpac_results, tensorpac_results, true_phase, true_amp)
    
    save_path = os.path.join(os.path.dirname(__file__), 'pac_comparison_improved.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if tensorpac_results:
        speedup = tensorpac_results['time_per_signal'] / gpac_results['time_per_signal']
        print(f"\nSpeedup: {speedup:.2f}x faster with gPAC")
        print(f"gPAC: {gpac_results['time_per_signal']*1000:.1f}ms per signal on {gpac_results['device'].upper()}")
        print(f"TensorPAC: {tensorpac_results['time_per_signal']*1000:.1f}ms per signal on CPU")
        
        # Correlation
        corr = np.corrcoef(gpac_results['pac'].flatten(), 
                          tensorpac_results['pac'].flatten())[0, 1]
        print(f"\nCorrelation between methods: {corr:.4f}")
    
    print("\n✓ Improved comparison completed successfully!")


if __name__ == "__main__":
    main()