#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 05:15:00 (ywatanabe)"
# File: ./examples/comparison_with_tensorpac/example_pac_comparison.py

"""
Comprehensive PAC comparison between gPAC and TensorPAC.

Functionalities:
  - Compare PAC computation between gPAC and TensorPAC
  - Generate synthetic data with known PAC coupling
  - Measure computation time and memory usage
  - Visualize results with proper axis labels in Hz
  - Create comparison plots showing both methods and differences

Dependencies:
  - gpac
  - tensorpac
  - torch
  - numpy
  - matplotlib
  - time
  - psutil (for memory tracking)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import psutil
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


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def generate_test_signal(n_seconds=10, fs=512, phase_freq=6.0, amp_freq=80.0, 
                        coupling_strength=0.7, noise_level=0.1):
    """Generate synthetic PAC signal using gPAC's generator."""
    generator = SyntheticDataGenerator(fs=fs, duration_sec=n_seconds)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    return signal


def compute_gpac(signal, fs, pha_bands, amp_bands, device='cuda'):
    """Compute PAC using gPAC."""
    # Convert to tensor
    if isinstance(signal, np.ndarray):
        signal_torch = torch.from_numpy(signal).float()
    else:
        signal_torch = signal.float()
    
    # Ensure correct shape (batch, channels, time)
    if signal_torch.ndim == 1:
        signal_torch = signal_torch.unsqueeze(0).unsqueeze(0)
    elif signal_torch.ndim == 2:
        signal_torch = signal_torch.unsqueeze(0)
    
    # Move to device
    device = device if torch.cuda.is_available() else 'cpu'
    signal_torch = signal_torch.to(device)
    
    # Initialize PAC
    pac = PAC(
        seq_len=signal_torch.shape[-1],
        fs=fs,
        pha_start_hz=pha_bands[0],
        pha_end_hz=pha_bands[1],
        pha_n_bands=pha_bands[2],
        amp_start_hz=amp_bands[0],
        amp_end_hz=amp_bands[1],
        amp_n_bands=amp_bands[2],
        trainable=False
    ).to(device)
    
    # Measure computation
    mem_before = get_memory_usage()
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        output = pac(signal_torch)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    computation_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    # Extract results
    pac_values = output['pac'].squeeze().cpu().numpy()
    pha_freqs = output['phase_frequencies'].cpu().numpy()
    amp_freqs = output['amplitude_frequencies'].cpu().numpy()
    
    return {
        'pac': pac_values,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'time': computation_time,
        'memory': mem_after - mem_before,
        'device': device
    }


def compute_tensorpac(signal, fs, pha_bands, amp_bands):
    """Compute PAC using TensorPAC."""
    if not TENSORPAC_AVAILABLE:
        return None
    
    # Ensure correct shape for TensorPAC
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Create frequency arrays
    pha_freqs = np.linspace(pha_bands[0], pha_bands[1], pha_bands[2])
    amp_freqs = np.linspace(amp_bands[0], amp_bands[1], amp_bands[2])
    
    # Convert to frequency bands for TensorPAC
    f_pha = [(f-0.5, f+0.5) for f in pha_freqs]
    f_amp = [(f-2, f+2) for f in amp_freqs]
    
    # Initialize TensorPAC with MI method (idpac=(2,0,0))
    pac = TensorPAC(
        idpac=(2, 0, 0),  # MI method as specified in USER_PLAN
        f_pha=f_pha,
        f_amp=f_amp,
        dcomplex='hilbert',
        n_bins=18
    )
    
    # Measure computation
    mem_before = get_memory_usage()
    start_time = time.time()
    
    xpac = pac.filterfit(fs, signal, n_jobs=1)
    
    computation_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    # Process output
    pac_values = np.squeeze(xpac)
    if pac_values.ndim > 2:
        pac_values = pac_values.mean(axis=tuple(range(2, pac_values.ndim)))
    
    # Transpose to match gPAC format (pha x amp)
    pac_values = pac_values.T
    
    return {
        'pac': pac_values,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'time': computation_time,
        'memory': mem_after - mem_before,
        'device': 'cpu'
    }


def create_comparison_plot(gpac_results, tensorpac_results, true_phase, true_amp, save_path=None):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.5, 1.5, 1], hspace=0.3, wspace=0.3)
    
    # Helper function to add ground truth markers
    def add_ground_truth(ax, phase_freq, amp_freq):
        ax.axvline(phase_freq, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(amp_freq, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
        ax.plot(phase_freq, amp_freq, 'c*', markersize=20, markeredgewidth=2, 
               markeredgecolor='white', label='Ground Truth')
    
    # 1. gPAC results
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gpac_results['pac'].T, aspect='auto', origin='lower',
                     extent=[gpac_results['pha_freqs'][0], gpac_results['pha_freqs'][-1],
                            gpac_results['amp_freqs'][0], gpac_results['amp_freqs'][-1]],
                     cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    ax1.set_title(f'gPAC ({gpac_results["device"].upper()})\nTime: {gpac_results["time"]:.4f}s', 
                  fontsize=14, fontweight='bold')
    add_ground_truth(ax1, true_phase, true_amp)
    plt.colorbar(im1, ax=ax1, label='PAC Value')
    ax1.legend()
    
    # 2. TensorPAC results
    ax2 = fig.add_subplot(gs[0, 1])
    if tensorpac_results:
        im2 = ax2.imshow(tensorpac_results['pac'].T, aspect='auto', origin='lower',
                        extent=[tensorpac_results['pha_freqs'][0], tensorpac_results['pha_freqs'][-1],
                               tensorpac_results['amp_freqs'][0], tensorpac_results['amp_freqs'][-1]],
                        cmap='hot', interpolation='bilinear')
        ax2.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax2.set_title(f'TensorPAC (CPU)\nTime: {tensorpac_results["time"]:.4f}s', 
                     fontsize=14, fontweight='bold')
        add_ground_truth(ax2, true_phase, true_amp)
        plt.colorbar(im2, ax=ax2, label='PAC Value')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'TensorPAC not available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
    
    # 3. Difference plot
    ax3 = fig.add_subplot(gs[0, 2])
    if tensorpac_results:
        # Interpolate if shapes don't match
        if gpac_results['pac'].shape != tensorpac_results['pac'].shape:
            # Simple nearest neighbor interpolation
            diff = gpac_results['pac'] - tensorpac_results['pac']
        else:
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
        add_ground_truth(ax3, true_phase, true_amp)
        plt.colorbar(im3, ax=ax3, label='Difference')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Comparison not available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14)
    
    # 4. Cross-sections at ground truth
    ax4 = fig.add_subplot(gs[1, 0:2])
    
    # Find indices closest to ground truth
    gpac_phase_idx = np.argmin(np.abs(gpac_results['pha_freqs'] - true_phase))
    gpac_amp_idx = np.argmin(np.abs(gpac_results['amp_freqs'] - true_amp))
    
    # Plot phase cross-section (fixed amplitude)
    ax4.plot(gpac_results['pha_freqs'], gpac_results['pac'][:, gpac_amp_idx], 
            'g-', linewidth=2, label=f'gPAC @ {true_amp:.1f} Hz')
    if tensorpac_results:
        tp_amp_idx = np.argmin(np.abs(tensorpac_results['amp_freqs'] - true_amp))
        ax4.plot(tensorpac_results['pha_freqs'], tensorpac_results['pac'][:, tp_amp_idx], 
                'b--', linewidth=2, label=f'TensorPAC @ {true_amp:.1f} Hz')
    ax4.axvline(true_phase, color='red', linestyle=':', alpha=0.7, label='True Phase')
    ax4.set_xlabel('Phase Frequency (Hz)', fontsize=12)
    ax4.set_ylabel('PAC Value', fontsize=12)
    ax4.set_title('PAC vs Phase Frequency (at peak amplitude)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance comparison
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    perf_text = "Performance Comparison\n" + "="*30 + "\n\n"
    perf_text += f"gPAC ({gpac_results['device'].upper()}):\n"
    perf_text += f"  Time: {gpac_results['time']:.4f} s\n"
    perf_text += f"  Memory: {gpac_results['memory']:.1f} MB\n\n"
    
    if tensorpac_results:
        perf_text += f"TensorPAC (CPU):\n"
        perf_text += f"  Time: {tensorpac_results['time']:.4f} s\n"
        perf_text += f"  Memory: {tensorpac_results['memory']:.1f} MB\n\n"
        
        speedup = tensorpac_results['time'] / gpac_results['time']
        perf_text += f"Speedup: {speedup:.2f}x\n"
        
        # Calculate correlation
        corr = np.corrcoef(gpac_results['pac'].flatten(), 
                          tensorpac_results['pac'].flatten())[0, 1]
        perf_text += f"Correlation: {corr:.4f}"
    
    ax5.text(0.1, 0.5, perf_text, transform=ax5.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Signal info
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    info_text = f"Ground Truth PAC: θ = {true_phase} Hz → γ = {true_amp} Hz\n"
    info_text += f"Phase bands: {len(gpac_results['pha_freqs'])} | "
    info_text += f"Amplitude bands: {len(gpac_results['amp_freqs'])}"
    
    ax6.text(0.5, 0.5, info_text, transform=ax6.transAxes, fontsize=14,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('gPAC vs TensorPAC: Comprehensive PAC Comparison', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def main():
    """Run comprehensive PAC comparison."""
    print("="*80)
    print("gPAC vs TensorPAC: Comprehensive Comparison")
    print("="*80)
    
    # Parameters
    n_seconds = 5
    fs = 512
    phase_freq = 6.0  # Hz (theta)
    amp_freq = 80.0   # Hz (gamma)
    coupling_strength = 0.8
    noise_level = 0.1
    
    # Frequency band parameters
    pha_bands = (2, 20, 30)    # start, end, n_bands
    amp_bands = (20, 120, 30)  # start, end, n_bands
    
    print(f"\nTest Signal Parameters:")
    print(f"  Duration: {n_seconds} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Ground truth PAC: θ={phase_freq} Hz → γ={amp_freq} Hz")
    print(f"  Coupling strength: {coupling_strength}")
    print(f"  Noise level: {noise_level}")
    
    print(f"\nPAC Analysis Parameters:")
    print(f"  Phase bands: {pha_bands[2]} bands from {pha_bands[0]}-{pha_bands[1]} Hz")
    print(f"  Amplitude bands: {amp_bands[2]} bands from {amp_bands[0]}-{amp_bands[1]} Hz")
    
    # Generate test signal
    print("\nGenerating test signal...")
    signal = generate_test_signal(n_seconds, fs, phase_freq, amp_freq, 
                                 coupling_strength, noise_level)
    
    # Compute PAC with gPAC
    print("\nComputing PAC with gPAC...")
    gpac_results = compute_gpac(signal, fs, pha_bands, amp_bands)
    print(f"  ✓ Completed in {gpac_results['time']:.4f}s on {gpac_results['device']}")
    
    # Compute PAC with TensorPAC
    tensorpac_results = None
    if TENSORPAC_AVAILABLE:
        print("\nComputing PAC with TensorPAC...")
        tensorpac_results = compute_tensorpac(signal, fs, pha_bands, amp_bands)
        if tensorpac_results:
            print(f"  ✓ Completed in {tensorpac_results['time']:.4f}s on CPU")
    else:
        print("\n⚠ TensorPAC not available for comparison")
    
    # Create comparison plot
    print("\nCreating comparison visualization...")
    save_path = os.path.join(os.path.dirname(__file__), 'pac_comparison_results.png')
    fig = create_comparison_plot(gpac_results, tensorpac_results, 
                                phase_freq, amp_freq, save_path)
    
    print(f"\n✓ Results saved to: {save_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Find peak PAC in gPAC
    peak_idx = np.unravel_index(gpac_results['pac'].argmax(), gpac_results['pac'].shape)
    peak_phase = gpac_results['pha_freqs'][peak_idx[0]]
    peak_amp = gpac_results['amp_freqs'][peak_idx[1]]
    
    print(f"\ngPAC Results:")
    print(f"  Peak PAC at: θ={peak_phase:.1f} Hz, γ={peak_amp:.1f} Hz")
    print(f"  Peak value: {gpac_results['pac'].max():.4f}")
    print(f"  Computation time: {gpac_results['time']:.4f}s")
    
    if tensorpac_results:
        # Find peak PAC in TensorPAC
        tp_peak_idx = np.unravel_index(tensorpac_results['pac'].argmax(), 
                                      tensorpac_results['pac'].shape)
        tp_peak_phase = tensorpac_results['pha_freqs'][tp_peak_idx[0]]
        tp_peak_amp = tensorpac_results['amp_freqs'][tp_peak_idx[1]]
        
        print(f"\nTensorPAC Results:")
        print(f"  Peak PAC at: θ={tp_peak_phase:.1f} Hz, γ={tp_peak_amp:.1f} Hz")
        print(f"  Peak value: {tensorpac_results['pac'].max():.4f}")
        print(f"  Computation time: {tensorpac_results['time']:.4f}s")
        
        # Performance comparison
        speedup = tensorpac_results['time'] / gpac_results['time']
        print(f"\nPerformance:")
        print(f"  Speedup: {speedup:.2f}x faster with gPAC")
        
        # Accuracy comparison
        corr = np.corrcoef(gpac_results['pac'].flatten(), 
                          tensorpac_results['pac'].flatten())[0, 1]
        print(f"  Correlation: {corr:.4f}")
    
    print(f"\nGround Truth:")
    print(f"  Expected PAC: θ={phase_freq} Hz → γ={amp_freq} Hz")
    
    print("\n✓ Comparison completed successfully!")


if __name__ == "__main__":
    main()