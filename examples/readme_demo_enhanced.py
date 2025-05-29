#!/usr/bin/env python3
"""
Enhanced demo script for gPAC package showcasing PAC analysis with synthetic data.
Generates animated GIF comparison between gPAC and TensorPAC implementations.

This demo creates:
- Synthetic data with known PAC coupling
- PAC calculations using both gPAC and TensorPAC  
- Animated GIF visualization comparing results
- Performance benchmarks with speed comparison
- Ground truth PAC target range indication
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
from PIL import Image
import io

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
                                 coupling_strength=0.7, noise_level=0.1):
    """
    Generate synthetic signal with known PAC coupling using gPAC's generator.
    """
    # Use gPAC's synthetic data generator
    generator = SyntheticDataGenerator(fs=fs, duration_sec=n_seconds)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq, 
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    
    time = np.linspace(0, n_seconds, len(signal))
    return signal, time


def calculate_gpac(signal, fs=250, n_pha_bands=100, n_amp_bands=100):
    """
    Calculate PAC using gPAC with high resolution (hres mode).
    Following reference: p.idpac = (2,0,0) for MI, hres -> n_bands = 100
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal = signal.to(device)
    
    # Initialize PAC calculator with high resolution
    pac = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_pha_bands,
        amp_start_hz=20,
        amp_end_hz=120,
        amp_n_bands=n_amp_bands,
        trainable=False
    ).to(device)
    
    # Time the computation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        output = pac(signal)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    computation_time = time.time() - start_time
    
    pac_values = output['pac'].squeeze().cpu().numpy()
    pha_freqs = output['phase_frequencies'].cpu().numpy()
    amp_freqs = output['amplitude_frequencies'].cpu().numpy()
    
    return pac_values, pha_freqs, amp_freqs, computation_time


def calculate_tensorpac(signal, fs=250, n_pha_bands=100, n_amp_bands=100):
    """
    Calculate PAC using TensorPAC with matching parameters.
    Using idpac=(2,0,0) for MI as specified in reference.
    """
    if not TENSORPAC_AVAILABLE:
        return None, None, None, None
    
    # Ensure correct shape for TensorPAC (n_epochs, n_times)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Create high-resolution frequency vectors
    pha_freqs = np.linspace(2, 20, n_pha_bands)
    amp_freqs = np.linspace(20, 120, n_amp_bands)
    
    # Initialize TensorPAC with MI method using filterfit
    pac = TensorPAC(
        idpac=(2, 0, 0),  # MI method as specified
        f_pha=pha_freqs.reshape(-1, 1),  # Convert to proper shape
        f_amp=amp_freqs.reshape(-1, 1),  # Convert to proper shape
        dcomplex='hilbert',
        n_bins=18
    )
    
    # Time the computation using filterfit method
    start_time = time.time()
    pac_values = pac.filterfit(fs, signal, n_jobs=1)
    computation_time = time.time() - start_time
    
    # pac_values shape is (n_amp, n_pha, n_epochs, n_times)
    # Average over epochs and time, then transpose to match gPAC format
    pac_values = np.squeeze(pac_values)
    if pac_values.ndim == 4:
        pac_values = pac_values.mean(axis=(2, 3))  # Average over epochs and time
    elif pac_values.ndim == 3:
        pac_values = pac_values.mean(axis=2)  # Average over time
    
    # Transpose to (n_pha, n_amp) to match gPAC
    pac_values = pac_values.T
    
    return pac_values, pha_freqs, amp_freqs, computation_time


def create_animated_visualization(signal, time, gpac_results, tensorpac_results, 
                                phase_freq=6.0, amp_freq=60.0, output_path='readme_demo.gif'):
    """
    Create animated GIF visualization as specified in USER_PLAN.
    """
    frames = []
    n_frames = 30  # Number of frames for animation
    
    # Create frames showing progressive PAC calculation
    for frame_idx in range(n_frames):
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.5, 0.3], hspace=0.3)
        
        # Progress indicator
        progress = (frame_idx + 1) / n_frames
        
        # Top: Raw signal with sliding window
        ax_signal = fig.add_subplot(gs[0, :])
        window_size = int(len(signal) * progress)
        ax_signal.plot(time[:window_size], signal[:window_size], 'k-', linewidth=0.8, alpha=0.8)
        ax_signal.set_xlabel('Time (s)', fontsize=12)
        ax_signal.set_ylabel('Amplitude', fontsize=12)
        ax_signal.set_title(f'Raw Synthetic Signal (Phase: {phase_freq} Hz, Amplitude: {amp_freq} Hz)', 
                           fontsize=14, fontweight='bold')
        ax_signal.set_xlim(0, time[-1])
        ax_signal.set_ylim(signal.min() * 1.1, signal.max() * 1.1)
        ax_signal.grid(True, alpha=0.3)
        
        # Add ground truth markers
        ax_signal.text(0.02, 0.95, f'Ground Truth PAC: θ={phase_freq} Hz → γ={amp_freq} Hz', 
                      transform=ax_signal.transAxes, fontsize=12, 
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Bottom panels: PAC results
        gpac_pac, gpac_pha, gpac_amp, gpac_time = gpac_results
        
        # Mask PAC values based on progress
        pac_mask = np.ones_like(gpac_pac) * progress
        
        # Bottom left: gPAC
        ax_gpac = fig.add_subplot(gs[1, 0])
        im1 = ax_gpac.imshow(gpac_pac * pac_mask, aspect='auto', origin='lower',
                            extent=[gpac_pha[0], gpac_pha[-1], gpac_amp[0], gpac_amp[-1]],
                            cmap='hot', interpolation='bilinear', vmin=0, vmax=gpac_pac.max())
        ax_gpac.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax_gpac.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax_gpac.set_title(f'gPAC (GPU Accelerated)\nComputation Time: {gpac_time:.4f}s', 
                         fontsize=12, fontweight='bold', color='green')
        
        # Mark ground truth
        ax_gpac.plot(phase_freq, amp_freq, 'c*', markersize=15, markeredgewidth=2, 
                    markeredgecolor='white', label='Ground Truth')
        ax_gpac.legend(loc='upper right')
        
        cbar1 = plt.colorbar(im1, ax=ax_gpac)
        cbar1.set_label('PAC Value', fontsize=10)
        
        # Bottom center: TensorPAC
        ax_tensorpac = fig.add_subplot(gs[1, 1])
        if TENSORPAC_AVAILABLE and tensorpac_results[0] is not None:
            tensorpac_pac, tensorpac_pha, tensorpac_amp, tensorpac_time = tensorpac_results
            im2 = ax_tensorpac.imshow(tensorpac_pac * pac_mask, aspect='auto', origin='lower',
                                     extent=[tensorpac_pha[0], tensorpac_pha[-1], 
                                            tensorpac_amp[0], tensorpac_amp[-1]],
                                     cmap='hot', interpolation='bilinear', 
                                     vmin=0, vmax=tensorpac_pac.max())
            ax_tensorpac.set_xlabel('Phase Frequency (Hz)', fontsize=12)
            ax_tensorpac.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
            ax_tensorpac.set_title(f'TensorPAC (CPU)\nComputation Time: {tensorpac_time:.4f}s', 
                                 fontsize=12, fontweight='bold', color='blue')
            
            # Mark ground truth
            ax_tensorpac.plot(phase_freq, amp_freq, 'c*', markersize=15, markeredgewidth=2,
                            markeredgecolor='white', label='Ground Truth')
            ax_tensorpac.legend(loc='upper right')
            
            cbar2 = plt.colorbar(im2, ax=ax_tensorpac)
            cbar2.set_label('PAC Value', fontsize=10)
        else:
            ax_tensorpac.text(0.5, 0.5, 'TensorPAC not available', 
                            ha='center', va='center', transform=ax_tensorpac.transAxes, fontsize=14)
            ax_tensorpac.set_xlabel('Phase Frequency (Hz)', fontsize=12)
            ax_tensorpac.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
            ax_tensorpac.set_title('TensorPAC', fontsize=12, fontweight='bold')
        
        # Bottom right: Difference
        ax_diff = fig.add_subplot(gs[1, 2])
        if TENSORPAC_AVAILABLE and tensorpac_results[0] is not None:
            diff = (gpac_pac - tensorpac_pac) * pac_mask
            max_diff = np.abs(diff).max()
            im3 = ax_diff.imshow(diff, aspect='auto', origin='lower',
                               extent=[gpac_pha[0], gpac_pha[-1], gpac_amp[0], gpac_amp[-1]],
                               cmap='RdBu_r', interpolation='bilinear',
                               vmin=-max_diff, vmax=max_diff)
            ax_diff.set_xlabel('Phase Frequency (Hz)', fontsize=12)
            ax_diff.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
            ax_diff.set_title('Difference (gPAC - TensorPAC)', fontsize=12, fontweight='bold')
            
            # Mark ground truth
            ax_diff.plot(phase_freq, amp_freq, 'k*', markersize=15, markeredgewidth=2,
                       markeredgecolor='white', label='Ground Truth')
            ax_diff.legend(loc='upper right')
            
            cbar3 = plt.colorbar(im3, ax=ax_diff)
            cbar3.set_label('Difference', fontsize=10)
        else:
            ax_diff.text(0.5, 0.5, 'Comparison not available', 
                       ha='center', va='center', transform=ax_diff.transAxes, fontsize=14)
            ax_diff.set_xlabel('Phase Frequency (Hz)', fontsize=12)
            ax_diff.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
            ax_diff.set_title('Difference', fontsize=12, fontweight='bold')
        
        # Bottom: Performance comparison text
        ax_perf = fig.add_subplot(gs[2, :])
        ax_perf.axis('off')
        
        perf_text = f"Performance Comparison (Progress: {progress*100:.1f}%)\n"
        perf_text += f"gPAC Time: {gpac_time:.4f}s | "
        
        if TENSORPAC_AVAILABLE and tensorpac_time is not None:
            speedup = tensorpac_time / gpac_time
            perf_text += f"TensorPAC Time: {tensorpac_time:.4f}s | "
            perf_text += f"Speedup: {speedup:.2f}x faster"
        else:
            perf_text += "TensorPAC: Not available"
        
        ax_perf.text(0.5, 0.5, perf_text, ha='center', va='center', 
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('gPAC: GPU-Accelerated Phase-Amplitude Coupling Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Convert to image for GIF
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
    
    # Save as GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                  duration=200, loop=0)  # 200ms per frame
    
    return output_path


def main():
    """Main demo function."""
    print("=" * 80)
    print("gPAC Demo: GPU-Accelerated Phase-Amplitude Coupling Analysis")
    print("Creating animated GIF visualization...")
    print("=" * 80)
    
    # Parameters matching USER_PLAN requirements
    n_seconds = 5
    fs = 512  # Higher sampling rate for better resolution
    phase_freq = 6.0  # Hz (theta band) - Ground truth
    amp_freq = 80.0   # Hz (gamma band) - Ground truth
    coupling_strength = 0.8
    noise_level = 0.1
    
    # High resolution as specified in references
    n_pha_bands = 100  # hres mode
    n_amp_bands = 100  # hres mode
    
    print(f"\nGenerating synthetic signal with gPAC generator:")
    print(f"  Duration: {n_seconds} seconds")
    print(f"  Sampling rate: {fs} Hz") 
    print(f"  Ground Truth PAC:")
    print(f"    - Phase frequency: {phase_freq} Hz (Theta)")
    print(f"    - Amplitude frequency: {amp_freq} Hz (Gamma)")
    print(f"  Coupling strength: {coupling_strength}")
    print(f"  Noise level: {noise_level}")
    
    # Generate synthetic signal using gPAC
    signal, time = generate_synthetic_pac_signal(
        n_seconds=n_seconds,
        fs=fs,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    
    print(f"\nCalculating PAC with high resolution:")
    print(f"  Phase bands: {n_pha_bands} (2-20 Hz)")
    print(f"  Amplitude bands: {n_amp_bands} (20-120 Hz)")
    print(f"  Method: Modulation Index (MI)")
    
    # Calculate PAC using gPAC
    print("\n  Computing with gPAC (GPU)...", end='', flush=True)
    gpac_results = calculate_gpac(signal, fs, n_pha_bands, n_amp_bands)
    print(f" Done! (Time: {gpac_results[3]:.4f}s)")
    
    # Calculate PAC using TensorPAC
    if TENSORPAC_AVAILABLE:
        print("  Computing with TensorPAC (CPU)...", end='', flush=True)
        tensorpac_results = calculate_tensorpac(signal, fs, n_pha_bands, n_amp_bands)
        if tensorpac_results[0] is not None:
            print(f" Done! (Time: {tensorpac_results[3]:.4f}s)")
        else:
            print(" Failed!")
            tensorpac_results = (None, None, None, None)
    else:
        tensorpac_results = (None, None, None, None)
    
    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(f"gPAC computation time: {gpac_results[3]:.4f} seconds")
    if TENSORPAC_AVAILABLE and tensorpac_results[3] is not None:
        print(f"TensorPAC computation time: {tensorpac_results[3]:.4f} seconds")
        speedup = tensorpac_results[3] / gpac_results[3]
        print(f"SPEEDUP FACTOR: {speedup:.2f}x faster with gPAC!")
    else:
        print("TensorPAC: Not available for comparison")
    
    print(f"\nGround Truth PAC Target:")
    print(f"  Phase: {phase_freq} Hz")
    print(f"  Amplitude: {amp_freq} Hz")
    
    # Create animated visualization
    print("\nGenerating animated GIF visualization...")
    output_path = os.path.join(os.path.dirname(__file__), 'readme_demo.gif')
    gif_path = create_animated_visualization(
        signal, time, gpac_results, tensorpac_results, 
        phase_freq, amp_freq, output_path
    )
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Animated GIF saved to: {gif_path}")
    print(f"\nThe GIF shows:")
    print("  - Top: Raw synthetic signal with known PAC")
    print("  - Bottom left: PAC calculated by gPAC (GPU)")
    print("  - Bottom center: PAC calculated by TensorPAC (CPU)")
    print("  - Bottom right: Difference (gPAC - TensorPAC)")
    print("  - Performance metrics and speedup factor")
    print("  - Ground truth PAC location marked with stars")


if __name__ == "__main__":
    main()