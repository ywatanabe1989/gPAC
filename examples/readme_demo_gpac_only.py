#!/usr/bin/env python3
"""
gPAC demo script showcasing PAC analysis with synthetic data.
Generates animated GIF visualization as specified in USER_PLAN.

This demo creates:
- Synthetic data with known PAC coupling using gPAC generator
- PAC calculations using gPAC with high resolution
- Animated GIF visualization 
- Performance benchmarks
- Ground truth PAC target range indication
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys
import os
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpac import PAC, SyntheticDataGenerator


def main():
    """Main demo function."""
    print("=" * 80)
    print("gPAC Demo: GPU-Accelerated Phase-Amplitude Coupling Analysis")
    print("Creating animated GIF visualization...")
    print("=" * 80)
    
    # Parameters
    n_seconds = 5
    fs = 512  
    phase_freq = 6.0  # Hz (theta) - Ground truth
    amp_freq = 80.0   # Hz (gamma) - Ground truth
    coupling_strength = 0.8
    noise_level = 0.1
    
    # High resolution as specified
    n_pha_bands = 50  # Reduced for faster computation
    n_amp_bands = 50  
    
    print(f"\nGenerating synthetic signal:")
    print(f"  Duration: {n_seconds} seconds")
    print(f"  Sampling rate: {fs} Hz") 
    print(f"  Ground Truth PAC:")
    print(f"    - Phase frequency: {phase_freq} Hz (Theta)")
    print(f"    - Amplitude frequency: {amp_freq} Hz (Gamma)")
    
    # Generate synthetic signal
    generator = SyntheticDataGenerator(fs=fs, duration_sec=n_seconds)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq, 
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    time_vec = np.linspace(0, n_seconds, len(signal))
    
    # Convert to tensor
    signal_torch = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = signal_torch.to(device)
    
    # Initialize PAC calculator
    print(f"\nInitializing PAC calculator on {device}...")
    pac = PAC(
        seq_len=len(signal),
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=n_pha_bands,
        amp_start_hz=20,
        amp_end_hz=120,
        amp_n_bands=n_amp_bands,
        trainable=False
    ).to(device)
    
    # Calculate PAC
    print("Calculating PAC...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        output = pac(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    computation_time = time.time() - start_time
    
    pac_values = output['pac'].squeeze().cpu().numpy()
    pha_freqs = output['phase_frequencies'].cpu().numpy()
    amp_freqs = output['amplitude_frequencies'].cpu().numpy()
    
    print(f"✅ PAC calculation completed in {computation_time:.4f} seconds")
    
    # Create animated GIF
    print("\nGenerating animated GIF...")
    frames = []
    n_frames = 20
    
    for frame_idx in range(n_frames):
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2], width_ratios=[2, 1])
        
        progress = (frame_idx + 1) / n_frames
        
        # Top left: Raw signal
        ax_signal = fig.add_subplot(gs[0, :])
        window_size = int(len(signal) * progress)
        ax_signal.plot(time_vec[:window_size], signal[:window_size], 'b-', linewidth=1, alpha=0.8)
        ax_signal.set_xlabel('Time (s)', fontsize=12)
        ax_signal.set_ylabel('Amplitude', fontsize=12)
        ax_signal.set_title('Raw Synthetic Signal', fontsize=14, fontweight='bold')
        ax_signal.set_xlim(0, n_seconds)
        ax_signal.set_ylim(signal.min() * 1.1, signal.max() * 1.1)
        ax_signal.grid(True, alpha=0.3)
        
        # Add ground truth annotation
        ax_signal.text(0.02, 0.95, f'Ground Truth: θ={phase_freq} Hz → γ={amp_freq} Hz', 
                      transform=ax_signal.transAxes, fontsize=12, 
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Bottom left: PAC matrix
        ax_pac = fig.add_subplot(gs[1, 0])
        pac_mask = np.ones_like(pac_values) * progress
        im = ax_pac.imshow(pac_values * pac_mask, aspect='auto', origin='lower',
                          extent=[pha_freqs[0], pha_freqs[-1], amp_freqs[0], amp_freqs[-1]],
                          cmap='hot', interpolation='bilinear', vmin=0, vmax=pac_values.max())
        ax_pac.set_xlabel('Phase Frequency (Hz)', fontsize=12)
        ax_pac.set_ylabel('Amplitude Frequency (Hz)', fontsize=12)
        ax_pac.set_title('PAC Analysis (gPAC)', fontsize=14, fontweight='bold')
        
        # Mark ground truth
        ax_pac.plot(phase_freq, amp_freq, 'c*', markersize=20, markeredgewidth=2, 
                   markeredgecolor='white', label='Ground Truth')
        ax_pac.legend(loc='upper right')
        
        cbar = plt.colorbar(im, ax=ax_pac)
        cbar.set_label('PAC Value', fontsize=10)
        
        # Bottom right: Performance info
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.axis('off')
        
        info_text = f"""
Performance Metrics:
━━━━━━━━━━━━━━━━━━━━
Device: {device.upper()}
Time: {computation_time:.4f} seconds
━━━━━━━━━━━━━━━━━━━━

Parameters:
━━━━━━━━━━━━━━━━━━━━
Sampling Rate: {fs} Hz
Duration: {n_seconds} seconds
Phase Bands: {n_pha_bands}
Amplitude Bands: {n_amp_bands}
━━━━━━━━━━━━━━━━━━━━

Ground Truth PAC:
━━━━━━━━━━━━━━━━━━━━
θ = {phase_freq} Hz (Theta)
γ = {amp_freq} Hz (Gamma)
Coupling: {coupling_strength}

Progress: {progress*100:.1f}%
"""
        ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes, 
                    fontsize=11, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('gPAC: GPU-Accelerated Phase-Amplitude Coupling', 
                    fontsize=16, fontweight='bold')
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
    
    # Save as GIF
    output_path = os.path.join(os.path.dirname(__file__), 'readme_demo.gif')
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                  duration=150, loop=0)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Animated GIF saved to: {output_path}")
    print(f"\nThe GIF shows:")
    print("  - Top: Raw synthetic signal with progressive visualization")
    print("  - Bottom left: PAC matrix calculated by gPAC")
    print("  - Bottom right: Performance metrics and parameters")
    print("  - Ground truth PAC location marked with cyan star")


if __name__ == "__main__":
    main()