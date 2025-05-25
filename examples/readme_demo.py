#!/usr/bin/env python3
"""
gPAC Package Demo: Synthetic Data Generation and PAC Analysis
============================================================

This demo shows how to use the gPAC package to:
1. Generate synthetic PAC signals with different coupling properties
2. Calculate PAC using gPAC
3. Compare with Tensorpac (if available)
4. Visualize results as a GIF-like animation

Features demonstrated:
- SyntheticDataGenerator for creating realistic PAC signals
- calculate_pac function for fast PAC computation
- Visualization of raw signals and PAC results
- Performance comparison between methods
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import gPAC components
import gpac

# Try to import tensorpac for comparison
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
    print("✅ Tensorpac available for comparison")
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("⚠️  Tensorpac not available - skipping comparison")


def create_demo_signal():
    """Create a demo signal with known PAC coupling."""
    # Signal parameters
    fs = 512.0  # Sampling frequency
    duration = 2.0  # Duration in seconds
    
    # Create time vector
    t = np.linspace(0, duration, int(fs * duration))
    
    # PAC parameters - theta-gamma coupling
    pha_freq = 6.0  # Hz (theta)
    amp_freq = 80.0  # Hz (gamma)
    coupling_strength = 0.8
    
    # Generate phase signal (theta oscillation)
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    
    # Generate amplitude modulation based on phase
    modulation = (1 + coupling_strength * np.cos(2 * np.pi * pha_freq * t)) / 2
    
    # Generate carrier signal (gamma oscillation)
    carrier = np.sin(2 * np.pi * amp_freq * t)
    
    # Apply modulation to carrier
    modulated_carrier = modulation * carrier
    
    # Combine signals
    pac_signal = phase_signal + 0.5 * modulated_carrier
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    signal = pac_signal + noise
    
    # Reshape to gPAC format: (batch, channels, segments, time)
    signal_4d = signal.reshape(1, 1, 1, -1)
    
    return signal_4d, fs, t, pha_freq, amp_freq


def calculate_gpac_pac(signal, fs):
    """Calculate PAC using gPAC."""
    print("🔄 Calculating PAC with gPAC...")
    start_time = time.time()
    
    # Calculate PAC with same resolution as tensorpac
    pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=100,  # Match tensorpac "hres"
        amp_start_hz=60.0,
        amp_end_hz=120.0,
        amp_n_bands=70,   # Match tensorpac "mres"
        n_perm=200  # For statistical testing
    )
    
    gpac_time = time.time() - start_time
    print(f"✅ gPAC calculation completed in {gpac_time:.3f} seconds")
    
    return pac_values, pha_freqs, amp_freqs, gpac_time


def calculate_tensorpac_pac(signal, fs):
    """Calculate PAC using Tensorpac (if available)."""
    if not TENSORPAC_AVAILABLE:
        return None, None, None, None
    
    print("🔄 Calculating PAC with Tensorpac...")
    start_time = time.time()
    
    try:
        # Convert signal format for tensorpac (time, trials)
        signal_tp = signal[0, 0, 0, :].reshape(-1, 1)  # (time, trials)
        
        # Create Tensorpac object following mngs reference configuration
        pac_tp = Pac(
            f_pha="hres",  # High resolution: ~100 bands  
            f_amp="mres",  # Medium resolution: ~70 bands
            dcomplex='wavelet'
        )
        
        # Use the correct idpac configuration from mngs reference
        k = 2
        pac_tp.idpac = (k, 0, 0)
        
        # Extract phases and amplitudes separately
        phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
        amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
        
        # Calculate PAC using fit method
        xpac = pac_tp.fit(phases, amplitudes)
        
        # Average across time and transpose to match gPAC format
        pac_values_tp = xpac.mean(axis=-1).T  # (pha, amp) format
        
        tensorpac_time = time.time() - start_time
        print(f"✅ Tensorpac calculation completed in {tensorpac_time:.3f} seconds")
        
        return pac_values_tp, pac_tp.f_pha, pac_tp.f_amp, tensorpac_time
        
    except Exception as e:
        print(f"⚠️  Tensorpac calculation failed: {e}")
        print("📊 Continuing with gPAC results only...")
        return None, None, None, None


def create_visualization_gif(signal, fs, t, pha_freq, amp_freq, 
                           pac_gpac, pha_freqs, amp_freqs, gpac_time,
                           pac_tensorpac=None, pha_freqs_tp=None, amp_freqs_tp=None, 
                           tensorpac_time=None):
    """Create a comprehensive visualization showing all results."""
    
    # Set up the figure with subplots
    if TENSORPAC_AVAILABLE and pac_tensorpac is not None:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    else:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top panel: Raw signal
    ax_signal = fig.add_subplot(gs[0, :])
    signal_1d = signal[0, 0, 0, :]
    ax_signal.plot(t, signal_1d, 'k-', linewidth=1)
    ax_signal.set_title(f'Synthetic PAC Signal (θ={pha_freq}Hz modulating γ={amp_freq}Hz)', 
                       fontsize=14, fontweight='bold')
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.grid(True, alpha=0.3)
    
    # Add text annotation
    ax_signal.text(0.02, 0.95, f'Sampling Rate: {fs} Hz\nDuration: {len(t)/fs:.1f} s', 
                  transform=ax_signal.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # gPAC results
    ax_gpac = fig.add_subplot(gs[1, 0])
    pac_gpac_2d = pac_gpac[0, 0].cpu().numpy()  # Extract from tensor
    im1 = ax_gpac.imshow(pac_gpac_2d, aspect='auto', origin='lower', 
                        extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
                        cmap='viridis')
    ax_gpac.set_title(f'gPAC Results\n({gpac_time:.3f}s)', fontweight='bold')
    ax_gpac.set_xlabel('Amplitude Frequency (Hz)')
    ax_gpac.set_ylabel('Phase Frequency (Hz)')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax_gpac, shrink=0.8)
    cbar1.set_label('PAC Z-score')
    
    # Mark the ground truth coupling
    ax_gpac.plot(amp_freq, pha_freq, 'r*', markersize=15, markeredgecolor='white', 
                markeredgewidth=2, label='Ground Truth')
    ax_gpac.legend()
    
    if TENSORPAC_AVAILABLE and pac_tensorpac is not None:
        # Tensorpac results
        ax_tp = fig.add_subplot(gs[1, 1])
        # Get proper frequency ranges for tensorpac
        if hasattr(pha_freqs_tp, 'mean'):
            pha_range_tp = [pha_freqs_tp.mean(axis=-1).min(), pha_freqs_tp.mean(axis=-1).max()]
            amp_range_tp = [amp_freqs_tp.mean(axis=-1).min(), amp_freqs_tp.mean(axis=-1).max()]
        else:
            pha_range_tp = [pha_freqs_tp.min(), pha_freqs_tp.max()]
            amp_range_tp = [amp_freqs_tp.min(), amp_freqs_tp.max()]
            
        im2 = ax_tp.imshow(pac_tensorpac.squeeze(), aspect='auto', origin='lower',
                          extent=[amp_range_tp[0], amp_range_tp[1], 
                                 pha_range_tp[0], pha_range_tp[1]],
                          cmap='viridis')
        ax_tp.set_title(f'Tensorpac Results\n({tensorpac_time:.3f}s)', fontweight='bold')
        ax_tp.set_xlabel('Amplitude Frequency (Hz)')
        ax_tp.set_ylabel('Phase Frequency (Hz)')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax_tp, shrink=0.8)
        cbar2.set_label('PAC Value')
        
        # Mark the ground truth coupling
        ax_tp.plot(amp_freq, pha_freq, 'r*', markersize=15, markeredgecolor='white', 
                  markeredgewidth=2, label='Ground Truth')
        ax_tp.legend()
        
        # Difference plot
        ax_diff = fig.add_subplot(gs[1, 2])
        
        # Check if shapes match for difference calculation
        pac_tp_2d = pac_tensorpac.squeeze()
        if pac_gpac_2d.shape == pac_tp_2d.shape:
            # Normalize both for comparison
            pac_gpac_norm = (pac_gpac_2d - pac_gpac_2d.min()) / (pac_gpac_2d.max() - pac_gpac_2d.min())
            pac_tp_norm = (pac_tp_2d - pac_tp_2d.min()) / (pac_tp_2d.max() - pac_tp_2d.min())
            
            diff = pac_gpac_norm - pac_tp_norm
            diff_title = 'Difference\n(gPAC - Tensorpac)'
        else:
            # Create a placeholder showing the shape mismatch
            diff = np.zeros_like(pac_gpac_2d)
            diff_title = f'Shape Mismatch\ngPAC: {pac_gpac_2d.shape}\nTensorpac: {pac_tp_2d.shape}'
        
        im3 = ax_diff.imshow(diff, aspect='auto', origin='lower',
                           extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
                           cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_diff.set_title(diff_title, fontweight='bold')
        ax_diff.set_xlabel('Amplitude Frequency (Hz)')
        ax_diff.set_ylabel('Phase Frequency (Hz)')
        
        # Add colorbar
        cbar3 = plt.colorbar(im3, ax=ax_diff, shrink=0.8)
        cbar3.set_label('Normalized Difference')
        
        # Performance comparison
        ax_perf = fig.add_subplot(gs[2, :])
        methods = ['gPAC', 'Tensorpac']
        times = [gpac_time, tensorpac_time]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax_perf.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
        ax_perf.set_ylabel('Computation Time (seconds)')
        ax_perf.set_title('Calculation Speed Comparison', fontweight='bold')
        ax_perf.grid(True, alpha=0.3, axis='y')
        
        # Add time labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Add speedup information
        speedup = tensorpac_time / gpac_time
        ax_perf.text(0.5, 0.8, f'gPAC is {speedup:.1f}x faster', 
                    transform=ax_perf.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=12, fontweight='bold')
    
    else:
        # Only show gPAC results with additional info
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.text(0.5, 0.5, 
                    f'Ground Truth PAC:\n'
                    f'Phase freq: {pha_freq} Hz\n'
                    f'Amplitude freq: {amp_freq} Hz\n\n'
                    f'gPAC Performance:\n'
                    f'Calculation time: {gpac_time:.3f}s\n'
                    f'Tensor shape: {list(pac_gpac.shape)}\n'
                    f'Max PAC value: {pac_gpac.max():.3f}\n'
                    f'At frequencies: ({pha_freqs[pac_gpac[0,0].argmax()//len(amp_freqs)]:.1f}, '
                    f'{amp_freqs[pac_gpac[0,0].argmax()%len(amp_freqs)]:.1f}) Hz',
                    transform=ax_info.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=11)
        ax_info.set_title('Ground Truth & Results Summary', fontweight='bold')
        ax_info.axis('off')
    
    plt.tight_layout()
    return fig


def demonstrate_synthetic_data_generator():
    """Demonstrate the SyntheticDataGenerator capabilities."""
    print("\n" + "="*60)
    print("🎯 DEMONSTRATING SYNTHETIC DATA GENERATION")
    print("="*60)
    
    # Create generator with custom parameters
    generator = gpac.SyntheticDataGenerator(
        fs=512.0,
        duration_sec=2.0,
        n_samples=50,  # Small number for demo
        n_channels=2,
        n_segments=1,
        n_classes=3,  # Use 3 classes for demo
        random_seed=42
    )
    
    print(f"📊 Generator configured with {generator.params['n_classes']} classes:")
    for class_id, class_info in generator.class_definitions.items():
        print(f"  Class {class_id} ({class_info['name']}): "
              f"θ={class_info['pha_range']} Hz, γ={class_info['amp_range']} Hz")
    
    # Generate dataset
    print("\n🔄 Generating synthetic dataset...")
    start_time = time.time()
    datasets = generator.generate_and_split(train_ratio=0.7, val_ratio=0.2)
    generation_time = time.time() - start_time
    
    print(f"✅ Dataset generated in {generation_time:.3f} seconds")
    print(f"📈 Train: {len(datasets['train'])} samples")
    print(f"📈 Validation: {len(datasets['val'])} samples") 
    print(f"📈 Test: {len(datasets['test'])} samples")
    
    # Test the generated data with gPAC
    print("\n🧪 Testing generated data with gPAC...")
    sample_signal, sample_label, sample_metadata = datasets['train'][0]
    
    # Add batch dimension for gPAC
    sample_signal_batch = sample_signal.unsqueeze(0)
    
    # Calculate PAC
    pac_result, pha_freqs, amp_freqs = gpac.calculate_pac(
        sample_signal_batch,
        fs=generator.params['fs'],
        pha_n_bands=15,
        amp_n_bands=15
    )
    
    print(f"✅ PAC calculation successful!")
    print(f"📊 Sample from class {sample_label} ({sample_metadata['class_names']})")
    print(f"📊 Expected coupling: θ={sample_metadata['pha_freqs']:.1f} Hz, "
          f"γ={sample_metadata['amp_freqs']:.1f} Hz")
    print(f"📊 PAC result shape: {pac_result.shape}")
    
    return datasets, generator


def main():
    """Run the complete demo."""
    print("🚀 Starting gPAC Package Demo")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Create demo signal
    print("📡 Creating synthetic PAC signal...")
    signal, fs, t, pha_freq, amp_freq = create_demo_signal()
    print(f"✅ Signal created: {signal.shape} at {fs} Hz")
    print(f"🎯 Ground truth coupling: θ={pha_freq} Hz → γ={amp_freq} Hz")
    
    # 2. Calculate PAC with gPAC
    pac_gpac, pha_freqs, amp_freqs, gpac_time = calculate_gpac_pac(signal, fs)
    
    # 3. Calculate PAC with Tensorpac (if available)
    pac_tensorpac, pha_freqs_tp, amp_freqs_tp, tensorpac_time = calculate_tensorpac_pac(signal, fs)
    
    # 4. Create visualization
    print("\n📊 Creating visualization...")
    fig = create_visualization_gif(
        signal, fs, t, pha_freq, amp_freq,
        pac_gpac, pha_freqs, amp_freqs, gpac_time,
        pac_tensorpac, pha_freqs_tp, amp_freqs_tp, tensorpac_time
    )
    
    # Save the figure
    output_path = Path("readme_demo_output.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved to: {output_path.absolute()}")
    
    # 5. Demonstrate SyntheticDataGenerator
    datasets, generator = demonstrate_synthetic_data_generator()
    
    # 6. Final summary
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 PAC calculated for {signal.shape[0]} signal(s)")
    print(f"⚡ gPAC processing time: {gpac_time:.3f} seconds")
    
    if tensorpac_time:
        if gpac_time > tensorpac_time:
            speedup = gpac_time / tensorpac_time
            print(f"⚡ Tensorpac processing time: {tensorpac_time:.3f} seconds")
            print(f"🚀 Tensorpac is {speedup:.1f}x faster than gPAC")
        else:
            speedup = tensorpac_time / gpac_time
            print(f"⚡ Tensorpac processing time: {tensorpac_time:.3f} seconds")
            print(f"🚀 gPAC is {speedup:.1f}x faster than Tensorpac")
    
    print(f"🎲 Generated {sum(len(ds) for ds in [datasets['train'], datasets['val'], datasets['test']])} "
          f"synthetic samples across {len(generator.class_definitions)} classes")
    print(f"💾 Demo visualization: {output_path.absolute()}")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("🖼️  Run in interactive environment to see plots")
    
    return {
        'signal': signal,
        'pac_gpac': pac_gpac,
        'pac_tensorpac': pac_tensorpac,
        'gpac_time': gpac_time,
        'tensorpac_time': tensorpac_time,
        'datasets': datasets,
        'generator': generator,
        'figure': fig
    }


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    results = main()