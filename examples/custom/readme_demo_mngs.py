#!/usr/bin/env python3
"""
Enhanced demo script for gPAC package using mngs framework.
Showcases PAC analysis with synthetic data and generates visualizations.

This demo creates:
- Synthetic data with known PAC coupling
- PAC calculations using both gPAC and TensorPAC  
- Animated GIF visualization comparing results
- Performance benchmarks with speed comparison
- Ground truth PAC target range indication
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time
import mngs

# Import gPAC
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
    """Generate synthetic signal with known PAC coupling using gPAC's generator."""
    # Use gPAC's synthetic data generator
    generator = SyntheticDataGenerator(fs=fs, duration_sec=n_seconds)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    return signal


def compute_pac_gpac(signal, fs, low_freq_range, high_freq_range, low_freq_width=2, high_freq_width=20):
    """Compute PAC using gPAC."""
    start_time = time.time()
    
    # Create gPAC instance
    pac = PAC(
        low_freq_range=low_freq_range,
        high_freq_range=high_freq_range,
        low_freq_width=low_freq_width,
        high_freq_width=high_freq_width,
        fs=fs,
        n_jobs=1
    )
    
    # Convert to torch tensor and compute PAC
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.from_numpy(signal).float()
    else:
        signal_tensor = signal
        
    if signal_tensor.dim() == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif signal_tensor.dim() == 2:
        signal_tensor = signal_tensor.unsqueeze(0)  # Add batch dim
    
    # Compute PAC
    pac_values = pac(signal_tensor)
    
    end_time = time.time()
    compute_time = end_time - start_time
    
    # Convert back to numpy
    pac_values = pac_values.cpu().numpy().squeeze()
    
    return pac_values, compute_time


def compute_pac_tensorpac(signal, fs, low_freq_range, high_freq_range):
    """Compute PAC using TensorPAC."""
    if not TENSORPAC_AVAILABLE:
        return None, 0
        
    start_time = time.time()
    
    # Create TensorPAC instance  
    pac = TensorPAC(idpac=(2, 0, 0), f_pha=low_freq_range, f_amp=high_freq_range, dcomplex='hilbert')
    
    # Prepare signal
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        
    # Compute PAC
    pac_values = pac.filterfit(fs, signal)
    pac_values = pac_values.squeeze()
    
    end_time = time.time()
    compute_time = end_time - start_time
    
    return pac_values, compute_time


@mngs.plt.config(dpi=150)
def create_comparison_animation(signals, pac_results_gpac, pac_results_tensorpac, 
                              low_freq_range, high_freq_range, phase_freq, amp_freq,
                              time_gpac, time_tensorpac, save_path):
    """Create animated comparison between gPAC and TensorPAC results."""
    n_timepoints = len(pac_results_gpac)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)
    
    # Axes for signal
    ax_signal = fig.add_subplot(gs[0, :])
    
    # Axes for PAC matrices
    ax_gpac = fig.add_subplot(gs[1, 0])
    ax_tensorpac = fig.add_subplot(gs[1, 1])
    
    # Axes for PAC time series
    ax_timeseries = fig.add_subplot(gs[2, :])
    
    # Initialize plots
    frames = []
    vmin, vmax = 0, 0.8
    
    for i in range(0, n_timepoints, 5):  # Sample every 5th frame for smaller file size
        artists = []
        
        # Clear axes
        ax_signal.clear()
        ax_gpac.clear()
        ax_tensorpac.clear()
        ax_timeseries.clear()
        
        # Plot signal with sliding window
        window_size = 500  # 2 seconds at 250 Hz
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signals[0]), i + window_size // 2)
        
        ax_signal.plot(signals[0][start_idx:end_idx], 'b-', alpha=0.7)
        ax_signal.axvline(i - start_idx, color='r', linestyle='--', alpha=0.5)
        ax_signal.set_title(f'Input Signal (Window around t={i/250:.2f}s)', fontsize=14)
        ax_signal.set_xlabel('Samples')
        ax_signal.set_ylabel('Amplitude')
        
        # Plot PAC matrices
        im1 = ax_gpac.imshow(pac_results_gpac[i], aspect='auto', 
                            extent=[high_freq_range[0], high_freq_range[-1], 
                                   low_freq_range[-1], low_freq_range[0]],
                            vmin=vmin, vmax=vmax, cmap='hot')
        ax_gpac.set_title(f'gPAC (t={i/250:.2f}s)\nCompute time: {time_gpac:.3f}s', fontsize=14)
        ax_gpac.set_xlabel('Frequency for Amplitude (Hz)')
        ax_gpac.set_ylabel('Frequency for Phase (Hz)')
        
        # Add ground truth target
        rect = plt.Rectangle((amp_freq-5, phase_freq-0.5), 10, 1, 
                           fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
        ax_gpac.add_patch(rect)
        ax_gpac.text(amp_freq, phase_freq-1.5, 'Target', color='cyan', ha='center', fontsize=10)
        
        if pac_results_tensorpac is not None:
            im2 = ax_tensorpac.imshow(pac_results_tensorpac[i], aspect='auto',
                                    extent=[high_freq_range[0], high_freq_range[-1], 
                                           low_freq_range[-1], low_freq_range[0]],
                                    vmin=vmin, vmax=vmax, cmap='hot')
            ax_tensorpac.set_title(f'TensorPAC (t={i/250:.2f}s)\nCompute time: {time_tensorpac:.3f}s', fontsize=14)
            ax_tensorpac.set_xlabel('Frequency for Amplitude (Hz)')
            ax_tensorpac.set_ylabel('Frequency for Phase (Hz)')
            
            # Add ground truth target
            rect2 = plt.Rectangle((amp_freq-5, phase_freq-0.5), 10, 1, 
                               fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
            ax_tensorpac.add_patch(rect2)
            ax_tensorpac.text(amp_freq, phase_freq-1.5, 'Target', color='cyan', ha='center', fontsize=10)
        else:
            ax_tensorpac.text(0.5, 0.5, 'TensorPAC not available', 
                            transform=ax_tensorpac.transAxes, ha='center', va='center')
        
        # Plot PAC strength over time at target frequencies
        target_phase_idx = np.argmin(np.abs(low_freq_range - phase_freq))
        target_amp_idx = np.argmin(np.abs(high_freq_range - amp_freq))
        
        ax_timeseries.plot(pac_results_gpac[:i+1, target_phase_idx, target_amp_idx], 
                          'b-', label='gPAC', linewidth=2)
        if pac_results_tensorpac is not None:
            ax_timeseries.plot(pac_results_tensorpac[:i+1, target_phase_idx, target_amp_idx], 
                             'r--', label='TensorPAC', linewidth=2)
        
        ax_timeseries.set_xlim(0, n_timepoints)
        ax_timeseries.set_ylim(0, 1)
        ax_timeseries.set_xlabel('Time point')
        ax_timeseries.set_ylabel('PAC Strength')
        ax_timeseries.set_title(f'PAC Strength at Target Frequencies ({phase_freq}Hz - {amp_freq}Hz)', fontsize=14)
        ax_timeseries.legend()
        ax_timeseries.grid(True, alpha=0.3)
        
        # Add colorbar
        if i == 0:
            cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.3])
            plt.colorbar(im1, cax=cbar_ax, label='PAC Strength')
        
        # Capture frame
        mngs.io.save(fig, save_path.replace('.gif', f'_frame_{i:04d}.png'), dpi=100)
        frames.append(Image.open(save_path.replace('.gif', f'_frame_{i:04d}.png')))
    
    # Save as GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:], 
                   duration=100, loop=0, optimize=True)
    
    # Clean up frame files
    import glob
    for frame_file in glob.glob(save_path.replace('.gif', '_frame_*.png')):
        os.remove(frame_file)
    
    plt.close(fig)
    print(f"Animation saved to {save_path}")


def main():
    """Main demo function."""
    # Set up output directory using mngs
    output_dir = mngs.io.get_dirpath(__file__, "outputs")
    mngs.io.makedirs(output_dir)
    
    # Parameters
    n_seconds = 5
    fs = 250
    phase_freq = 6.0
    amp_freq = 60.0
    coupling_strength = 0.7
    noise_level = 0.1
    
    # Generate synthetic data
    print("Generating synthetic PAC signal...")
    signal = generate_synthetic_pac_signal(
        n_seconds=n_seconds,
        fs=fs,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    
    # Define frequency ranges
    low_freq_range = np.arange(2, 20, 1)
    high_freq_range = np.arange(30, 100, 2)
    
    # Compute sliding window PAC
    print("\nComputing PAC using gPAC...")
    window_size = int(1 * fs)  # 1 second window
    step_size = int(0.1 * fs)  # 0.1 second step
    
    pac_results_gpac = []
    total_time_gpac = 0
    
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        pac_values, compute_time = compute_pac_gpac(
            window, fs, low_freq_range, high_freq_range
        )
        pac_results_gpac.append(pac_values)
        total_time_gpac += compute_time
    
    pac_results_gpac = np.array(pac_results_gpac)
    avg_time_gpac = total_time_gpac / len(pac_results_gpac)
    
    # Compute PAC using TensorPAC if available
    if TENSORPAC_AVAILABLE:
        print("Computing PAC using TensorPAC...")
        pac_results_tensorpac = []
        total_time_tensorpac = 0
        
        for start in range(0, len(signal) - window_size, step_size):
            window = signal[start:start + window_size]
            pac_values, compute_time = compute_pac_tensorpac(
                window, fs, low_freq_range, high_freq_range
            )
            pac_results_tensorpac.append(pac_values)
            total_time_tensorpac += compute_time
        
        pac_results_tensorpac = np.array(pac_results_tensorpac)
        avg_time_tensorpac = total_time_tensorpac / len(pac_results_tensorpac)
    else:
        pac_results_tensorpac = None
        avg_time_tensorpac = 0
    
    # Print performance comparison
    print(f"\nPerformance Comparison:")
    print(f"gPAC average time per window: {avg_time_gpac:.4f} seconds")
    if TENSORPAC_AVAILABLE:
        print(f"TensorPAC average time per window: {avg_time_tensorpac:.4f} seconds")
        print(f"Speed-up factor: {avg_time_tensorpac/avg_time_gpac:.2f}x")
    
    # Create animated comparison
    print("\nCreating animated comparison...")
    gif_path = output_dir / "pac_comparison.gif"
    create_comparison_animation(
        [signal], pac_results_gpac, pac_results_tensorpac,
        low_freq_range, high_freq_range, phase_freq, amp_freq,
        avg_time_gpac, avg_time_tensorpac, str(gif_path)
    )
    
    # Save results using mngs
    results = {
        'pac_results_gpac': pac_results_gpac,
        'pac_results_tensorpac': pac_results_tensorpac,
        'avg_time_gpac': avg_time_gpac,
        'avg_time_tensorpac': avg_time_tensorpac,
        'parameters': {
            'n_seconds': n_seconds,
            'fs': fs,
            'phase_freq': phase_freq,
            'amp_freq': amp_freq,
            'coupling_strength': coupling_strength,
            'noise_level': noise_level
        }
    }
    mngs.io.save(results, output_dir / "pac_results.pkl")
    
    print(f"\nResults saved to {output_dir}")
    print("Demo completed!")


if __name__ == "__main__":
    main()