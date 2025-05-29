#!/usr/bin/env python3
"""
Example: Phase-Amplitude Coupling (PAC) Analysis with gPAC

This example demonstrates:
- Basic PAC computation using gPAC
- Synthetic data generation with known PAC
- Visualization of PAC comodulogram
- Performance comparison with TensorPAC (if available)

All outputs are saved using mngs framework conventions.
"""

import numpy as np
import torch
import mngs
from gpac import PAC, SyntheticDataGenerator

# Optional: Import TensorPAC for comparison
try:
    from tensorpac import Pac as TensorPAC
    HAS_TENSORPAC = True
except ImportError:
    HAS_TENSORPAC = False
    print("TensorPAC not available for comparison")


def generate_pac_signal(fs=250, duration=5.0, phase_freq=6.0, amp_freq=60.0, 
                       coupling_strength=0.7, noise_level=0.1):
    """Generate synthetic signal with known PAC."""
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    signal = generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    return signal


def compute_pac(signal, fs, phase_freqs, amp_freqs):
    """Compute PAC using gPAC."""
    # Initialize PAC calculator
    pac = PAC(
        low_freq_range=phase_freqs,
        high_freq_range=amp_freqs, 
        low_freq_width=2.0,
        high_freq_width=20.0,
        fs=fs,
        n_jobs=1
    )
    
    # Convert to tensor if needed
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float()
    
    # Add batch and channel dimensions if needed
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(0)
    
    # Compute PAC
    with mngs.gen.tqdm_disable():
        pac_values = pac(signal)
    
    return pac_values.squeeze().cpu().numpy()


@mngs.io.decorator.cache(is_returns_from_cache=True)
def compute_pac_tensorpac(signal, fs, phase_freqs, amp_freqs):
    """Compute PAC using TensorPAC for comparison."""
    if not HAS_TENSORPAC:
        return None
        
    pac = TensorPAC(idpac=(2, 0, 0), f_pha=phase_freqs, f_amp=amp_freqs)
    
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        
    pac_values = pac.filterfit(fs, signal)
    return pac_values.squeeze()


@mngs.plt.subplots(
    nrows=1, 
    ncols=2 if HAS_TENSORPAC else 1,
    figsize=(12 if HAS_TENSORPAC else 6, 5),
    facecolor="white"
)
def visualize_pac_results(fig, signal, pac_gpac, pac_tensorpac, 
                         phase_freqs, amp_freqs, phase_freq, amp_freq):
    """Visualize PAC results."""
    axes = fig.axes
    
    # Plot gPAC results
    ax = axes[0]
    im = ax.imshow(pac_gpac, aspect='auto', origin='lower',
                   extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                   cmap='hot', vmin=0, vmax=0.8)
    ax.set_xlabel('Amplitude Frequency (Hz)')
    ax.set_ylabel('Phase Frequency (Hz)')
    ax.set_title('gPAC Results')
    
    # Mark the true coupling
    ax.plot(amp_freq, phase_freq, 'co', markersize=10, markeredgewidth=2, 
            markeredgecolor='cyan', fillstyle='none')
    ax.text(amp_freq + 5, phase_freq, 'True\nCoupling', color='cyan', fontsize=10)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='PAC Strength')
    
    # Plot TensorPAC results if available
    if HAS_TENSORPAC and pac_tensorpac is not None:
        ax = axes[1]
        im = ax.imshow(pac_tensorpac, aspect='auto', origin='lower',
                       extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                       cmap='hot', vmin=0, vmax=0.8)
        ax.set_xlabel('Amplitude Frequency (Hz)')
        ax.set_ylabel('Phase Frequency (Hz)')
        ax.set_title('TensorPAC Results')
        
        # Mark the true coupling
        ax.plot(amp_freq, phase_freq, 'co', markersize=10, markeredgewidth=2,
                markeredgecolor='cyan', fillstyle='none')
        
        # Add colorbar
        fig.colorbar(im, ax=ax, label='PAC Strength')
    
    return fig


def main():
    """Main example function."""
    # Set random seed for reproducibility
    mngs.gen.fix_seeds(42)
    
    # Parameters
    fs = 250  # Sampling frequency
    duration = 10.0  # Duration in seconds
    phase_freq = 6.0  # Phase frequency (theta)
    amp_freq = 60.0  # Amplitude frequency (gamma)
    coupling_strength = 0.7
    noise_level = 0.1
    
    # Frequency ranges for PAC computation
    phase_freqs = np.arange(2, 20, 1)
    amp_freqs = np.arange(30, 100, 2)
    
    # Generate synthetic signal
    print("Generating synthetic PAC signal...")
    signal = generate_pac_signal(
        fs=fs,
        duration=duration,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )
    
    # Compute PAC using gPAC
    print("Computing PAC with gPAC...")
    start_time = mngs.gen.get_time()
    pac_gpac = compute_pac(signal, fs, phase_freqs, amp_freqs)
    gpac_time = mngs.gen.get_time() - start_time
    print(f"  Time: {gpac_time:.3f} seconds")
    
    # Compute PAC using TensorPAC if available
    pac_tensorpac = None
    if HAS_TENSORPAC:
        print("Computing PAC with TensorPAC...")
        start_time = mngs.gen.get_time()
        pac_tensorpac, from_cache = compute_pac_tensorpac(signal, fs, phase_freqs, amp_freqs)
        tensorpac_time = mngs.gen.get_time() - start_time
        if not from_cache:
            print(f"  Time: {tensorpac_time:.3f} seconds")
            print(f"  Speedup: {tensorpac_time/gpac_time:.2f}x")
        else:
            print("  (loaded from cache)")
    
    # Visualize results
    print("\nVisualizing results...")
    fig = visualize_pac_results(
        signal, pac_gpac, pac_tensorpac,
        phase_freqs, amp_freqs, phase_freq, amp_freq
    )
    
    # Save outputs
    sdir = mngs.io.get_dirpath(__file__, "outputs")
    
    # Save figure
    spath = sdir / "pac_analysis.png"
    mngs.io.save(fig, spath)
    print(f"  Figure saved to: {spath}")
    
    # Save numerical results
    results = {
        'pac_gpac': pac_gpac,
        'pac_tensorpac': pac_tensorpac,
        'parameters': {
            'fs': fs,
            'duration': duration,
            'phase_freq': phase_freq,
            'amp_freq': amp_freq,
            'coupling_strength': coupling_strength,
            'noise_level': noise_level
        },
        'computation_time': {
            'gpac': gpac_time,
            'tensorpac': tensorpac_time if HAS_TENSORPAC else None
        }
    }
    spath = sdir / "pac_results.pkl"
    mngs.io.save(results, spath)
    print(f"  Results saved to: {spath}")
    
    # Find peak PAC value in gPAC results
    peak_idx = np.unravel_index(np.argmax(pac_gpac), pac_gpac.shape)
    peak_phase = phase_freqs[peak_idx[0]]
    peak_amp = amp_freqs[peak_idx[1]]
    peak_value = pac_gpac[peak_idx]
    
    print(f"\nPeak PAC coupling detected:")
    print(f"  Phase frequency: {peak_phase:.1f} Hz (true: {phase_freq} Hz)")
    print(f"  Amplitude frequency: {peak_amp:.1f} Hz (true: {amp_freq} Hz)")
    print(f"  PAC strength: {peak_value:.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()