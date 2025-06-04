#!/usr/bin/env python3
"""
Example: Bandpass Filtering with gPAC

This example demonstrates:
- Using gPAC's bandpass filter for neural signal processing
- Visualizing filter frequency response
- Filtering signals in phase and amplitude frequency bands
- Comparing with SciPy's filtering (optional)

All outputs are saved using mngs framework conventions.
"""

import numpy as np
import torch
import mngs
from gpac import BandPassFilter
from scipy import signal as sp_signal


def generate_test_signal(fs=1000, duration=5.0):
    """Generate a test signal with multiple frequency components."""
    t = np.arange(0, duration, 1/fs)
    
    # Create signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t) +     # 5 Hz (theta)
        0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz (alpha) 
        0.3 * np.sin(2 * np.pi * 40 * t) +  # 40 Hz (gamma)
        0.2 * np.sin(2 * np.pi * 80 * t) +  # 80 Hz (high gamma)
        0.1 * np.random.randn(len(t))       # noise
    )
    
    return signal, t


@mngs.plt.subplots(nrows=2, ncols=2, figsize=(12, 8), facecolor="white")
def visualize_filter_response(fig, filter_module, fs, phase_freqs, amp_freqs):
    """Visualize filter frequency responses."""
    axes = fig.axes
    
    # Get filter coefficients for phase frequencies
    phase_filters = []
    for freq in phase_freqs:
        b, a = filter_module._design_fir_filter(
            freq - filter_module.low_freq_width/2,
            freq + filter_module.low_freq_width/2,
            fs
        )
        phase_filters.append((b, a))
    
    # Get filter coefficients for amplitude frequencies  
    amp_filters = []
    for freq in amp_freqs:
        b, a = filter_module._design_fir_filter(
            freq - filter_module.high_freq_width/2,
            freq + filter_module.high_freq_width/2,
            fs
        )
        amp_filters.append((b, a))
    
    # Plot phase filter responses
    ax = axes[0]
    for i, (freq, (b, a)) in enumerate(zip(phase_freqs, phase_filters)):
        w, h = sp_signal.freqz(b, a, fs=fs, worN=512)
        ax.plot(w, 20 * np.log10(np.abs(h)), label=f'{freq} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Phase Frequency Filters')
    ax.set_xlim(0, 30)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot amplitude filter responses
    ax = axes[1]
    for i, (freq, (b, a)) in enumerate(zip(amp_freqs, amp_filters)):
        w, h = sp_signal.freqz(b, a, fs=fs, worN=512)
        ax.plot(w, 20 * np.log10(np.abs(h)), label=f'{freq} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Amplitude Frequency Filters')
    ax.set_xlim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Example: Show impulse response of one filter
    ax = axes[2]
    b, a = phase_filters[len(phase_filters)//2]  # Middle frequency
    impulse = sp_signal.unit_impulse(100)
    response = sp_signal.lfilter(b, a, impulse)
    ax.plot(response)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Impulse Response ({phase_freqs[len(phase_freqs)//2]} Hz)')
    ax.grid(True, alpha=0.3)
    
    # Hide the fourth subplot
    axes[3].set_visible(False)
    
    return fig


@mngs.plt.subplots(nrows=3, ncols=1, figsize=(12, 10), facecolor="white")
def visualize_filtering_results(fig, t, original_signal, filtered_phase, filtered_amp, 
                               target_phase_freq, target_amp_freq):
    """Visualize filtering results."""
    axes = fig.axes
    
    # Plot original signal
    ax = axes[0]
    ax.plot(t[:1000], original_signal[:1000], 'b-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Original Signal (first second)')
    ax.grid(True, alpha=0.3)
    
    # Plot phase-filtered signal
    ax = axes[1]
    ax.plot(t[:1000], filtered_phase[:1000], 'g-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Phase-Filtered Signal ({target_phase_freq} Hz)')
    ax.grid(True, alpha=0.3)
    
    # Plot amplitude-filtered signal
    ax = axes[2]
    ax.plot(t[:1000], filtered_amp[:1000], 'r-', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Amplitude-Filtered Signal ({target_amp_freq} Hz)')
    ax.grid(True, alpha=0.3)
    
    return fig


def main():
    """Main example function."""
    # Set random seed
    mngs.gen.fix_seeds(42)
    
    # Parameters
    fs = 1000  # Sampling frequency
    duration = 5.0  # Duration in seconds
    
    # Define frequency ranges
    phase_freqs = np.array([4, 6, 8, 10, 12])  # Theta/alpha range
    amp_freqs = np.array([30, 40, 50, 60, 80])  # Gamma range
    
    # Generate test signal
    print("Generating test signal...")
    signal, t = generate_test_signal(fs=fs, duration=duration)
    
    # Initialize BandPassFilter
    print("\nInitializing BandPassFilter...")
    filter_module = BandPassFilter(
        low_freq_range=phase_freqs,
        high_freq_range=amp_freqs,
        low_freq_width=2.0,  # 2 Hz bandwidth for phase
        high_freq_width=20.0,  # 20 Hz bandwidth for amplitude
        fs=fs,
        filter_length=1001
    )
    
    # Convert signal to tensor
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    
    # Apply filtering
    print("Applying bandpass filtering...")
    with torch.no_grad():
        # Get filtered signals for all frequency pairs
        phase_filtered, amp_filtered = filter_module(signal_tensor)
    
    # Extract specific frequency bands for visualization
    target_phase_idx = len(phase_freqs) // 2  # Middle frequency
    target_amp_idx = len(amp_freqs) // 2
    
    filtered_phase_signal = phase_filtered[0, 0, target_phase_idx, 0, :].numpy()
    filtered_amp_signal = amp_filtered[0, 0, 0, target_amp_idx, :].numpy()
    
    # Create output directory
    sdir = mngs.io.get_dirpath(__file__, "outputs")
    
    # Visualize filter responses
    print("\nVisualizing filter responses...")
    fig_response = visualize_filter_response(
        filter_module, fs, phase_freqs[:3], amp_freqs[:3]
    )
    spath = sdir / "filter_frequency_response.png"
    mngs.io.save(fig_response, spath)
    print(f"  Filter response saved to: {spath}")
    
    # Visualize filtering results
    print("\nVisualizing filtering results...")
    fig_results = visualize_filtering_results(
        t, signal, filtered_phase_signal, filtered_amp_signal,
        phase_freqs[target_phase_idx], amp_freqs[target_amp_idx]
    )
    spath = sdir / "filtering_results.png"
    mngs.io.save(fig_results, spath)
    print(f"  Filtering results saved to: {spath}")
    
    # Save filter information
    filter_info = {
        'phase_frequencies': phase_freqs,
        'amplitude_frequencies': amp_freqs,
        'phase_bandwidth': filter_module.low_freq_width,
        'amplitude_bandwidth': filter_module.high_freq_width,
        'sampling_frequency': fs,
        'filter_length': filter_module.filter_length,
        'filter_type': 'FIR (firwin)',
        'window_type': 'hamming'
    }
    spath = sdir / "filter_info.yaml"
    mngs.io.save(filter_info, spath)
    print(f"  Filter info saved to: {spath}")
    
    # Compute and display power in different bands
    print("\nSignal power in different bands:")
    print(f"  Original signal total power: {np.var(signal):.3f}")
    print(f"  Phase-filtered ({phase_freqs[target_phase_idx]} Hz) power: "
          f"{np.var(filtered_phase_signal):.3f}")
    print(f"  Amplitude-filtered ({amp_freqs[target_amp_idx]} Hz) power: "
          f"{np.var(filtered_amp_signal):.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()