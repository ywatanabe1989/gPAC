#!/usr/bin/env python3
"""Test script to demonstrate improved edge handling in gPAC."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from gpac import PAC, generate_pac_signal

def test_edge_modes():
    """Test different edge handling modes."""
    
    # Generate synthetic PAC signal
    duration = 2
    sf = 1000
    signal = generate_pac_signal(
        duration=duration,
        fs=sf,
        phase_freq=10,
        amp_freq=60,
        coupling_strength=0.5,
        noise_level=0.1
    )
    
    # Convert to torch tensor and create time vector
    signal = torch.tensor(signal, dtype=torch.float32)
    time = torch.linspace(0, duration, len(signal))
    
    # Test different edge modes
    edge_modes = ["auto", "none", "adaptive", "fixed"]
    edge_lengths = [None, None, None, 0.05]  # 5% for fixed mode
    
    fig, axes = plt.subplots(len(edge_modes), 2, figsize=(12, 10))
    fig.suptitle("Edge Handling Comparison in gPAC", fontsize=14)
    
    results = {}
    
    for idx, (mode, length) in enumerate(zip(edge_modes, edge_lengths)):
        print(f"\nTesting edge_mode='{mode}'")
        
        # Create PAC analyzer with specific edge mode
        pac = PAC(
            seq_len=signal.shape[-1],
            fs=sf,
            pha_start_hz=8,
            pha_end_hz=12,
            pha_n_bands=1,
            amp_start_hz=50,
            amp_end_hz=70,
            amp_n_bands=1,
            edge_mode=mode,
            edge_length=length
        )
        
        # Get filtered signals for visualization
        with torch.no_grad():
            # Process signal through filters
            x = signal.unsqueeze(1)  # Add channel dimension
            filtered = pac.bandpass(x, edge_len=0)
            
            # Extract phase and amplitude
            hilbert_out = pac.hilbert(filtered)
            phase_signal = hilbert_out[0, 0, 0, :, 0]  # Phase band
            amp_signal = hilbert_out[0, 0, 1, :, 1]    # Amplitude band amplitude
            
            # Calculate edge length for this mode
            edge_len = pac._calculate_edge_length(signal.shape[-1])
            
            # Apply edge removal
            if edge_len > 0:
                phase_signal = phase_signal[edge_len:-edge_len]
                amp_signal = amp_signal[edge_len:-edge_len]
                time_trimmed = time[edge_len:-edge_len]
            else:
                time_trimmed = time
        
        # Compute PAC
        pac_result = pac(signal)
        pac_value = pac_result['pac'].item()
        
        # Store results
        results[mode] = {
            'pac_value': pac_value,
            'edge_length': edge_len,
            'signal_length': len(phase_signal)
        }
        
        # Plot phase
        ax_phase = axes[idx, 0]
        ax_phase.plot(time, signal[0].cpu().numpy(), alpha=0.3, label='Original')
        ax_phase.plot(time_trimmed.cpu().numpy(), phase_signal.cpu().numpy(), 
                      label=f'Phase (edge={edge_len})')
        ax_phase.set_ylabel(f'{mode.capitalize()} Mode')
        ax_phase.legend()
        ax_phase.grid(True, alpha=0.3)
        
        if idx == 0:
            ax_phase.set_title('Phase Signal')
        
        # Plot amplitude
        ax_amp = axes[idx, 1]
        ax_amp.plot(time_trimmed.cpu().numpy(), amp_signal.cpu().numpy(), 
                   label=f'Amplitude envelope')
        ax_amp.axhline(y=amp_signal.mean().item(), color='r', linestyle='--', 
                       alpha=0.5, label='Mean')
        ax_amp.set_ylabel(f'PAC={pac_value:.3f}')
        ax_amp.legend()
        ax_amp.grid(True, alpha=0.3)
        
        if idx == 0:
            ax_amp.set_title('Amplitude Envelope')
        
        if idx == len(edge_modes) - 1:
            ax_phase.set_xlabel('Time (s)')
            ax_amp.set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Print summary
    print("\n" + "="*60)
    print("Edge Handling Summary:")
    print("="*60)
    for mode, res in results.items():
        print(f"\n{mode.upper()} mode:")
        print(f"  - Edge length: {res['edge_length']} samples")
        print(f"  - Signal length after trimming: {res['signal_length']} samples")
        print(f"  - PAC value: {res['pac_value']:.4f}")
    
    # Save figure
    output_path = "/home/ywatanabe/proj/gPAC/example_outputs/edge_handling_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()

def test_padding_modes():
    """Test different padding modes for filtering."""
    
    # Create a signal with sharp transitions at edges
    sf = 1000
    duration = 0.5
    t = torch.linspace(0, duration, int(sf * duration))
    
    # Create signal with edge discontinuity
    signal = torch.sin(2 * np.pi * 10 * t)
    signal[:50] = 2.0  # Sharp edge at beginning
    signal[-50:] = -2.0  # Sharp edge at end
    
    # Add batch and channel dimensions
    signal = signal.unsqueeze(0).unsqueeze(0)
    
    # Test different padding modes
    padding_modes = ["reflect", "replicate", "circular", "zero"]
    
    fig, axes = plt.subplots(len(padding_modes), 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Padding Mode Effects on Edge Artifacts", fontsize=14)
    
    for idx, padding_mode in enumerate(padding_modes):
        print(f"\nTesting padding_mode='{padding_mode}'")
        
        # Create PAC with specific padding mode
        pac = PAC(
            seq_len=signal.shape[-1],
            fs=sf,
            pha_start_hz=8,
            pha_end_hz=12,
            pha_n_bands=1,
            amp_start_hz=50,
            amp_end_hz=70,
            amp_n_bands=1,
            edge_mode="none"  # No edge removal to see padding effects
        )
        
        # Update padding mode in the filter
        pac.bandpass.filter.padding_mode = padding_mode
        
        # Get filtered signal
        with torch.no_grad():
            filtered = pac.bandpass(signal[:, 0], edge_len=0)
            filtered_signal = filtered[0, 0, 0, :].cpu().numpy()
        
        # Plot
        ax = axes[idx]
        ax.plot(t, signal[0, 0].cpu().numpy(), alpha=0.5, label='Original')
        ax.plot(t, filtered_signal, label=f'{padding_mode.capitalize()} padding')
        ax.set_ylabel(padding_mode.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight edge regions
        ax.axvspan(0, 0.05, alpha=0.1, color='red')
        ax.axvspan(duration-0.05, duration, alpha=0.1, color='red')
        
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # Save figure
    output_path = "/home/ywatanabe/proj/gPAC/example_outputs/padding_modes_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("Testing improved edge handling in gPAC...")
    
    # Ensure output directory exists
    import os
    os.makedirs("/home/ywatanabe/proj/gPAC/example_outputs", exist_ok=True)
    
    # Run tests
    test_edge_modes()
    test_padding_modes()
    
    print("\nAll tests completed!")