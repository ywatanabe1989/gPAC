#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 12:45:00"
# Author: ywatanabe
# File: example__SyntheticDataGenerator.py

"""
Example demonstrating the SyntheticDataGenerator for creating test signals
with known phase-amplitude coupling properties.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mngs
import torch
from gpac import SyntheticDataGenerator


def main(args):
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Create synthetic data generator
    generator = SyntheticDataGenerator(
        phase_freq=8.0,  # 8 Hz phase frequency (theta)
        amplitude_freq=50.0,  # 50 Hz amplitude frequency (gamma)
        fs=1000.0,  # Sampling rate
        duration=10.0,  # Duration in seconds
        coupling_strength=0.8,  # PAC strength (0-1)
        noise_level=0.1,  # Noise level
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate synthetic signal
    signal = generator.generate()
    
    # Also get individual components for visualization
    phase_signal = generator.generate_phase_signal()
    amplitude_signal = generator.generate_amplitude_signal()
    coupled_signal = generator.generate_coupled_signal()

    # Convert to numpy for plotting
    time = np.linspace(0, generator.duration, int(generator.fs * generator.duration))
    signal_np = signal.cpu().numpy() if signal.is_cuda else signal.numpy()
    phase_np = phase_signal.cpu().numpy() if phase_signal.is_cuda else phase_signal.numpy()
    amplitude_np = amplitude_signal.cpu().numpy() if amplitude_signal.is_cuda else amplitude_signal.numpy()
    coupled_np = coupled_signal.cpu().numpy() if coupled_signal.is_cuda else coupled_signal.numpy()

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot phase signal
    axes[0].plot(time[:1000], phase_np[:1000], color='blue', alpha=0.8)
    axes[0].set_title(f'Phase Signal ({generator.phase_freq} Hz)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot amplitude signal
    axes[1].plot(time[:1000], amplitude_np[:1000], color='green', alpha=0.8)
    axes[1].set_title(f'Amplitude Signal ({generator.amplitude_freq} Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot coupled signal (no noise)
    axes[2].plot(time[:1000], coupled_np[:1000], color='orange', alpha=0.8)
    axes[2].set_title(f'Coupled Signal (strength={generator.coupling_strength})')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # Plot final signal (with noise)
    axes[3].plot(time[:1000], signal_np[:1000], color='red', alpha=0.8)
    axes[3].set_title(f'Final Signal (noise={generator.noise_level})')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = CONFIG.PATH.FIGURES / "synthetic_data_components.png"
    mngs.io.save(fig, fig_path)
    mngs.str.printc(f"Saved figure: {fig_path}", c="green")
    
    # Generate batch of signals for testing
    batch_size = 16
    batch_signals = generator.generate_batch(batch_size)
    mngs.str.printc(f"\nGenerated batch of {batch_size} signals: {batch_signals.shape}", c="cyan")
    
    # Test with different parameters
    mngs.str.printc("\nTesting different coupling strengths:", c="yellow")
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8))
    
    for i, strength in enumerate([0.0, 0.5, 1.0]):
        generator.coupling_strength = strength
        test_signal = generator.generate()
        test_np = test_signal.cpu().numpy() if test_signal.is_cuda else test_signal.numpy()
        
        axes2[i].plot(time[:2000], test_np[:2000], alpha=0.8)
        axes2[i].set_title(f'Coupling Strength = {strength}')
        axes2[i].set_ylabel('Amplitude')
        axes2[i].grid(True, alpha=0.3)
        
    axes2[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    fig2_path = CONFIG.PATH.FIGURES / "coupling_strength_comparison.png"
    mngs.io.save(fig2, fig2_path)
    mngs.str.printc(f"Saved figure: {fig2_path}", c="green")
    
    # Print summary
    print("\nSynthetic Data Generator Summary:")
    print(f"  Phase frequency: {generator.phase_freq} Hz")
    print(f"  Amplitude frequency: {generator.amplitude_freq} Hz")
    print(f"  Sampling rate: {generator.fs} Hz")
    print(f"  Duration: {generator.duration} s")
    print(f"  Device: {generator.device}")
    print(f"  Signal shape: {signal.shape}")

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.finish(
        CONFIG, sys.stdout, sys.stderr, plt, CC, verbose=False
    )
    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", "-shuffle", type=bool, default=False)
    parser.add_argument("--seed", "-seed", type=int, default=42)
    parser.add_argument("--num_workers", "-num_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)