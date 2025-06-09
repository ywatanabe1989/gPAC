#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simplified SyntheticDataGenerator example

"""Imports"""
import argparse
import sys
import numpy as np
import torch

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Main function"""
def main(args):
    """Run SyntheticDataGenerator demonstration."""
    import mngs
    from gpac import generate_pac_signal, SyntheticDataGenerator, PAC
    
    mngs.str.printc("🚀 SyntheticDataGenerator Demonstration", c="green")
    mngs.str.printc("=" * 60, c="green")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mngs.str.printc(f"Using device: {device}", c="cyan")
    
    # Part 1: Simple signal generation with generate_pac_signal
    mngs.str.printc("\n🎯 Part 1: Simple PAC signal generation", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    # Generate a simple PAC signal
    fs = 256
    duration = 2.0
    phase_freq = 6  # Theta
    amp_freq = 80   # Gamma
    
    signal = generate_pac_signal(
        n_samples=int(fs * duration),
        fs=fs,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=0.8,
        noise_level=0.1
    )
    
    mngs.str.printc(f"\nGenerated signal: θ={phase_freq} Hz → γ={amp_freq} Hz", c="cyan")
    mngs.str.printc(f"Signal shape: {signal.shape}", c="cyan")
    mngs.str.printc(f"Signal stats: mean={signal.mean():.4f}, std={signal.std():.4f}", c="cyan")
    
    # Part 2: Using SyntheticDataGenerator
    mngs.str.printc("\n🎯 Part 2: Using SyntheticDataGenerator", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        fs=512.0,
        duration_sec=1.0,
        random_seed=42
    )
    
    mngs.str.printc(f"\nGenerator parameters:", c="cyan")
    mngs.str.printc(f"  Sampling rate: {generator.fs} Hz", c="cyan")
    mngs.str.printc(f"  Duration: {generator.duration_sec} s", c="cyan")
    mngs.str.printc(f"  Samples per signal: {generator.n_samples}", c="cyan")
    
    # Generate signals with different coupling properties
    mngs.str.printc("\n🔄 Generating signals with different coupling...", c="blue")
    
    # No coupling
    signal_no_coupling = generator.generate_pac_signal(
        phase_freq=6.0,
        amp_freq=80.0,
        coupling_strength=0.0,
        noise_level=0.1
    )
    
    # Weak coupling
    signal_weak = generator.generate_pac_signal(
        phase_freq=6.0,
        amp_freq=80.0,
        coupling_strength=0.3,
        noise_level=0.1
    )
    
    # Strong coupling
    signal_strong = generator.generate_pac_signal(
        phase_freq=6.0,
        amp_freq=80.0,
        coupling_strength=0.8,
        noise_level=0.1
    )
    
    # Analyze PAC in generated signals
    mngs.str.printc("\n🔍 Analyzing PAC in generated signals...", c="blue")
    
    # Initialize PAC analyzer
    pac_analyzer = PAC(
        seq_len=generator.n_samples,
        fs=generator.fs,
        pha_start_hz=4,
        pha_end_hz=10,
        pha_n_bands=3,
        amp_start_hz=60,
        amp_end_hz=100,
        amp_n_bands=3,
    ).to(device)
    
    # Analyze signals
    signals = torch.stack([
        torch.from_numpy(signal_no_coupling),
        torch.from_numpy(signal_weak),
        torch.from_numpy(signal_strong)
    ]).float().unsqueeze(1).to(device)  # (3, 1, n_samples)
    
    with torch.no_grad():
        pac_results = pac_analyzer(signals)
    
    pac_values = pac_results["pac"].cpu().numpy()
    
    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot signals
    t = np.arange(generator.n_samples) / generator.fs
    titles = ["No Coupling", "Weak Coupling", "Strong Coupling"]
    signals_list = [signal_no_coupling, signal_weak, signal_strong]
    
    for i, (sig, title) in enumerate(zip(signals_list, titles)):
        ax = axes[0, i]
        ax.plot(t[:256], sig[:256], 'b-', linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Plot PAC matrices
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(pac_values[i, 0], aspect='auto', origin='lower', 
                      cmap='hot', vmin=0, vmax=pac_values.max())
        ax.set_xlabel("Amplitude Band")
        ax.set_ylabel("Phase Band")
        ax.set_title(f"PAC Matrix: {titles[i]}")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save figure
    spath = "synthetic_data_demo.gif"
    mngs.io.save(fig, spath)
    
    # Report results
    mngs.str.printc("\n📊 PAC Analysis Results", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    for i, title in enumerate(titles):
        max_pac = pac_values[i, 0].max()
        mngs.str.printc(f"{title}: max PAC = {max_pac:.4f}", c="cyan")
    
    mngs.str.printc("\n✅ SyntheticDataGenerator demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to: {spath}", c="green")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="SyntheticDataGenerator demonstration")
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt
    
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mngs
    
    args = parse_args()
    
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )
    
    exit_status = main(args)
    
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()