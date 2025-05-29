#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 01:05:00 (ywatanabe)"
# File: ./examples/gpac/example_ModulationIndex.py

"""
Functionalities:
  - Demonstrates ModulationIndex calculation in gPAC
  - Shows proper input format and dimensions
  - Visualizes amplitude distributions across phase bins
  - Demonstrates KL divergence calculation for PAC
  - Shows how to interpret modulation index results

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch
    - numpy
    - matplotlib
    - mngs
    
IO:
  - input-files:
    - None (generates synthetic data)
    
  - output-files:
    - ./scripts/example_ModulationIndex/modulation_index_demo.png
    - ./scripts/example_ModulationIndex/mi_results.npz
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""

def create_pac_coupled_signals(n_samples=1000, n_epochs=10, coupling_strength=0.7):
    """Create phase and amplitude signals with known coupling."""
    # Time vector
    t = np.linspace(0, 2*np.pi, n_samples)
    
    # Phase signal (theta rhythm, ~6 Hz)
    phase = np.zeros((1, 1, 1, n_epochs, n_samples))  # Single phase band
    for ep in range(n_epochs):
        phase_shift = np.random.uniform(0, 2*np.pi)
        phase[0, 0, 0, ep, :] = t + phase_shift
    
    # Amplitude signal (gamma rhythm, ~80 Hz) modulated by phase
    amplitude = np.zeros((1, 1, 1, n_epochs, n_samples))  # Single amplitude band
    for ep in range(n_epochs):
        # Create amplitude modulation based on phase
        modulation = 1 + coupling_strength * np.cos(phase[0, 0, 0, ep, :])
        # Add some baseline amplitude with modulation
        amplitude[0, 0, 0, ep, :] = modulation + 0.2 * np.random.randn(n_samples)
    
    # Convert to torch tensors
    phase_torch = torch.tensor(phase, dtype=torch.float32)
    amplitude_torch = torch.tensor(amplitude, dtype=torch.float32)
    
    return phase_torch, amplitude_torch


def main(args):
    """Run ModulationIndex demonstration."""
    import mngs
    from gpac import ModulationIndex
    
    mngs.str.printc("🚀 ModulationIndex Demonstration", c="green")
    mngs.str.printc("=" * 50, c="green")
    
    # Parameters
    n_samples = 1000
    n_epochs = 10
    n_bins = 18
    
    # Test 1: Single frequency pair with coupling
    mngs.str.printc("\n🎯 Test 1: Single frequency pair with coupling", c="cyan")
    phase_coupled, amplitude_coupled = create_pac_coupled_signals(
        n_samples=n_samples, n_epochs=n_epochs, coupling_strength=0.8
    )
    
    # Test 2: Multiple frequency bands
    mngs.str.printc("\n🎯 Test 2: Multiple frequency bands", c="cyan")
    n_pha_bands = 3
    n_amp_bands = 4
    
    # Create multi-band signals
    phase_multi = torch.randn(1, 1, n_pha_bands, n_epochs, n_samples)
    amplitude_multi = torch.randn(1, 1, n_amp_bands, n_epochs, n_samples)
    
    # Add coupling to specific band pair (band 1 phase -> band 2 amplitude)
    t = torch.linspace(0, 2*np.pi, n_samples)
    for ep in range(n_epochs):
        phase_multi[0, 0, 1, ep, :] = t + torch.rand(1) * 2 * np.pi
        modulation = 1 + 0.7 * torch.cos(phase_multi[0, 0, 1, ep, :])
        amplitude_multi[0, 0, 2, ep, :] = modulation + 0.2 * torch.randn(n_samples)
    
    # Initialize ModulationIndex
    mngs.str.printc("\n🔧 Initializing ModulationIndex...", c="blue")
    mi = ModulationIndex(n_bins=n_bins)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mi = mi.to(device)
    phase_coupled = phase_coupled.to(device)
    amplitude_coupled = amplitude_coupled.to(device)
    phase_multi = phase_multi.to(device)
    amplitude_multi = amplitude_multi.to(device)
    
    # Calculate MI for coupled signals
    mngs.str.printc("\n🔄 Calculating MI for coupled signals...", c="blue")
    results_coupled = mi(phase_coupled, amplitude_coupled)
    
    # Calculate MI for multi-band signals
    mngs.str.printc("🔄 Calculating MI for multi-band signals...", c="blue")
    results_multi = mi(phase_multi, amplitude_multi)
    
    mngs.str.printc("✅ Modulation index calculation completed", c="green")
    
    # Print results
    mngs.str.printc("\n📊 Results Summary", c="yellow")
    mngs.str.printc("=" * 50, c="yellow")
    
    mngs.str.printc("\nCoupled signals (single band):", c="cyan")
    mngs.str.printc(f"MI value: {results_coupled['mi'][0, 0, 0, 0].cpu().item():.4f}", c="cyan")
    mngs.str.printc(f"MI shape: {results_coupled['mi'].shape}", c="cyan")
    mngs.str.printc(f"MI per segment shape: {results_coupled['mi_per_segment'].shape}", c="cyan")
    
    mngs.str.printc("\nMulti-band signals:", c="cyan")
    mngs.str.printc(f"MI shape: {results_multi['mi'].shape}", c="cyan")
    mi_matrix = results_multi['mi'][0, 0].cpu().numpy()
    max_idx = np.unravel_index(mi_matrix.argmax(), mi_matrix.shape)
    mngs.str.printc(f"Max MI value: {mi_matrix.max():.4f} at phase band {max_idx[0]}, amp band {max_idx[1]}", c="cyan")
    mngs.str.printc(f"Expected coupling at: phase band 1, amp band 2", c="yellow")
    
    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot amplitude distribution for coupled signal
    ax = axes[0, 0]
    amp_dist = results_coupled['amplitude_distributions'][0, 0, 0, 0].cpu().numpy()
    phase_bins = results_coupled['phase_bin_centers'].cpu().numpy()
    
    im = ax.imshow(amp_dist.T, aspect='auto', origin='lower', cmap='hot')
    ax.set_xticks(np.arange(0, n_bins, 3))
    ax.set_xticklabels([f'{phase_bins[i]:.1f}' for i in range(0, n_bins, 3)])
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Amplitude bin")
    ax.set_title("Amplitude Distribution (Coupled)")
    plt.colorbar(im, ax=ax)
    
    # Plot phase-amplitude relationship
    ax = axes[0, 1]
    # Sample from first epoch
    phase_sample = phase_coupled[0, 0, 0, 0, :200].cpu().numpy()
    amp_sample = amplitude_coupled[0, 0, 0, 0, :200].cpu().numpy()
    scatter = ax.scatter(phase_sample, amp_sample, c=np.arange(200), cmap='viridis', s=20, alpha=0.6)
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Phase-Amplitude Relationship")
    plt.colorbar(scatter, ax=ax, label='Time')
    
    # Plot MI values for all segments
    ax = axes[0, 2]
    mi_per_seg = results_coupled['mi_per_segment'][0, 0, 0, 0].cpu().numpy()
    ax.plot(mi_per_seg, 'b-', linewidth=2)
    ax.axhline(y=results_coupled['mi'][0, 0, 0, 0].cpu().item(), color='r', linestyle='--', label='Mean MI')
    ax.set_xlabel("Segment")
    ax.set_ylabel("MI Value")
    ax.set_title("MI per Segment (Coupled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot multi-band MI matrix
    ax = axes[1, 0]
    im = ax.imshow(mi_matrix, aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel("Amplitude Band")
    ax.set_ylabel("Phase Band")
    ax.set_title("Multi-band MI Matrix")
    plt.colorbar(im, ax=ax, label='MI')
    
    # Mark expected coupling
    ax.plot(2, 1, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)
    ax.text(2.1, 1.1, 'Expected', color='white', fontsize=9)
    
    # Plot uniform distribution comparison
    ax = axes[1, 1]
    uniform_dist = np.ones(n_bins) / n_bins
    ax.bar(range(n_bins), amp_dist.mean(axis=1), alpha=0.7, label='Coupled signal')
    ax.plot(range(n_bins), uniform_dist, 'r--', linewidth=2, label='Uniform (no coupling)')
    ax.set_xlabel("Phase bin")
    ax.set_ylabel("Probability")
    ax.set_title("Average Amplitude Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot KL divergence interpretation
    ax = axes[1, 2]
    # Create signals with varying coupling strength
    coupling_strengths = np.linspace(0, 1, 10)
    mi_values = []
    
    for cs in coupling_strengths:
        phase_test, amp_test = create_pac_coupled_signals(
            n_samples=500, n_epochs=5, coupling_strength=cs
        )
        phase_test = phase_test.to(device)
        amp_test = amp_test.to(device)
        res = mi(phase_test, amp_test)
        mi_values.append(res['mi'][0, 0, 0, 0].cpu().item())
    
    ax.plot(coupling_strengths, mi_values, 'b-', linewidth=2, marker='o')
    ax.set_xlabel("Coupling Strength")
    ax.set_ylabel("MI Value")
    ax.set_title("MI vs Coupling Strength")
    ax.grid(True, alpha=0.3)
    
    # Save figure
    spath = "modulation_index_demo.png"
    mngs.io.save(fig, spath)
    
    # Save results
    results_data = {
        'mi_coupled': results_coupled['mi'].cpu().numpy(),
        'mi_multi': results_multi['mi'].cpu().numpy(),
        'amplitude_distributions': results_coupled['amplitude_distributions'].cpu().numpy(),
        'phase_bin_centers': results_coupled['phase_bin_centers'].cpu().numpy(),
        'phase_bin_edges': results_coupled['phase_bin_edges'].cpu().numpy(),
        'coupling_strengths': coupling_strengths,
        'mi_vs_coupling': mi_values
    }
    mngs.io.save(results_data, "mi_results.npz")
    
    mngs.str.printc("\n✅ ModulationIndex demo completed!", c="green")
    mngs.str.printc(f"💾 Results saved to: {spath}", c="green")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="ModulationIndex demonstration")
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

# EOF
