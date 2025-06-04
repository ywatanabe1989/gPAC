#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-30 18:15:00 (ywatanabe)"
# File: example_ModulationIndex.py

# ----------------------------------------
import os
__FILE__ = (
    "./examples/gpac/example_modulation_index.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Demonstrates ModulationIndex computation for quantifying PAC strength
  - Shows batch processing of multiple channels/signals
  - Implements permutation testing for statistical significance
  - Compares different parameter settings (n_bins)

Dependencies:
  - scripts: None
  - packages: numpy, torch, matplotlib, gpac, mngs

IO:
  - input-files: None (generates synthetic PAC signals)
  - output-files: ./examples/gpac/example_ModulationIndex_out/figures/modulation_index_example.png
"""

"""Imports"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def create_pac_signal(t, pac_strength=0.5, theta_freq=6, gamma_freq=40):
    """Create signal with theta-gamma PAC."""
    # Theta phase
    theta_phase = 2 * np.pi * theta_freq * t

    # Gamma amplitude modulated by theta phase
    gamma_amplitude = 1 + pac_strength * np.cos(theta_phase)
    gamma_signal = gamma_amplitude * np.sin(2 * np.pi * gamma_freq * t)

    # Composite signal
    signal = np.sin(theta_phase) + gamma_signal + 0.1 * np.random.randn(len(t))
    return signal, theta_phase, gamma_amplitude


def permutation_test(phase, amplitude, mi_calculator, n_perms=200):
    """Compute null distribution of MI values using permutation testing."""
    # Original MI
    mi_original = mi_calculator(phase, amplitude).item()

    # Permuted MIs
    mi_permuted = []
    for _ in range(n_perms):
        # Randomly shift amplitude
        shift = np.random.randint(0, phase.shape[-1])
        amp_shifted = torch.roll(amplitude, shifts=shift, dims=-1)
        mi_perm = mi_calculator(phase, amp_shifted).item()
        mi_permuted.append(mi_perm)

    # Compute z-score
    mi_permuted = np.array(mi_permuted)
    z_score = (mi_original - mi_permuted.mean()) / mi_permuted.std()
    p_value = (mi_permuted >= mi_original).sum() / n_perms

    return mi_original, z_score, p_value, mi_permuted


def main(args):
    """Main function to demonstrate ModulationIndex usage."""
    import mngs
    from gpac import ModulationIndex
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parameters
    n_samples = int(args.fs * args.duration)
    t = np.linspace(0, args.duration, n_samples)

    mngs.str.printc("ModulationIndex Example", c='yellow')
    print("=" * 50)

    # Example 1: Single signal MI computation
    print("\n1. Computing MI for single signal with strong PAC...")
    signal_strong, phase_true, amp_true = create_pac_signal(t, pac_strength=0.8)

    # For MI computation, we need phase and amplitude envelopes
    # In practice, these come from Hilbert transform of filtered signals
    # Here we'll use the true values for demonstration
    phase_tensor = (
        torch.tensor(phase_true, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    amplitude_tensor = (
        torch.tensor(amp_true, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    # Initialize MI calculator
    mi_calculator = ModulationIndex(n_bins=args.n_bins)

    # Compute MI
    mi_value = mi_calculator(phase_tensor, amplitude_tensor)
    print(f"Modulation Index: {mi_value.item():.4f}")

    # Example 2: Compare different PAC strengths
    print("\n2. Comparing MI values for different PAC strengths...")
    pac_strengths = [0.0, 0.2, 0.5, 0.8, 1.0]
    mi_values = []

    for strength in pac_strengths:
        signal, phase, amplitude = create_pac_signal(t, pac_strength=strength)
        phase_t = torch.tensor(phase, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        amp_t = torch.tensor(amplitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mi = mi_calculator(phase_t, amp_t)
        mi_values.append(mi.item())
        print(f"  PAC strength: {strength:.1f} -> MI: {mi.item():.4f}")

    # Example 3: Batch processing
    print("\n3. Batch processing multiple channels...")
    batch_size = 8
    n_channels = 4

    # Create batch of signals with varying PAC
    batch_phases = []
    batch_amplitudes = []

    for i in range(batch_size):
        pac_strength = np.random.uniform(0, 1)
        _, phase, amplitude = create_pac_signal(t, pac_strength)
        batch_phases.append(phase)
        batch_amplitudes.append(amplitude)

    # Stack into tensors
    batch_phase_tensor = torch.tensor(
        np.array(batch_phases), dtype=torch.float32
    ).unsqueeze(1)
    batch_amp_tensor = torch.tensor(
        np.array(batch_amplitudes), dtype=torch.float32
    ).unsqueeze(1)

    print(f"Batch phase shape: {batch_phase_tensor.shape}")
    print(f"Batch amplitude shape: {batch_amp_tensor.shape}")

    # Compute MI for batch
    batch_mi = mi_calculator(batch_phase_tensor, batch_amp_tensor)
    print(f"Batch MI shape: {batch_mi.shape}")
    print(f"MI values: {batch_mi.squeeze().numpy()}")

    # Example 4: Statistical significance with permutation testing
    print("\n4. Permutation testing for statistical significance...")

    # Test with strong PAC signal
    signal_test, phase_test, amp_test = create_pac_signal(t, pac_strength=0.7)
    phase_test_t = (
        torch.tensor(phase_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    amp_test_t = torch.tensor(amp_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    mi_orig, z_score, p_value, null_dist = permutation_test(
        phase_test_t, amp_test_t, mi_calculator, n_perms=args.n_perms
    )
    print(f"Original MI: {mi_orig:.4f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"P-value: {p_value:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. Signal with PAC
    time_window = (t >= 2) & (t <= 3)
    axes[0, 0].plot(t[time_window], signal_strong[time_window], "k-", alpha=0.7)
    axes[0, 0].set_title("Signal with Strong PAC")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    # 2. Phase-amplitude relationship
    phase_bins = np.linspace(-np.pi, np.pi, args.n_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    binned_amps = []

    for i in range(len(phase_bins) - 1):
        mask = (phase_true >= phase_bins[i]) & (phase_true < phase_bins[i + 1])
        if mask.sum() > 0:
            binned_amps.append(amp_true[mask].mean())
        else:
            binned_amps.append(0)

    axes[0, 1].bar(range(args.n_bins), binned_amps, color="purple", alpha=0.7)
    axes[0, 1].set_title(f"Phase-Amplitude Distribution (MI={mi_values[3]:.3f})")
    axes[0, 1].set_xlabel("Phase Bin")
    axes[0, 1].set_ylabel("Mean Amplitude")
    axes[0, 1].set_xticks([0, args.n_bins // 2, args.n_bins])
    axes[0, 1].set_xticklabels(["-π", "0", "π"])

    # 3. MI vs PAC strength
    axes[0, 2].plot(pac_strengths, mi_values, "o-", linewidth=2, markersize=8)
    axes[0, 2].set_title("MI vs PAC Strength")
    axes[0, 2].set_xlabel("PAC Strength")
    axes[0, 2].set_ylabel("Modulation Index")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Batch MI distribution
    axes[1, 0].hist(batch_mi.squeeze().numpy(), bins=20, alpha=0.7, color="green")
    axes[1, 0].set_title("Distribution of Batch MI Values")
    axes[1, 0].set_xlabel("Modulation Index")
    axes[1, 0].set_ylabel("Count")

    # 5. Permutation test results
    axes[1, 1].hist(null_dist, bins=30, alpha=0.7, color="gray", label="Null")
    axes[1, 1].axvline(
        mi_orig, color="red", linewidth=2, label=f"Original (z={z_score:.2f})"
    )
    axes[1, 1].set_title("Permutation Test")
    axes[1, 1].set_xlabel("Modulation Index")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    # 6. Different n_bins comparison
    n_bins_list = [9, 18, 36, 72]
    mi_by_bins = []

    for n_bins in n_bins_list:
        mi_calc_bins = ModulationIndex(n_bins=n_bins)
        mi_val = mi_calc_bins(phase_test_t, amp_test_t).item()
        mi_by_bins.append(mi_val)

    axes[1, 2].plot(n_bins_list, mi_by_bins, "o-", linewidth=2, markersize=8)
    axes[1, 2].set_title("MI vs Number of Phase Bins")
    axes[1, 2].set_xlabel("Number of Bins")
    axes[1, 2].set_ylabel("Modulation Index")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save using mngs
    save_path = CONFIG.PATH.FIGURES / "modulation_index_example.png"
    mngs.io.save(fig, save_path)
    print(f"\nSaved visualization to: {save_path}")
    plt.close(fig)

    # GPU acceleration example
    if args.use_gpu and torch.cuda.is_available():
        print("\n5. GPU-accelerated MI computation...")
        device = torch.device("cuda")

        # Move to GPU
        mi_gpu = mi_calculator.to(device)
        phase_gpu = batch_phase_tensor.to(device)
        amp_gpu = batch_amp_tensor.to(device)

        # Compute on GPU
        mi_values_gpu = mi_gpu(phase_gpu, amp_gpu)

        print(f"GPU computation successful!")
        print(f"MI values on GPU: {mi_values_gpu.cpu().squeeze().numpy()}")

    print("\n6. Summary:")
    print("- ModulationIndex quantifies PAC strength using entropy-based method")
    print("- MI increases with stronger phase-amplitude coupling")
    print("- Permutation testing provides statistical significance")
    print("- Supports batch processing and GPU acceleration")
    print("- Choice of n_bins affects sensitivity (18 bins is standard)")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description='ModulationIndex example for gPAC')
    parser.add_argument(
        "--fs", type=float, default=250.0, help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Signal duration (seconds)"
    )
    parser.add_argument("--n_bins", type=int, default=18, help="Number of phase bins")
    parser.add_argument(
        "--n_perms",
        type=int,
        default=100,
        help="Number of permutations for significance testing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU acceleration if available"
    )
    args = parser.parse_args()
    mngs.str.printc(args, c='yellow')
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == '__main__':
    run_main()

# EOF