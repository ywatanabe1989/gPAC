#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 00:55:00 (ywatanabe)"
# File: ./examples/trainability/example_pac_trainability.py

"""
Functionalities:
  - Trains PAC model to learn optimal frequency bands through backpropagation
  - Optimizes frequency parameters to detect known PAC coupling
  - Visualizes the evolution of learned bands during training
  - Compares learned bands with ground truth frequencies
  - Saves training progress figures and learned parameters

Dependencies:
  - scripts:
    - None
  - packages:
    - gpac
    - torch
    - numpy
    - matplotlib
    - tqdm
    
IO:
  - input-files:
    - None (generates synthetic PAC signals)
    
  - output-files:
    - ./scripts/example_pac_trainability/pac_trainability_demo.png
    - ./scripts/example_pac_trainability/pac_trainability_demo.csv
    - ./scripts/example_pac_trainability/learned_parameters.pkl
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

"""Warnings"""
import warnings

warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""


def generate_training_batch(batch_size, fs, duration, true_phase_freq, true_amp_freq):
    """Generate a batch of signals with and without PAC."""
    from gpac._SyntheticDataGenerator import generate_pac_signal

    batch_signals = []
    batch_labels = []

    for i in range(batch_size):
        if np.random.rand() > 0.5:
            # Generate PAC signal
            signal = generate_pac_signal(
                duration=duration,
                fs=fs,
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=0.5,
                noise_level=0.1,
                random_seed=i
            )
            # Reshape to match expected format [n_epochs, n_channels, n_times]
            signal = signal[np.newaxis, np.newaxis, :]
            batch_signals.append(signal)
            batch_labels.append(1.0)  # Has PAC
        else:
            # Generate noise
            signal = np.random.randn(1, 1, int(fs * duration))
            batch_signals.append(signal)
            batch_labels.append(0.0)  # No PAC

    # Stack and convert to torch
    signals = torch.from_numpy(np.stack(batch_signals)).float()
    labels = torch.tensor(batch_labels).float()

    return signals, labels


def main(args):
    """Run PAC trainability demonstration."""
    import mngs
    from gpac import PAC
    from gpac._Profiler import create_profiler

    mngs.str.printc(
        "🚀 PAC Trainability Example: Learning Optimal Frequency Bands", c="green"
    )
    mngs.str.printc("=" * 70, c="green")

    # Create profiler
    profiler = create_profiler(enable_gpu=True)

    # Parameters
    fs = 256.0
    duration = 2.0
    n_samples = int(fs * duration)
    n_epochs = 100
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # True PAC parameters (what we want to learn)
    true_phase_freq = 10.0  # Alpha (center of band)
    true_amp_freq = 75.0  # Gamma (center of band)

    mngs.str.printc(f"\n📍 Device: {device}", c="cyan")
    mngs.str.printc(f"\n🎯 Target frequencies to learn:", c="yellow")
    mngs.str.printc(f"   Phase: {true_phase_freq} Hz (Alpha)", c="yellow")
    mngs.str.printc(f"   Amplitude: {true_amp_freq} Hz (Gamma)", c="yellow")

    # Initialize PAC with learnable bands
    with profiler.profile("Model Initialization"):
        pac_model = PAC(
            seq_len=n_samples,
            fs=fs,
            pha_start_hz=4,
            pha_end_hz=16,
            pha_n_bands=2,
            amp_start_hz=30,
            amp_end_hz=100,
            amp_n_bands=2,
            trainable=True,  # Enable trainable filters
        ).to(device)

    # Get initial bands
    if hasattr(pac_model.bandpass, "get_filter_banks"):
        initial_bands = pac_model.bandpass.get_filter_banks()
        mngs.str.printc("\n📊 Initial frequency bands:", c="cyan")
        mngs.str.printc(
            f"   Phase bands: {initial_bands['pha_bands'].cpu().numpy()}", c="cyan"
        )
        mngs.str.printc(
            f"   Amplitude bands: {initial_bands['amp_bands'].cpu().numpy()}", c="cyan"
        )

    # Optimizer
    optimizer = optim.Adam(pac_model.parameters(), lr=0.01)

    # Loss function
    criterion = nn.BCELoss()

    # Storage for visualization
    losses = []
    phase_bands_history = []
    amp_bands_history = []

    # Training loop
    mngs.str.printc("\n🔄 Training to find optimal frequency bands...", c="blue")

    with profiler.profile("Training Loop"):
        for epoch in tqdm(range(n_epochs), desc="Training"):
            # Generate training batch
            signals, labels = generate_training_batch(
                batch_size, fs, duration, true_phase_freq, true_amp_freq
            )
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            pac_values = pac_model(signals)  # [batch, n_pha, n_amp]

            # Compute mean PAC strength as prediction
            pac_strength = pac_values.mean(dim=(1, 2))  # [batch]

            # Apply sigmoid for binary classification
            predictions = torch.sigmoid(pac_strength * 10)  # Scale for better gradients

            # Compute loss
            loss = criterion(predictions, labels)

            # Add regularization from filter
            if hasattr(pac_model.bandpass, "get_regularization_loss"):
                reg_loss = pac_model.bandpass.get_regularization_loss()
                total_loss = loss + 0.01 * reg_loss
            else:
                total_loss = loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Constrain parameters
            if hasattr(pac_model.bandpass, "constrain_parameters"):
                pac_model.bandpass.constrain_parameters()

            # Record for visualization
            losses.append(loss.item())

            # Record bands every 10 epochs
            if epoch % 10 == 0:
                if hasattr(pac_model.bandpass, "get_filter_banks"):
                    bands = pac_model.bandpass.get_filter_banks()
                    phase_bands_history.append(bands["pha_bands"].cpu().numpy())
                    amp_bands_history.append(bands["amp_bands"].cpu().numpy())

    # Get final bands
    final_bands = None
    if hasattr(pac_model.bandpass, "get_filter_banks"):
        final_bands = pac_model.bandpass.get_filter_banks()
        mngs.str.printc("\n✅ Final learned frequency bands:", c="green")
        mngs.str.printc(
            f"   Phase bands: {final_bands['pha_bands'].cpu().numpy()}", c="green"
        )
        mngs.str.printc(
            f"   Amplitude bands: {final_bands['amp_bands'].cpu().numpy()}", c="green"
        )

    # Visualization
    with profiler.profile("Visualization"):
        fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten for 1D indexing

        # Plot loss curve
        ax = axes[0]
        ax.plot(losses)
        ax.set_xyt("Epoch", "Loss", "Training Loss")
        ax.grid(True, alpha=0.3)

        # Plot phase band evolution
        ax = axes[1]
        if phase_bands_history:
            phase_bands_history = np.array(phase_bands_history)
            for i in range(phase_bands_history.shape[1]):
                ax.plot(
                    np.arange(0, n_epochs, 10),
                    phase_bands_history[:, i, 0],
                    "b--",
                    alpha=0.5,
                    label=f"Band {i} low",
                )
                ax.plot(
                    np.arange(0, n_epochs, 10),
                    phase_bands_history[:, i, 1],
                    "b-",
                    alpha=0.8,
                    label=f"Band {i} high",
                )
        ax.axhline(
            true_phase_freq, color="r", linestyle=":", linewidth=2, label="Target"
        )
        ax.set_xyt("Epoch", "Frequency (Hz)", "Phase Band Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot amplitude band evolution
        ax = axes[2]
        if amp_bands_history:
            amp_bands_history = np.array(amp_bands_history)
            for i in range(amp_bands_history.shape[1]):
                ax.plot(
                    np.arange(0, n_epochs, 10),
                    amp_bands_history[:, i, 0],
                    "g--",
                    alpha=0.5,
                    label=f"Band {i} low",
                )
                ax.plot(
                    np.arange(0, n_epochs, 10),
                    amp_bands_history[:, i, 1],
                    "g-",
                    alpha=0.8,
                    label=f"Band {i} high",
                )
        ax.axhline(true_amp_freq, color="r", linestyle=":", linewidth=2, label="Target")
        ax.set_xyt("Epoch", "Frequency (Hz)", "Amplitude Band Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Test on new data
        ax = axes[3]
        from gpac._SyntheticDataGenerator import generate_pac_signal

        test_signal = generate_pac_signal(
            n_epochs=1,
            n_channels=1,
            n_times=n_samples,
            pha_freq=true_phase_freq,
            amp_freq=true_amp_freq,
            coupling_strength=0.8,
            fs=fs,
        )
        test_signal = torch.from_numpy(test_signal).float().to(device)

        with torch.no_grad():
            pac_result = pac_model(test_signal.unsqueeze(0))
            pac_map = pac_result[0].cpu().numpy()

        im = ax.imshow(pac_map, aspect="auto", origin="lower", cmap="hot")
        ax.set_xyt(
            "Amplitude band index",
            "Phase band index",
            "PAC Detection with Learned Bands",
        )
        plt.colorbar(im, ax=ax)

    # Save figure
    spath = "./scripts/example_pac_trainability/pac_trainability_demo.png"
    mngs.io.save(fig, spath, symlink_from_cwd=True)

    # Save learned parameters
    if final_bands is not None:
        params = {
            "phase_bands": final_bands["pha_bands"].cpu().numpy(),
            "amplitude_bands": final_bands["amp_bands"].cpu().numpy(),
            "losses": losses,
            "true_phase_freq": true_phase_freq,
            "true_amp_freq": true_amp_freq,
        }
        mngs.io.save(
            params,
            "./scripts/example_pac_trainability/learned_parameters.pkl",
            symlink_from_cwd=True,
        )

    # Print profiling summary
    mngs.str.printc("\n" + "=" * 70, c="green")
    profiler.print_summary()

    # Check if target frequencies were learned
    if final_bands is not None:
        phase_bands_final = final_bands["pha_bands"].cpu().numpy()
        amp_bands_final = final_bands["amp_bands"].cpu().numpy()

        # Find closest bands to target
        phase_contained = any(
            (band[0] <= true_phase_freq <= band[1]) for band in phase_bands_final
        )
        amp_contained = any(
            (band[0] <= true_amp_freq <= band[1]) for band in amp_bands_final
        )

        mngs.str.printc(f"\n🎯 Target frequency detection:", c="yellow")
        mngs.str.printc(
            f"   Phase {true_phase_freq} Hz contained in learned bands: {phase_contained}",
            c="yellow",
        )
        mngs.str.printc(
            f"   Amplitude {true_amp_freq} Hz contained in learned bands: {amp_contained}",
            c="yellow",
        )

    mngs.str.printc(f"\n✅ Training complete! Results saved to: {spath}", c="green")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(
        description="Train PAC model to learn optimal frequency bands"
    )
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
