#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 01:10:00 (ywatanabe)"
# File: ./examples/gpac/example_SyntheticDataGenerator.py

"""
Functionalities:
  - Demonstrates synthetic PAC signal generation
  - Shows both simple generate_pac_signal and full SyntheticDataGenerator
  - Creates datasets for machine learning with different PAC classes
  - Visualizes generated signals and their PAC properties
  - Demonstrates dataset splitting for train/val/test

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
    - None
    
  - output-files:
    - ./example_SyntheticDataGenerator_out/synthetic_pac_demo.gif
    - ./example_SyntheticDataGenerator_out/dataset_statistics.csv
    - ./example_SyntheticDataGenerator_out/synthetic_datasets.pt
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""

def analyze_pac_with_gpac(signal, fs, device='cpu'):
    """Analyze PAC in generated signal using gPAC."""
    from gpac import PAC
    
    # Prepare signal for PAC analysis
    if signal.ndim == 1:
        signal = signal.reshape(1, 1, 1, -1)
    elif signal.ndim == 3:
        signal = signal.unsqueeze(1)  # Add channel dimension
    
    signal_torch = torch.from_numpy(signal).float().to(device)
    
    # Initialize PAC analyzer
    pac = PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=10,
        amp_start_hz=30,
        amp_end_hz=100,
        amp_n_bands=10,
    ).to(device)
    
    # Calculate PAC
    with torch.no_grad():
        results = pac(signal_torch)
    
    return results


def main(args):
    """Run SyntheticDataGenerator demonstration."""
    import mngs
    from gpac import generate_pac_signal, SyntheticDataGenerator
    
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
    amp_freq = 80  # Gamma
    
    mngs.str.printc(f"\nGenerating signal: θ={phase_freq} Hz → γ={amp_freq} Hz", c="cyan")
    signal = generate_pac_signal(
        duration=duration,
        fs=fs,
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=0.8,
        noise_level=0.1,
    )
    
    mngs.str.printc(f"Signal shape: {signal.shape}", c="cyan")
    mngs.str.printc(f"Signal stats: mean={signal.mean():.4f}, std={signal.std():.4f}", c="cyan")
    
    # Analyze PAC in generated signal
    mngs.str.printc("\n🔍 Analyzing PAC in generated signal...", c="blue")
    pac_results = analyze_pac_with_gpac(signal, fs, device)
    
    pac_matrix = pac_results['pac'][0, 0].cpu().numpy()
    max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
    pha_freqs = pac_results['phase_frequencies'].numpy()
    amp_freqs = pac_results['amplitude_frequencies'].numpy()
    
    mngs.str.printc(f"Peak PAC at: θ={pha_freqs[max_idx[0]]:.1f} Hz, γ={amp_freqs[max_idx[1]]:.1f} Hz", c="cyan")
    mngs.str.printc(f"Expected at: θ={phase_freq} Hz, γ={amp_freq} Hz", c="yellow")
    
    # Part 2: Full SyntheticDataGenerator for ML datasets
    mngs.str.printc("\n🎯 Part 2: SyntheticDataGenerator for ML", c="yellow")
    mngs.str.printc("=" * 60, c="yellow")
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        fs=512.0,
        duration_sec=1.0,
        random_seed=42
    )
    
    # Show class definitions
    mngs.str.printc("\n📋 PAC Class Definitions:", c="cyan")
    for class_id, class_info in generator.class_definitions.items():
        mngs.str.printc(
            f"  Class {class_id} ({class_info['name']}): "
            f"θ={class_info['pha_range']} Hz, γ={class_info['amp_range']} Hz, "
            f"strength={class_info['coupling_strength']:.2f}",
            c="cyan"
        )
    
    # Generate and split dataset
    mngs.str.printc("\n🔄 Generating datasets...", c="blue")
    datasets = generator.generate_and_split(train_ratio=0.7, val_ratio=0.2)
    
    mngs.str.printc(f"\n📊 Dataset sizes:", c="green")
    mngs.str.printc(f"  Train: {len(datasets['train'])} samples", c="cyan")
    mngs.str.printc(f"  Val: {len(datasets['val'])} samples", c="cyan") 
    mngs.str.printc(f"  Test: {len(datasets['test'])} samples", c="cyan")
    
    # Create DataLoader example
    train_loader = DataLoader(
        datasets['train'],
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    # Get a batch
    batch_signals, batch_labels, batch_metadata = next(iter(train_loader))
    mngs.str.printc(f"\n📦 Batch shapes:", c="green")
    mngs.str.printc(f"  Signals: {batch_signals.shape}", c="cyan")
    mngs.str.printc(f"  Labels: {batch_labels.shape}", c="cyan")
    
    # Create visualization
    mngs.str.printc("\n📊 Creating visualization...", c="cyan")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Simple generated signal
    ax = axes[0, 0]
    t = np.arange(len(signal)) / fs
    ax.plot(t[:500], signal[:500], 'b-', linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Simple PAC Signal (θ={phase_freq}Hz→γ={amp_freq}Hz)")
    ax.grid(True, alpha=0.3)
    
    # PAC matrix for simple signal
    ax = axes[0, 1]
    im = ax.imshow(
        pac_matrix, 
        aspect='auto', 
        origin='lower',
        extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]],
        cmap='hot'
    )
    ax.set_xlabel("Amplitude Frequency (Hz)")
    ax.set_ylabel("Phase Frequency (Hz)")
    ax.set_title("PAC Analysis")
    ax.plot(amp_freq, phase_freq, 'wo', markersize=10, markeredgecolor='w', markeredgewidth=2)
    plt.colorbar(im, ax=ax)
    
    # Class distribution
    ax = axes[0, 2]
    class_counts = {}
    for _, label, _ in datasets['train']:
        class_id = label.item()
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    class_names = [generator.class_definitions[c]['name'] for c in classes]
    
    ax.bar(class_names, counts, color=['blue', 'green', 'red'][:len(classes)])
    ax.set_xlabel("PAC Class")
    ax.set_ylabel("Count")
    ax.set_title("Training Set Distribution")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Examples from each class
    for i in range(3):
        ax = axes[1, i]
        # Find sample from class i
        for sig, label, meta in datasets['train']:
            if label.item() == i:
                # Plot first channel, first segment
                sig_plot = sig[0, 0].numpy()
                t_plot = np.arange(len(sig_plot)) / generator.fs
                ax.plot(t_plot[:256], sig_plot[:256], linewidth=0.8)
                class_info = generator.class_definitions[i]
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Class {i}: {class_info['name']}\n"
                    f"θ={class_info['pha_range']}Hz, γ={class_info['amp_range']}Hz"
                )
                ax.grid(True, alpha=0.3)
                break
    
    # Row 3: Multi-channel and multi-segment visualization
    # Multi-channel example
    ax = axes[2, 0]
    example_signal = batch_signals[0].numpy()  # (channels, segments, time)
    for ch in range(example_signal.shape[0]):
        ax.plot(
            example_signal[ch, 0, :256] + ch * 3,  # Offset for visibility
            label=f'Channel {ch+1}'
        )
    ax.set_xlabel("Time samples")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multi-channel Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Multi-segment example
    ax = axes[2, 1]
    for seg in range(example_signal.shape[1]):
        ax.plot(
            example_signal[0, seg, :256] + seg * 3,  # Offset for visibility
            label=f'Segment {seg+1}'
        )
    ax.set_xlabel("Time samples")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multi-segment Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metadata visualization
    ax = axes[2, 2]
    metadata_text = "Sample Metadata:\n\n"
    sample_meta = batch_metadata
    metadata_text += f"Phase freqs: {sample_meta['phase_freq'][0]:.1f} Hz\n"
    metadata_text += f"Amp freqs: {sample_meta['amp_freq'][0]:.1f} Hz\n"
    metadata_text += f"Coupling: {sample_meta['coupling_strength'][0]:.2f}\n"
    metadata_text += f"Noise: {sample_meta['noise_level'][0]:.2f}\n"
    metadata_text += f"Class: {sample_meta['class_name'][0]}"
    
    ax.text(0.1, 0.5, metadata_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center')
    ax.axis('off')
    ax.set_title("Dataset Metadata Example")
    
    # Save figure
    spath = "synthetic_pac_demo.gif"
    mngs.io.save(fig, spath, symlink_from_cwd=True)
    
    # Save dataset statistics
    stats = {
        'n_train': len(datasets['train']),
        'n_val': len(datasets['val']),
        'n_test': len(datasets['test']),
        'n_classes': generator.params['n_classes'],
        'fs': generator.fs,
        'duration_sec': generator.duration_sec,
        'n_channels': generator.params['n_channels'],
        'n_segments': generator.params['n_segments'],
        'class_definitions': generator.class_definitions
    }
    mngs.io.save(
        stats, 
        "dataset_statistics.csv",
        symlink_from_cwd=True
    )
    
    # Save datasets
    torch.save(
        datasets,
        "synthetic_datasets.pt"
    )
    mngs.str.printc(f"\n💾 Saved datasets to synthetic_datasets.pt", c="green")
    
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

# EOF
