#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 12:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/readme_demo.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/readme_demo.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
gPAC Package Demo: PAC Analysis with GPU Acceleration
=====================================================

This demo shows how to use the gPAC package to:
1. Generate synthetic PAC signals with different coupling properties
2. Calculate PAC using gPAC with TensorPAC-compatible filter design
3. Compare with original Tensorpac (if available)
4. Visualize results and performance differences

Features demonstrated:
- SyntheticDataGenerator for creating realistic PAC signals
- calculate_pac function for fast PAC computation
- TensorPAC-compatible filter design by default
- GPU acceleration benefits
- Visualization of raw signals and PAC results
- Fair performance comparison (post-initialization)
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# Import gPAC components
import gpac
import matplotlib.pyplot as plt
import numpy as np
import torch

# Try to import tensorpac for comparison
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
    print("✅ Tensorpac available for comparison")
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("⚠️  Tensorpac not available - using gPAC only")


def create_demo_signal():
    """Create a demo signal with known PAC coupling."""
    # Signal parameters
    fs = 512.0  # Sampling frequency
    duration = 5.0  # Duration in seconds

    # Create time vector
    t = np.linspace(0, duration, int(fs * duration))

    # PAC parameters - theta-gamma coupling
    pha_freq = 6.0  # Hz (theta)
    amp_freq = 80.0  # Hz (gamma)
    coupling_strength = 0.8

    # Generate phase signal (theta oscillation)
    phase_signal = np.sin(2 * np.pi * pha_freq * t)

    # Generate amplitude modulation based on phase
    modulation = (1 + coupling_strength * np.cos(2 * np.pi * pha_freq * t)) / 2

    # Generate carrier signal (gamma oscillation)
    carrier = np.sin(2 * np.pi * amp_freq * t)

    # Apply modulation to carrier
    modulated_carrier = modulation * carrier

    # Combine signals
    pac_signal = phase_signal + 0.5 * modulated_carrier

    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    signal = pac_signal + noise

    # Reshape to gPAC format: (batch, channels, segments, time)
    signal_4d = signal.reshape(1, 1, 1, -1)

    return signal_4d, fs, t, pha_freq, amp_freq


def calculate_gpac_pac_fair(signal, fs, pha_n_bands=50, amp_n_bands=30):
    """
    Calculate PAC using gPAC with fair timing (post-initialization).
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        pha_n_bands: Number of phase bands
        amp_n_bands: Number of amplitude bands
        
    Returns:
        pac_values, pha_freqs, amp_freqs, computation_time, setup_time
    """
    print(f"🔄 Setting up gPAC model...")
    setup_start = time.time()
    
    # Initialize model
    model = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=pha_n_bands,
        amp_start_hz=60.0,
        amp_end_hz=120.0,
        amp_n_bands=amp_n_bands,
        n_perm=None,
    )
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    signal_torch = torch.tensor(signal, dtype=torch.float32).to(device)
    
    setup_time = time.time() - setup_start
    print(f"✅ gPAC setup completed in {setup_time:.3f} seconds")
    
    # Warm up (important for GPU)
    print("🔄 Warming up gPAC...")
    with torch.no_grad():
        _ = model(signal_torch)
    
    # Time computation only
    print("🔄 Computing PAC with gPAC...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    comp_start = time.time()
    
    with torch.no_grad():
        pac_values = model(signal_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    comp_time = time.time() - comp_start
    
    # Get frequency vectors
    pha_freqs = model.PHA_MIDS_HZ.cpu().numpy()
    amp_freqs = model.AMP_MIDS_HZ.cpu().numpy()
    
    print(f"✅ gPAC computation completed in {comp_time:.3f} seconds")
    
    return pac_values, pha_freqs, amp_freqs, comp_time, setup_time


def calculate_tensorpac_pac_fair(signal, fs, pha_n_bands=50, amp_n_bands=30):
    """Calculate PAC using Tensorpac with fair timing (post-initialization)."""
    if not TENSORPAC_AVAILABLE:
        return None, None, None, None, None

    print("🔄 Setting up Tensorpac model...")
    setup_start = time.time()

    # Convert signal format for tensorpac (time, trials)
    signal_tp = signal[0, 0, 0, :].reshape(-1, 1)  # (time, trials)

    # Create Tensorpac object with matching parameters
    f_pha = np.linspace(2, 20, pha_n_bands)
    f_amp = np.linspace(60, 120, amp_n_bands)
    
    pac_tp = Pac(
        idpac=(2, 0, 0),  # Modulation Index
        f_pha=f_pha,
        f_amp=f_amp,
        dcomplex='hilbert',
        cycle=(3, 6),  # Match TensorPAC defaults
    )
    
    # The Pac object initialization includes filter creation
    # To be fair, we need to trigger the filter initialization
    # This happens on first call to filterfit, so let's do a dummy run
    print("🔄 Initializing Tensorpac filters...")
    dummy_signal = np.random.randn(1, 100)
    _ = pac_tp.filterfit(fs, dummy_signal, n_perm=0)
    
    setup_time = time.time() - setup_start
    print(f"✅ Tensorpac setup completed in {setup_time:.3f} seconds")

    try:
        # Time computation only (filters already initialized)
        print("🔄 Computing PAC with Tensorpac...")
        comp_start = time.time()
        
        # Calculate PAC
        pac_values_tp = pac_tp.filterfit(fs, signal_tp.T, n_perm=0)
        
        comp_time = time.time() - comp_start
        
        # Transpose to match gPAC format
        pac_values_tp = pac_values_tp.squeeze().T  # (pha, amp) format

        print(f"✅ Tensorpac computation completed in {comp_time:.3f} seconds")

        return pac_values_tp, f_pha, f_amp, comp_time, setup_time

    except Exception as e:
        print(f"⚠️  Tensorpac calculation failed: {e}")
        return None, None, None, None, None


def create_comprehensive_visualization(results_dict, signal, fs, t, pha_freq, amp_freq):
    """Create a comprehensive visualization showing all results with aligned axes."""
    
    n_methods = len(results_dict)
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 4 + 3 * ((n_methods + 1) // 2)))
    gs = fig.add_gridspec(
        2 + ((n_methods + 1) // 2), 
        2 if n_methods > 1 else 1,
        height_ratios=[1] + [1] * ((n_methods + 1) // 2) + [0.8]
    )

    # Top panel: Raw signal
    ax_signal = fig.add_subplot(gs[0, :])
    signal_1d = signal[0, 0, 0, :]
    ax_signal.plot(t[:1000], signal_1d[:1000], 'k-', linewidth=1)
    ax_signal.set_title(
        f"Synthetic PAC Signal (θ={pha_freq}Hz modulating γ={amp_freq}Hz)",
        fontsize=14,
        fontweight="bold",
    )
    ax_signal.set_xlabel("Time (s)")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.grid(True, alpha=0.3)

    # Add text annotation
    ax_signal.text(
        0.02,
        0.95,
        f"Sampling Rate: {fs} Hz\nDuration: {len(t)/fs:.1f} s\nShowing first 1000 samples",
        transform=ax_signal.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Find common color scale and axis ranges for all PAC plots
    all_pac_values = []
    all_pha_freqs = []
    all_amp_freqs = []
    
    for name, data in results_dict.items():
        if data['pac'] is not None:
            pac_2d = data['pac'].cpu().numpy() if hasattr(data['pac'], 'cpu') else data['pac']
            if pac_2d.ndim > 2:
                pac_2d = pac_2d[0, 0]
            all_pac_values.append(pac_2d)
            all_pha_freqs.append(data['pha_freqs'])
            all_amp_freqs.append(data['amp_freqs'])
    
    if all_pac_values:
        vmin = min(pac.min() for pac in all_pac_values)
        vmax = max(pac.max() for pac in all_pac_values)
        # Common frequency ranges
        pha_min = min(freqs[0] for freqs in all_pha_freqs)
        pha_max = max(freqs[-1] for freqs in all_pha_freqs)
        amp_min = min(freqs[0] for freqs in all_amp_freqs)
        amp_max = max(freqs[-1] for freqs in all_amp_freqs)
    else:
        vmin, vmax = 0, 1
        pha_min, pha_max = 2, 20
        amp_min, amp_max = 60, 120

    # PAC results
    row = 1
    col = 0
    for i, (name, data) in enumerate(results_dict.items()):
        if data['pac'] is None:
            continue
            
        ax = fig.add_subplot(gs[row, col])
        
        pac_2d = data['pac'].cpu().numpy() if hasattr(data['pac'], 'cpu') else data['pac']
        if pac_2d.ndim > 2:
            pac_2d = pac_2d[0, 0]  # Extract from tensor if needed
        
        im = ax.imshow(
            pac_2d,
            aspect="auto",
            origin="lower",
            extent=[amp_min, amp_max, pha_min, pha_max],  # Use common axis ranges
            cmap="viridis",
            vmin=vmin,  # Use common scale
            vmax=vmax   # Use common scale
        )
        
        # Set axis limits explicitly
        ax.set_xlim(amp_min, amp_max)
        ax.set_ylim(pha_min, pha_max)
        
        # Add timing info to title
        title_lines = [name]
        if data.get('setup_time') is not None:
            title_lines.append(f"Setup: {data['setup_time']:.3f}s")
        title_lines.append(f"Computation: {data['time']:.3f}s")
        ax.set_title('\n'.join(title_lines), fontweight="bold")
        
        ax.set_xlabel("Amplitude Frequency (Hz)")
        ax.set_ylabel("Phase Frequency (Hz)")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("PAC Value")
        
        # Mark the ground truth coupling
        ax.plot(
            amp_freq,
            pha_freq,
            "r*",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Ground Truth",
        )
        ax.legend()
        
        col += 1
        if col >= 2 and i < n_methods - 1:
            col = 0
            row += 1

    # Performance comparison with separate bars
    ax_perf = fig.add_subplot(gs[-1, :])
    
    methods = []
    comp_times = []
    setup_times = []
    
    for name, data in results_dict.items():
        if data['pac'] is not None:
            methods.append(name)
            comp_times.append(data['time'])
            setup_times.append(data.get('setup_time', 0))
    
    # Create grouped bar chart
    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.35  # Width of bars
    
    # Define colors for each method
    color_map = {
        'gPAC': ('#87CEEB', '#4682B4'),  # Light/dark blue
        'gPAC (filtfilt)': ('#90EE90', '#228B22'),  # Light/dark green
        'TensorPAC': ('#FFB6C1', '#DC143C')  # Light/dark pink/red
    }
    
    setup_colors = [color_map.get(m, ('#D3D3D3', '#696969'))[0] for m in methods]
    comp_colors = [color_map.get(m, ('#D3D3D3', '#696969'))[1] for m in methods]
    
    # Plot setup and computation times as separate bars
    bars1 = ax_perf.bar(x - width/2, setup_times, width, label='Setup', 
                        color=setup_colors, alpha=0.8, edgecolor="black")
    bars2 = ax_perf.bar(x + width/2, comp_times, width, label='Computation', 
                        color=comp_colors, alpha=0.8, edgecolor="black")
    
    ax_perf.set_ylabel("Time (seconds)")
    ax_perf.set_title("Performance Comparison: Setup vs Computation Time", fontweight="bold")
    ax_perf.set_xlabel("Method")
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(methods)
    ax_perf.legend(loc='upper right')
    ax_perf.grid(True, alpha=0.3, axis="y")
    
    # Add time labels on bars
    for i, (setup, comp) in enumerate(zip(setup_times, comp_times)):
        # Setup time label
        ax_perf.text(i - width/2, setup + 0.002, f"{setup:.3f}s", 
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
        # Computation time label
        ax_perf.text(i + width/2, comp + 0.002, f"{comp:.3f}s", 
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Add speedup information for computation
    if len(comp_times) > 1:
        fastest_comp = min(comp_times)
        fastest_idx = comp_times.index(fastest_comp)
        
        # Calculate and display speedup
        for i, comp_time in enumerate(comp_times):
            if i != fastest_idx:
                speedup = comp_time / fastest_comp
                y_pos = max(max(setup_times), max(comp_times)) * 0.9
                ax_perf.text(
                    i, y_pos,
                    f"{speedup:.1f}x slower\n(computation)",
                    ha="center",
                    va="center",
                    fontsize=11,
                    style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
                )

    plt.tight_layout()
    return fig


def demonstrate_synthetic_data_generator():
    """Demonstrate the SyntheticDataGenerator capabilities."""
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATING SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Create generator with custom parameters
    generator = gpac.SyntheticDataGenerator(
        fs=512.0,
        duration_sec=2.0,
        n_samples=50,  # Small number for demo
        n_channels=2,
        n_segments=2,
        n_classes=3,  # Use 3 classes for demo
        random_seed=42,
    )

    print(f"📊 Generator configured with {generator.params['n_classes']} classes:")
    for class_id, class_info in generator.class_definitions.items():
        print(
            f"  Class {class_id} ({class_info['name']}): "
            f"θ={class_info['pha_range']} Hz, γ={class_info['amp_range']} Hz"
        )

    # Generate dataset
    print("\n🔄 Generating synthetic dataset...")
    start_time = time.time()
    datasets = generator.generate_and_split(train_ratio=0.7, val_ratio=0.2)
    generation_time = time.time() - start_time

    print(f"✅ Dataset generated in {generation_time:.3f} seconds")
    print(f"📈 Train: {len(datasets['train'])} samples")
    print(f"📈 Validation: {len(datasets['val'])} samples")
    print(f"📈 Test: {len(datasets['test'])} samples")

    # Test the generated data with gPAC
    print("\n🧪 Testing generated data with gPAC...")
    sample_signal, sample_label, sample_metadata = datasets["train"][0]

    # Add batch dimension for gPAC
    sample_signal_batch = sample_signal.unsqueeze(0)

    # Calculate PAC
    pac_result, pha_freqs, amp_freqs = gpac.calculate_pac(
        sample_signal_batch,
        fs=generator.params["fs"],
        pha_n_bands=50,
        amp_n_bands=30,
    )

    print(f"✅ PAC calculation successful!")
    print(f"📊 Sample from class {sample_label} ({sample_metadata['class_names']})")
    print(
        f"📊 Expected coupling: θ={sample_metadata['pha_freqs']:.1f} Hz, "
        f"γ={sample_metadata['amp_freqs']:.1f} Hz"
    )
    print(f"📊 PAC result shape: {pac_result.shape}")

    return datasets, generator


def main():
    """Run the complete demo."""
    print("🚀 Starting gPAC Package Demo")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU available, using CPU")

    # 1. Create demo signal
    print("\n📡 Creating synthetic PAC signal...")
    signal, fs, t, pha_freq, amp_freq = create_demo_signal()
    print(f"✅ Signal created: {signal.shape} at {fs} Hz")
    print(f"🎯 Ground truth coupling: θ={pha_freq} Hz → γ={amp_freq} Hz")

    # 2. Calculate PAC with different methods (fair comparison)
    results = {}
    
    # gPAC (now with TensorPAC-compatible defaults)
    pac_gpac, pha_freqs, amp_freqs, comp_time, setup_time = calculate_gpac_pac_fair(signal, fs)
    results['gPAC'] = {
        'pac': pac_gpac,
        'pha_freqs': pha_freqs,
        'amp_freqs': amp_freqs,
        'time': comp_time,
        'setup_time': setup_time
    }
    
    # Original TensorPAC (if available for comparison)
    if TENSORPAC_AVAILABLE:
        pac_tp, pha_freqs_tp, amp_freqs_tp, comp_time_tp, setup_time_tp = calculate_tensorpac_pac_fair(signal, fs)
        if pac_tp is not None:
            results['TensorPAC'] = {
                'pac': pac_tp,
                'pha_freqs': pha_freqs_tp,
                'amp_freqs': amp_freqs_tp,
                'time': comp_time_tp,
                'setup_time': setup_time_tp
            }

    # 3. Performance analysis
    print("\n⚡ PERFORMANCE SUMMARY (Fair Comparison)")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("🚀 GPU Acceleration Active")
    
    print("\nTiming breakdown:")
    for name, data in results.items():
        if data['pac'] is not None:
            print(f"\n{name}:")
            print(f"  Setup time: {data['setup_time']:.3f}s")
            print(f"  Computation time: {data['time']:.3f}s")
            print(f"  Total time: {data['setup_time'] + data['time']:.3f}s")
    
    # Show computation-only speedup if TensorPAC is available
    if TENSORPAC_AVAILABLE and 'TensorPAC' in results:
        speedup = results['TensorPAC']['time'] / results['gPAC']['time']
        print(f"\n📊 gPAC is {speedup:.1f}x faster than TensorPAC (computation only)")
        print("   Note: In real-world usage with large datasets, setup time is amortized")
    
    # 4. Create visualization
    print("\n📊 Creating visualization...")
    fig = create_comprehensive_visualization(results, signal, fs, t, pha_freq, amp_freq)

    # Save the figure
    output_path = Path("readme_demo_output.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"💾 Visualization saved to: {output_path.absolute()}")

    # 5. Demonstrate SyntheticDataGenerator
    datasets, generator = demonstrate_synthetic_data_generator()

    # 6. Final summary
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 PAC calculated for {signal.shape[0]} signal(s)")
    
    print("\n🔑 KEY FINDINGS:")
    print("1. gPAC uses TensorPAC-compatible filter design by default")
    print("2. Fair comparison shows computation speedup after initialization")
    print("3. Filter parameters: 3 cycles for phase, 6 cycles for amplitude")
    print("4. Setup overhead is minimal for real-world batch processing")
    
    print(
        f"\n🎲 Generated {sum(len(ds) for ds in [datasets['train'], datasets['val'], datasets['test']])} "
        f"synthetic samples across {len(generator.class_definitions)} classes"
    )
    print(f"💾 Demo visualization: {output_path.absolute()}")

    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("🖼️  Run in interactive environment to see plots")

    return {
        "signal": signal,
        "results": results,
        "datasets": datasets,
        "generator": generator,
        "figure": fig,
    }


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    results = main()

# EOF