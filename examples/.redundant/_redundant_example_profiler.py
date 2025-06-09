#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-01-29 01:15:00 (ywatanabe)"
# File: ./examples/gpac/example_profiler.py

"""
Functionalities:
  - Demonstrates the Profiler class for performance monitoring
  - Profiles different stages of PAC computation
  - Shows GPU vs CPU performance differences
  - Visualizes profiling results and timing breakdowns
  - Demonstrates memory usage tracking

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
    - None (generates synthetic signals)
    
  - output-files:
    - ./scripts/example_profiler/profiler_demo.gif
    - ./scripts/example_profiler/profiling_results.csv
    - ./scripts/example_profiler/timing_breakdown.gif
"""

"""Imports"""
import argparse
import sys
import numpy as np
import torch
import time

"""Warnings"""
import warnings
warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# None

"""Functions & Classes"""

def generate_test_signals(batch_size, n_channels, n_segments, seq_len, device):
    """Generate test signals for profiling."""
    # Create synthetic PAC signal with known properties
    fs = 512.0
    duration = seq_len / fs
    t = np.linspace(0, duration, seq_len)
    
    signals = []
    for _ in range(batch_size):
        batch_signals = []
        for _ in range(n_channels):
            channel_signals = []
            for _ in range(n_segments):
                # Phase component (theta, 6 Hz)
                phase_signal = np.sin(2 * np.pi * 6 * t)
                
                # Amplitude component (gamma, 80 Hz) modulated by phase
                modulation = 1 + 0.7 * np.sin(2 * np.pi * 6 * t)
                amp_signal = modulation * np.sin(2 * np.pi * 80 * t)
                
                # Combine with noise
                signal = phase_signal + 0.3 * amp_signal + 0.1 * np.random.randn(seq_len)
                channel_signals.append(signal)
            batch_signals.append(channel_signals)
        signals.append(batch_signals)
    
    # Convert to tensor
    signals_np = np.array(signals)  # (batch, channels, segments, time)
    signals_torch = torch.from_numpy(signals_np).float().to(device)
    
    return signals_torch, fs


def main(args):
    """Demonstrate profiler usage with PAC computation."""
    import mngs
    from gpac import PAC, generate_pac_signal
    from gpac._Profiler import create_profiler
    
    mngs.str.printc("🚀 gPAC Profiler Demonstration", c="green")
    mngs.str.printc("="*60, c="green")
    
    # Create profiler
    profiler = create_profiler(enable_gpu=True)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mngs.str.printc(f"📍 Using device: {device}", c="cyan")
    if device == 'cuda':
        mngs.str.printc(f"GPU: {torch.cuda.get_device_name()}", c="cyan")
    
    # Parameters
    batch_size = 10
    n_channels = 8
    n_segments = 4
    seq_len = 1024
    fs = 512.0
    
    # Profile data generation
    mngs.str.printc("\n📡 Profiling data generation...", c="blue")
    with profiler.profile("Data Generation"):
        signals, fs = generate_test_signals(
            batch_size, n_channels, n_segments, seq_len, device
        )
    
    mngs.str.printc(f"Generated signals shape: {signals.shape}", c="cyan")
    
    # Profile model initialization
    mngs.str.printc("\n🔧 Profiling model initialization...", c="blue")
    with profiler.profile("Model Initialization"):
        pac = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=10,
            amp_start_hz=30.0,
            amp_end_hz=100.0,
            amp_n_bands=10,
            n_perm=None,
            trainable=False
        ).to(device)
    
    # Warmup (not profiled)
    mngs.str.printc("\n🔥 Warming up...", c="yellow")
    with torch.no_grad():
        _ = pac(signals[:1])
    
    # Profile single forward pass
    mngs.str.printc("\n⏱️  Profiling single forward pass...", c="blue")
    with profiler.profile("Single Forward Pass"):
        torch.cuda.synchronize() if device == 'cuda' else None
        with torch.no_grad():
            pac_values_single = pac(signals[:1])
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # Profile batch forward pass
    mngs.str.printc("⏱️  Profiling batch forward pass...", c="blue")
    with profiler.profile(f"Batch Forward Pass ({batch_size} samples)"):
        torch.cuda.synchronize() if device == 'cuda' else None
        with torch.no_grad():
            pac_values_batch = pac(signals)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # Profile with permutation testing
    mngs.str.printc("\n⏱️  Profiling with permutation testing...", c="blue")
    pac_perm = PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=10,
        amp_start_hz=30.0,
        amp_end_hz=100.0,
        amp_n_bands=10,
        n_perm=100,  # 100 permutations
        trainable=False
    ).to(device)
    
    with profiler.profile("Forward Pass with Permutations"):
        torch.cuda.synchronize() if device == 'cuda' else None
        with torch.no_grad():
            pac_values_perm = pac_perm(signals[:2])  # Smaller batch for permutations
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # Profile backward pass (if trainable)
    if device == 'cuda':
        mngs.str.printc("\n⏱️  Profiling backward pass...", c="blue")
        pac_trainable = PAC(
            seq_len=seq_len,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=10,
            amp_start_hz=30.0,
            amp_end_hz=100.0,
            amp_n_bands=10,
            n_perm=None,
            trainable=True  # Enable gradients
        ).to(device)
        
        # Forward pass
        pac_values = pac_trainable(signals[:2])
        loss = pac_values['pac'].mean()
        
        with profiler.profile("Backward Pass"):
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
    
    # Profile visualization
    mngs.str.printc("\n📊 Profiling visualization...", c="blue")
    with profiler.profile("Visualization"):
        fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot signal snippet
        ax = axes[0, 0]
        signal_plot = signals[0, 0, 0, :256].cpu().numpy()
        ax.plot(signal_plot, 'b-', linewidth=0.8)
        ax.set_xyt('Time samples', 'Amplitude', 'Signal Snippet')
        ax.grid(True, alpha=0.3)
        
        # Plot PAC matrix
        ax = axes[0, 1]
        pac_matrix = pac_values_batch['pac'][0, 0].cpu().numpy()
        im = ax.imshow(pac_matrix, aspect='auto', origin='lower', cmap='hot')
        ax.set_xyt('Amplitude Band', 'Phase Band', 'PAC Matrix')
        plt.colorbar(im, ax=ax)
        
        # Plot timing breakdown
        ax = axes[1, 0]
        timings = profiler.get_summary_dict()
        operations = list(timings.keys())
        if 'total_time' in operations:
            operations.remove('total_time')
        times = [timings[op] for op in operations]
        
        bars = ax.barh(operations, times)
        ax.set_xlabel('Time (s)')
        ax.set_title('Timing Breakdown')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Color code by operation type
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)
        
        # Plot relative timings
        ax = axes[1, 1]
        if timings.get('total_time', 0) > 0:
            percentages = [100 * t / timings['total_time'] for t in times]
            ax.pie(percentages, labels=operations, autopct='%1.1f%%', 
                   colors=colors[:len(operations)])
            ax.set_title('Relative Time Distribution')
        
        plt.tight_layout()
        spath = './scripts/example_profiler/profiler_demo.gif'
        mngs.io.save(fig, spath)
    
    # Print profiling summary
    mngs.str.printc("\n" + "="*60, c="green")
    profiler.print_summary()
    
    # Save timing results
    timing_results = profiler.get_summary_dict()
    mngs.io.save(
        timing_results,
        './scripts/example_profiler/profiling_results.csv'
    )
    
    # Additional performance metrics
    mngs.str.printc("\n📊 Performance Metrics", c="yellow")
    mngs.str.printc("="*60, c="yellow")
    
    single_time = timing_results.get('Single Forward Pass', 0)
    batch_time = timing_results.get(f'Batch Forward Pass ({batch_size} samples)', 0)
    
    if single_time > 0:
        mngs.str.printc(f"Single sample processing: {single_time*1000:.2f} ms", c="cyan")
        mngs.str.printc(f"Throughput: {1/single_time:.1f} samples/second", c="cyan")
    
    if batch_time > 0:
        mngs.str.printc(f"\nBatch processing ({batch_size} samples): {batch_time*1000:.2f} ms", c="cyan")
        mngs.str.printc(f"Per-sample time in batch: {batch_time/batch_size*1000:.2f} ms", c="cyan")
        mngs.str.printc(f"Batch efficiency: {single_time*batch_size/batch_time:.1f}x", c="cyan")
        mngs.str.printc(f"Batch throughput: {batch_size/batch_time:.1f} samples/second", c="cyan")
    
    # Memory usage (if GPU)
    if device == 'cuda':
        mngs.str.printc("\n💾 GPU Memory Usage", c="yellow")
        mngs.str.printc("="*60, c="yellow")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        mngs.str.printc(f"Allocated: {allocated:.1f} MB", c="cyan")
        mngs.str.printc(f"Reserved: {reserved:.1f} MB", c="cyan")
    
    # Create detailed timing breakdown figure
    fig, ax = mngs.plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort operations by time
    sorted_ops = sorted([(op, t) for op, t in timing_results.items() if op != 'total_time'], 
                       key=lambda x: x[1], reverse=True)
    
    if sorted_ops:
        ops, times = zip(*sorted_ops)
        y_pos = np.arange(len(ops))
        
        bars = ax.barh(y_pos, times)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ops)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Detailed Timing Breakdown')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{time:.3f}s', va='center')
    
    spath = './scripts/example_profiler/timing_breakdown.gif'
    mngs.io.save(fig, spath)
    
    mngs.str.printc("\n✅ Profiler demonstration complete!", c="green")
    mngs.str.printc(f"💾 Results saved to: ./scripts/example_profiler/", c="green")
    
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs
    
    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Profiler demonstration")
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