#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 06:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/comparison_with_tensorpac/example_pac_comparison_batch.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/comparison_with_tensorpac/example_pac_comparison_batch.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Batch processing comparison to show gPAC's strength

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gpac import PAC, SyntheticDataGenerator
import mngs

try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False


def benchmark_batch_processing(batch_sizes=[1, 4, 8, 16, 32]):
    """Benchmark PAC computation with different batch sizes."""
    
    # Parameters
    fs = 512
    duration = 5
    n_samples = int(fs * duration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate test signals
    mngs.str.printc("Generating test signals...", "yellow")
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration)
    max_batch = max(batch_sizes)
    signals = []
    for i in range(max_batch):
        signal = generator.generate_pac_signal(
            phase_freq=5 + i * 0.5,  # Vary frequency
            amp_freq=70 + i * 2,
            coupling_strength=0.8,
            noise_level=0.1
        )
        signals.append(signal)
    signals = np.array(signals)
    
    # Initialize PAC analyzers
    mngs.str.printc("Initializing PAC analyzers...", "yellow")
    pac_gpac = PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=15,
        amp_start_hz=50,
        amp_end_hz=100,
        amp_n_bands=15,
        trainable=False
    ).to(device)
    
    if TENSORPAC_AVAILABLE:
        # Get frequency centers from the bandpass filter
        pha_freqs = pac_gpac.PHA_MIDS_HZ.cpu().numpy()
        amp_freqs = pac_gpac.AMP_MIDS_HZ.cpu().numpy()
        pha_bands = [(f-0.5, f+0.5) for f in pha_freqs]
        amp_bands = [(f-2, f+2) for f in amp_freqs]
        
        pac_tp = TensorPAC(
            idpac=(2, 0, 0),
            f_pha=pha_bands,
            f_amp=amp_bands,
            dcomplex='hilbert',
            n_bins=18
        )
    
    # Benchmark different batch sizes
    results = {'batch_sizes': batch_sizes, 'gpac_times': [], 'tensorpac_times': []}
    
    for batch_size in batch_sizes:
        mngs.str.printc(f"\nTesting batch size: {batch_size}", "cyan")
        
        # Prepare batch
        batch_signals = signals[:batch_size]
        
        # Time gPAC
        signals_torch = torch.from_numpy(batch_signals).float().unsqueeze(1).to(device)
        
        # Warm-up
        with torch.no_grad():
            _ = pac_gpac(signals_torch)
        torch.cuda.synchronize() if device == 'cuda' else None
        
        # Actual timing
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            output = pac_gpac(signals_torch)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        gpac_time = time.time() - start
        results['gpac_times'].append(gpac_time)
        
        mngs.str.printc(f"  gPAC: {gpac_time:.4f}s ({gpac_time/batch_size:.4f}s per signal)", "green")
        
        # Time TensorPAC
        if TENSORPAC_AVAILABLE:
            start = time.time()
            
            # TensorPAC processes one at a time
            for i in range(batch_size):
                _ = pac_tp.filterfit(fs, batch_signals[i:i+1], n_jobs=1)
            
            tp_time = time.time() - start
            results['tensorpac_times'].append(tp_time)
            
            mngs.str.printc(f"  TensorPAC: {tp_time:.4f}s ({tp_time/batch_size:.4f}s per signal)", "green")
    
    return results


def main():
    """Run batch processing comparison."""
    mngs.str.printc("="*80, "blue")
    mngs.str.printc("gPAC vs TensorPAC: Batch Processing Comparison", "blue")
    mngs.str.printc("="*80, "blue")
    
    # Run benchmarks
    results = benchmark_batch_processing()
    
    # Create visualization
    mngs.str.printc("\nCreating batch processing visualization...", "yellow")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Total time comparison
    ax1.plot(results['batch_sizes'], results['gpac_times'], 'b-o', 
             linewidth=2, markersize=8, label='gPAC (GPU)')
    if results['tensorpac_times']:
        ax1.plot(results['batch_sizes'], results['tensorpac_times'], 'r-s', 
                 linewidth=2, markersize=8, label='TensorPAC (CPU)')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Total Time (seconds)', fontsize=12)
    ax1.set_title('Total Processing Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Time per signal comparison
    gpac_per_signal = [t/b for t, b in zip(results['gpac_times'], results['batch_sizes'])]
    ax2.plot(results['batch_sizes'], gpac_per_signal, 'b-o', 
             linewidth=2, markersize=8, label='gPAC (GPU)')
    if results['tensorpac_times']:
        tp_per_signal = [t/b for t, b in zip(results['tensorpac_times'], results['batch_sizes'])]
        ax2.plot(results['batch_sizes'], tp_per_signal, 'r-s', 
                 linewidth=2, markersize=8, label='TensorPAC (CPU)')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Time per Signal (seconds)', fontsize=12)
    ax2.set_title('Amortized Processing Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Batch Processing Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    mngs.io.save(fig, "pac_batch_comparison.png", dpi=150)
    mngs.str.printc("\n✓ Saved batch comparison figure", "green")
    
    mngs.io.save(results, "pac_batch_results.yaml")
    mngs.str.printc("✓ Saved batch results", "green")
    
    # Print summary
    mngs.str.printc("\n" + "="*80, "magenta")
    mngs.str.printc("BATCH PROCESSING SUMMARY", "magenta")
    mngs.str.printc("="*80, "magenta")
    
    if len(results['tensorpac_times']) > 0:
        # Find crossover point
        speedups = [tp/gp for tp, gp in zip(results['tensorpac_times'], results['gpac_times'])]
        
        mngs.str.printc("\nSpeedup factors (gPAC vs TensorPAC):", "white")
        for bs, speedup in zip(results['batch_sizes'], speedups):
            if speedup > 1:
                mngs.str.printc(f"  Batch size {bs}: {speedup:.2f}x faster", "green")
            else:
                mngs.str.printc(f"  Batch size {bs}: {1/speedup:.2f}x slower", "yellow")
        
        # Check if there's a crossover
        if any(s > 1 for s in speedups):
            crossover_idx = next(i for i, s in enumerate(speedups) if s > 1)
            mngs.str.printc(f"\n✓ gPAC becomes faster at batch size ≥ {results['batch_sizes'][crossover_idx]}", "green")
    
    mngs.str.printc("\n✓ Batch processing analysis completed!", "green")


if __name__ == "__main__":
    main()

# EOF