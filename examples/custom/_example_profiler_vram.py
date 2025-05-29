#!/usr/bin/env python3
"""Example demonstrating improved VRAM tracking with gPAC operations."""

import torch
import numpy as np
from gpac import PAC, generate_pac_signal
from gpac._Profiler import Profiler

def main():
    """Demonstrate VRAM tracking with PAC computation."""
    
    # Create profiler
    profiler = Profiler(enable_gpu=True)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("-" * 60)
    
    # Generate synthetic PAC signal
    with profiler.profile("Signal Generation"):
        n_seconds = 10
        sf = 1000  # Sampling frequency
        signal, time = generate_pac_signal(
            n_seconds=n_seconds,
            sf=sf,
            f_phase=10,
            f_amp=60,
            device=device
        )
    
    # Create PAC analyzer
    with profiler.profile("PAC Initialization"):
        pac = PAC(
            f_phase=(8, 12),
            f_amp=(50, 70),
            seq_len=signal.shape[-1],
            sf=sf,
            device=device
        )
    
    # Compute PAC values
    with profiler.profile("PAC Computation"):
        pac_values = pac(signal)
        print(f"PAC value: {pac_values.item():.4f}")
    
    # Batch processing example
    with profiler.profile("Batch Processing"):
        batch_size = 32
        batch_signal = signal.repeat(batch_size, 1, 1)
        batch_pac_values = pac(batch_signal)
        print(f"Batch PAC mean: {batch_pac_values.mean().item():.4f}")
    
    # Large batch to show memory usage
    with profiler.profile("Large Batch Processing"):
        large_batch_size = 128
        large_batch_signal = signal.repeat(large_batch_size, 1, 1)
        large_batch_pac_values = pac(large_batch_signal)
        print(f"Large batch PAC mean: {large_batch_pac_values.mean().item():.4f}")
    
    # Cleanup
    del signal, batch_signal, large_batch_signal
    
    with profiler.profile("Post-Cleanup State"):
        torch.cuda.empty_cache()
        # Small operation to measure final state
        test_tensor = torch.randn(100, 100, device=device)
    
    # Print profiling summary
    print("\n" + "="*60)
    profiler.print_summary()
    
    # Show memory insights
    print("\n💡 Memory Insights:")
    summary = profiler.get_summary_dict()
    if 'peak_vram_allocated_gb' in summary and 'peak_vram_reserved_gb' in summary:
        efficiency = (summary['peak_vram_allocated_gb'] / summary['peak_vram_reserved_gb']) * 100
        print(f"Memory efficiency: {efficiency:.1f}%")
        print(f"PyTorch cache overhead: {summary['peak_vram_reserved_gb'] - summary['peak_vram_allocated_gb']:.3f} GB")

if __name__ == "__main__":
    main()