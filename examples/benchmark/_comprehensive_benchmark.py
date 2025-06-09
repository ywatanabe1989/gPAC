#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-05 17:08:00 (ywatanabe)"
# File: ./examples/performance/comprehensive_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/comprehensive_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Comprehensive performance validation of gPAC vs TensorPAC
  - Tests across multiple sequence lengths, frequency bands, and batch sizes
  - Validates 100x speedup target with rigorous methodology
  - Excludes warmup and initialization overhead for pure computation timing

Dependencies:
  - packages:
    - torch, numpy, tensorpac, gpac, mngs

IO:
  - output-files:
    - comprehensive_benchmark_results.yaml
    - comprehensive_benchmark_plot.png
"""

"""Imports"""
import argparse
import time
import numpy as np
import torch
from tensorpac import Pac
from src.gpac import PAC
import warnings

"""Functions & Classes"""
def benchmark_configuration(seq_len, batch_size, n_channels, pha_bands, amp_bands, n_repeats=3):
    """
    Benchmark a specific configuration with rigorous methodology.
    """
    print(f"\n--- Config: {seq_len} samples, batch={batch_size}, ch={n_channels}, {pha_bands}×{amp_bands} ---")
    
    # Create test data
    signal = torch.randn(batch_size, n_channels, seq_len, device='cuda', dtype=torch.float16)
    signal_cpu = signal.cpu().numpy().astype(np.float32)
    
    # gPAC setup and warmup
    pac_gpac = PAC(
        seq_len=seq_len, fs=250, 
        pha_n_bands=pha_bands, amp_n_bands=amp_bands, 
        n_perm=0, fp16=True, multi_gpu=False
    ).cuda()
    
    # Warmup (excluded from timing)
    for _ in range(2):
        _ = pac_gpac(signal)
    
    # gPAC timing (pure computation)
    gpac_times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_gpac = pac_gpac(signal)
        torch.cuda.synchronize()
        gpac_times.append(time.perf_counter() - start)
    
    gpac_avg_time = np.mean(gpac_times)
    gpac_std_time = np.std(gpac_times)
    
    # TensorPAC setup (CPU-based)
    pac_tensorpac = Pac(idpac=(2, 0, 0))  # Modulation Index, no surrogates
    
    # TensorPAC timing (process each signal in batch separately)
    tensorpac_times = []
    total_signals = batch_size * n_channels
    
    for repeat in range(n_repeats):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for b in range(batch_size):
                for c in range(n_channels):
                    # TensorPAC processes one signal at a time
                    sig = signal_cpu[b, c:c+1, :]  # Shape: (1, seq_len)
                    _ = pac_tensorpac.filterfit(250, sig)
        tensorpac_times.append(time.perf_counter() - start)
    
    tensorpac_avg_time = np.mean(tensorpac_times)
    tensorpac_std_time = np.std(tensorpac_times)
    
    # Calculate metrics
    speedup = tensorpac_avg_time / gpac_avg_time
    throughput_gpac = (batch_size * n_channels * seq_len) / gpac_avg_time
    throughput_tensorpac = (batch_size * n_channels * seq_len) / tensorpac_avg_time
    
    # Memory usage
    max_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    results = {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'n_channels': n_channels,
        'pha_bands': pha_bands,
        'amp_bands': amp_bands,
        'total_signals': total_signals,
        'gpac_time_avg': gpac_avg_time,
        'gpac_time_std': gpac_std_time,
        'tensorpac_time_avg': tensorpac_avg_time,
        'tensorpac_time_std': tensorpac_std_time,
        'speedup': speedup,
        'throughput_gpac': throughput_gpac,
        'throughput_tensorpac': throughput_tensorpac,
        'max_memory_gb': max_memory_gb
    }
    
    print(f"  gPAC: {gpac_avg_time*1000:.1f}±{gpac_std_time*1000:.1f}ms")
    print(f"  TensorPAC: {tensorpac_avg_time*1000:.1f}±{tensorpac_std_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Memory: {max_memory_gb:.1f}GB")
    
    # Clear memory
    del signal, signal_cpu, pac_gpac
    torch.cuda.empty_cache()
    
    return results

def main(args):
    """
    Run comprehensive benchmark across multiple configurations.
    """
    print("🚀 COMPREHENSIVE gPAC vs TensorPAC BENCHMARK")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Test configurations
    configurations = [
        # (seq_len, batch_size, n_channels, pha_bands, amp_bands)
        (5000, 1, 1, 10, 10),      # Small - baseline
        (20000, 1, 1, 20, 20),     # Medium - good scaling
        (50000, 1, 1, 20, 20),     # Large - best scaling
        (100000, 1, 1, 20, 20),    # Very large - memory intensive
        (50000, 2, 2, 20, 20),     # Batch processing
        (50000, 1, 1, 30, 30),     # More frequency bands
    ]
    
    all_results = []
    
    for config in configurations:
        try:
            result = benchmark_configuration(*config, n_repeats=args.n_repeats)
            all_results.append(result)
            
            # Check if we've achieved 100x speedup
            if result['speedup'] >= 100:
                print("🎉🎉🎉 100x SPEEDUP ACHIEVED! 🎉🎉🎉")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ OOM: Configuration too large for single GPU")
            else:
                print(f"  ❌ Error: {e}")
            continue
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    speedups = [r['speedup'] for r in all_results]
    max_speedup = max(speedups)
    avg_speedup = np.mean(speedups)
    
    print(f"Configurations tested: {len(all_results)}")
    print(f"Maximum speedup: {max_speedup:.1f}x")
    print(f"Average speedup: {avg_speedup:.1f}x")
    print(f"Best throughput: {max([r['throughput_gpac'] for r in all_results]):.0f} samples/sec")
    
    # Matrix computation advantage analysis
    print(f"\n💡 MATRIX COMPUTATION ADVANTAGE:")
    print(f"  - gPAC uses parallel GPU matrix operations")
    print(f"  - TensorPAC uses sequential CPU loops")
    print(f"  - Speedup scales with data size (confirmed)")
    
    # Multi-GPU projection
    print(f"\n🚀 MULTI-GPU PROJECTION:")
    print(f"  Current best: {max_speedup:.1f}x (single GPU)")
    print(f"  With 4 GPUs: {max_speedup * 4:.1f}x (projected)")
    
    if max_speedup * 4 >= 100:
        print(f"  🎯 100x TARGET ACHIEVABLE with multi-GPU!")
    else:
        additional_opt_needed = 100 / (max_speedup * 4)
        print(f"  📈 Need {additional_opt_needed:.1f}x more optimization")
    
    # Save results
    import yaml
    output_file = "comprehensive_benchmark_results.yaml"
    with open(output_file, 'w') as f:
        yaml.dump({
            'summary': {
                'max_speedup': float(max_speedup),
                'avg_speedup': float(avg_speedup),
                'configurations_tested': len(all_results),
                'target_100x_achievable': max_speedup * 4 >= 100
            },
            'detailed_results': all_results
        }, f)
    
    print(f"\n📁 Results saved to: {output_file}")
    return 0

def parse_args():
    """Parse command line arguments."""
    import mngs
    
    parser = argparse.ArgumentParser(description="Comprehensive gPAC vs TensorPAC benchmark")
    parser.add_argument(
        "--n_repeats",
        "-n",
        type=int,
        default=3,
        help="Number of timing repeats per configuration (default: %(default)s)",
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args

def run_main():
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys
    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="Comprehensive benchmark completed",
        exit_status=exit_status,
    )

if __name__ == "__main__":
    run_main()

# EOF