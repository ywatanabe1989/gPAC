#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-05 16:08:00 (ywatanabe)"
# File: ./examples/performance/ultra_speed_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/performance/ultra_speed_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests PURE COMPUTATION SPEED targeting 100x speedup vs TensorPAC
  - Excludes initialization and GPU transfer time
  - Separates initialization vs computation timing
  - Assumes data is already on GPU (realistic dataloader scenario)

Dependencies:
  - scripts: None
  - packages: gpac, tensorpac, torch, numpy, matplotlib, mngs

IO:
  - input-files: None
  - output-files: ultra_speed_results.yaml, ultra_speed_comparison.gif
"""

"""Imports"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from tensorpac import Pac as TensorPAC

sys.path.append("../../src")
from gpac import PAC

"""Functions & Classes"""
def create_test_data(batch_size=8, n_channels=16, seq_len=2048, device='cuda'):
    """Create test data already on GPU"""
    print(f"Creating test data: batch={batch_size}, channels={n_channels}, samples={seq_len}")
    
    # Create data directly on GPU to exclude transfer time
    data = torch.randn(batch_size, n_channels, seq_len, device=device, dtype=torch.float32)
    
    # For TensorPAC, create equivalent numpy data on CPU
    data_np = torch.randn(batch_size, n_channels, seq_len, dtype=torch.float32).numpy()
    
    return data, data_np

def benchmark_gpac_pure_computation(data, pac_model, n_warmup=5, n_trials=20):
    """Benchmark PURE computation time (exclude initialization)"""
    print("🔥 gPAC Ultra-Fast Mode: Pure computation benchmark")
    
    # Warmup runs to stabilize GPU
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = pac_model(data)
        torch.cuda.synchronize()
    
    # Actual timing - PURE COMPUTATION only
    times = []
    for i in range(n_trials):
        torch.cuda.synchronize()  # Ensure clean start
        start_time = time.time()
        
        with torch.no_grad():
            result = pac_model(data)
        
        torch.cuda.synchronize()  # Ensure completion
        comp_time = time.time() - start_time
        times.append(comp_time)
        
        if i % 5 == 0:
            print(f"  Trial {i+1}/{n_trials}: {comp_time:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, result

def benchmark_tensorpac_pure_computation(data_np, fs, pha_n_bands, amp_n_bands, n_warmup=3, n_trials=10):
    """Benchmark TensorPAC computation (exclude initialization)"""
    print("🐢 TensorPAC: Pure computation benchmark")
    
    # Initialize TensorPAC (this is initialization, not counted)
    pha_edges = np.linspace(2, 20, pha_n_bands + 1)
    amp_edges = np.linspace(30, 100, amp_n_bands + 1)
    f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]
    f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    pac = TensorPAC(
        idpac=(2, 0, 0),
        f_pha=f_pha,
        f_amp=f_amp,
        dcomplex="hilbert",
        verbose=False,
    )
    
    # Reshape data for TensorPAC
    batch_size, n_chs, n_samples = data_np.shape
    data_reshaped = data_np.reshape(-1, n_samples)
    
    # Warmup
    for _ in range(n_warmup):
        _ = pac.filterfit(fs, data_reshaped, n_perm=0, verbose=False)
    
    # Actual timing - PURE COMPUTATION only
    times = []
    for i in range(n_trials):
        start_time = time.time()
        result = pac.filterfit(fs, data_reshaped, n_perm=0, verbose=False)
        comp_time = time.time() - start_time
        times.append(comp_time)
        
        if i % 3 == 0:
            print(f"  Trial {i+1}/{n_trials}: {comp_time:.4f}s")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, result

def run_ultra_speed_benchmark():
    """Run the ultra-speed benchmark targeting 100x speedup"""
    print("🚀 ULTRA-SPEED BENCHMARK: Targeting 100x Speedup vs TensorPAC")
    print("=" * 80)
    print("⚡ Excludes initialization and GPU transfer time")
    print("⚡ Tests PURE COMPUTATION SPEED only")
    print("=" * 80)
    
    # Test configurations
    configs = [
        {"batch_size": 4, "n_channels": 8, "seq_len": 1024, "name": "Small"},
        {"batch_size": 8, "n_channels": 16, "seq_len": 2048, "name": "Medium"},
        {"batch_size": 16, "n_channels": 32, "seq_len": 4096, "name": "Large"},
    ]
    
    # PAC parameters
    fs = 250
    pha_n_bands = 10
    amp_n_bands = 15
    
    results = {}
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']} configuration...")
        print(f"   {config['batch_size']} batches × {config['n_channels']} channels × {config['seq_len']} samples")
        
        # Create test data
        data_gpu, data_np = create_test_data(
            config['batch_size'], config['n_channels'], config['seq_len']
        )
        
        # Initialize models (NOT counted in timing)
        print("🔧 Initializing models (NOT counted in timing)...")
        
        # gPAC initialization
        init_start = time.time()
        pac_model = PAC(
            seq_len=config['seq_len'],
            fs=fs,
            pha_start_hz=2, pha_end_hz=20, pha_n_bands=pha_n_bands,
            amp_start_hz=30, amp_end_hz=100, amp_n_bands=amp_n_bands,
            n_perm=None,  # No permutations for ultra-speed test
        ).cuda()
        init_time = time.time() - init_start
        print(f"   gPAC initialization: {init_time:.4f}s")
        
        # Benchmark PURE COMPUTATION
        print("\n📊 PURE COMPUTATION BENCHMARKS:")
        
        # gPAC ultra-fast
        gpac_time, gpac_std, gpac_result = benchmark_gpac_pure_computation(
            data_gpu, pac_model, n_warmup=5, n_trials=20
        )
        
        # TensorPAC
        tp_time, tp_std, tp_result = benchmark_tensorpac_pure_computation(
            data_np, fs, pha_n_bands, amp_n_bands, n_warmup=3, n_trials=10
        )
        
        # Calculate speedup
        speedup = tp_time / gpac_time
        
        # Calculate throughput
        total_samples = config['batch_size'] * config['n_channels'] * config['seq_len']
        gpac_throughput = total_samples / gpac_time / 1e6  # Million samples/sec
        tp_throughput = total_samples / tp_time / 1e6
        
        # Store results
        results[config['name']] = {
            'config': config,
            'gpac_time': gpac_time,
            'gpac_std': gpac_std,
            'tensorpac_time': tp_time,
            'tensorpac_std': tp_std,
            'speedup': speedup,
            'gpac_throughput': gpac_throughput,
            'tp_throughput': tp_throughput,
            'init_time': init_time,
            'total_samples': total_samples
        }
        
        # Print results
        print(f"\n🏆 {config['name']} Results:")
        print(f"   gPAC:      {gpac_time:.4f}s ± {gpac_std:.4f}s ({gpac_throughput:.1f} MSamples/s)")
        print(f"   TensorPAC: {tp_time:.4f}s ± {tp_std:.4f}s ({tp_throughput:.1f} MSamples/s)")
        print(f"   SPEEDUP:   {speedup:.1f}x {'🚀' if speedup > 10 else '📈' if speedup > 1 else '🐌'}")
        
        if speedup >= 100:
            print("   🎯 TARGET ACHIEVED: 100x+ speedup!")
        elif speedup >= 50:
            print("   🔥 EXCELLENT: 50x+ speedup!")
        elif speedup >= 10:
            print("   ✅ GOOD: 10x+ speedup")
        elif speedup >= 2:
            print("   📊 MODERATE: 2x+ speedup")
        else:
            print("   ⚠️  NEEDS OPTIMIZATION")
    
    return results

def create_visualization(results):
    """Create performance visualization"""
    configs = list(results.keys())
    speedups = [results[c]['speedup'] for c in configs]
    gpac_times = [results[c]['gpac_time'] for c in configs]
    tp_times = [results[c]['tensorpac_time'] for c in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Speedup plot
    bars = ax1.bar(configs, speedups, color=['red' if s >= 100 else 'orange' if s >= 10 else 'blue' for s in speedups])
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100x Target')
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10x Good')
    ax1.set_ylabel('Speedup (x)')
    ax1.set_title('gPAC vs TensorPAC Speedup\n(Pure Computation, No Init/Transfer)')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Add speedup values on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    # Computation time comparison
    x = np.arange(len(configs))
    width = 0.35
    ax2.bar(x - width/2, gpac_times, width, label='gPAC', alpha=0.8)
    ax2.bar(x + width/2, tp_times, width, label='TensorPAC', alpha=0.8)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Pure Computation Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig

def main(args):
    """Main function."""
    print("🎯 Target: 100x faster than TensorPAC")
    print("📏 Metric: Pure computation time (excluding init/transfer)")
    print()
    
    # Run benchmark
    results = run_ultra_speed_benchmark()
    
    # Create visualization
    fig = create_visualization(results)
    
    # Save results
    mngs.io.save(results, "ultra_speed_results.yaml")
    mngs.io.save(fig, "ultra_speed_comparison.gif")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🏁 ULTRA-SPEED BENCHMARK SUMMARY")
    print("=" * 80)
    
    max_speedup = max(results[c]['speedup'] for c in results)
    if max_speedup >= 100:
        print("🎯 SUCCESS: Achieved 100x+ speedup target!")
    elif max_speedup >= 50:
        print("🔥 EXCELLENT: Achieved 50x+ speedup!")
    elif max_speedup >= 10:
        print("✅ GOOD: Achieved 10x+ speedup")
    else:
        print("⚠️  NEEDS MORE OPTIMIZATION")
    
    print(f"\nBest performance: {max_speedup:.1f}x speedup")
    print("Results saved to ultra_speed_results.yaml")
    
    return 0

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="Ultra-speed benchmark targeting 100x speedup")
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args

def run_main() -> None:
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
        message="",
        exit_status=exit_status,
    )

if __name__ == "__main__":
    run_main()

# EOF