#!/usr/bin/env python3
"""
Fair performance benchmark comparing only computation time (after initialization).

This script separates initialization from computation to get accurate performance metrics:
1. Initialize both gPAC and Tensorpac models once
2. Time only the forward pass/computation for multiple runs
3. Calculate mean and std of computation times
"""

import time
import warnings
import torch
import numpy as np

import gpac

# Try to import tensorpac
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False

def create_test_signal(fs=512, duration=2.0):
    """Create consistent test signal."""
    t = np.linspace(0, duration, int(fs * duration))
    pha_freq, amp_freq = 6.0, 80.0
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    amplitude_mod = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
    carrier = np.sin(2 * np.pi * amp_freq * t)
    pac_signal = phase_signal + amplitude_mod * carrier * 0.5
    noise = np.random.normal(0, 0.1, len(t))
    signal = pac_signal + noise
    return signal.reshape(1, 1, 1, -1), fs

def benchmark_gpac_computation(signal, fs, pha_bands, amp_bands, n_runs=10):
    """Benchmark gPAC computation time only (after initialization)."""
    print(f"🔄 Benchmarking gPAC ({pha_bands}×{amp_bands}) - {n_runs} runs")
    
    # INITIALIZATION (timed separately)
    print("  Initializing gPAC model...", end=" ")
    init_start = time.time()
    
    # Create model
    model_cpu = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_n_bands=pha_bands,
        amp_n_bands=amp_bands,
        n_perm=None,
        trainable=False
    )
    
    if torch.cuda.is_available():
        model_gpu = gpac.PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_n_bands=pha_bands,
            amp_n_bands=amp_bands,
            n_perm=None,
            trainable=False
        ).cuda()
        signal_gpu = torch.tensor(signal, dtype=torch.float32).cuda()
    else:
        model_gpu = None
        signal_gpu = None
    
    signal_cpu = torch.tensor(signal, dtype=torch.float32)
    init_time = time.time() - init_start
    print(f"{init_time:.3f}s")
    
    # COMPUTATION BENCHMARKING
    # Warm up runs
    _ = model_cpu(signal_cpu)
    if model_gpu is not None:
        _ = model_gpu(signal_gpu)
        torch.cuda.synchronize()  # Ensure GPU operations complete
    
    # CPU timing
    print("  CPU computation...", end=" ")
    cpu_times = []
    for _ in range(n_runs):
        start_time = time.time()
        _ = model_cpu(signal_cpu)
        cpu_times.append(time.time() - start_time)
    
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"{cpu_mean:.4f}±{cpu_std:.4f}s")
    
    # GPU timing
    if model_gpu is not None:
        print("  GPU computation...", end=" ")
        gpu_times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()  # Ensure previous operations complete
            start_time = time.time()
            _ = model_gpu(signal_gpu)
            torch.cuda.synchronize()  # Ensure computation completes
            gpu_times.append(time.time() - start_time)
        
        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)
        print(f"{gpu_mean:.4f}±{gpu_std:.4f}s")
        
        speedup = cpu_mean / gpu_mean
        print(f"  GPU speedup: {speedup:.1f}x")
    else:
        gpu_mean = None
        gpu_std = None
        speedup = None
    
    return {
        'init_time': init_time,
        'cpu_mean': cpu_mean,
        'cpu_std': cpu_std,
        'gpu_mean': gpu_mean,
        'gpu_std': gpu_std,
        'speedup': speedup
    }

def benchmark_tensorpac_computation(signal, fs, pha_bands, amp_bands, n_runs=10):
    """Benchmark Tensorpac computation time only (after initialization)."""
    if not TENSORPAC_AVAILABLE:
        return None
    
    print(f"🔄 Benchmarking Tensorpac ({pha_bands}×{amp_bands}) - {n_runs} runs")
    
    # INITIALIZATION (timed separately)
    print("  Initializing Tensorpac model...", end=" ")
    init_start = time.time()
    
    signal_tp = signal[0, 0, 0, :].reshape(-1, 1)
    f_pha = np.linspace(2, 20, pha_bands)
    f_amp = np.linspace(60, 120, amp_bands)
    
    pac_tp = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='wavelet')
    pac_tp.idpac = (2, 0, 0)
    
    init_time = time.time() - init_start
    print(f"{init_time:.3f}s")
    
    # COMPUTATION BENCHMARKING  
    # Warm up run
    phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
    amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
    _ = pac_tp.fit(phases, amplitudes)
    
    print("  Computation...", end=" ")
    comp_times = []
    for _ in range(n_runs):
        start_time = time.time()
        # Only time the fit operation (filtering is initialization-like)
        _ = pac_tp.fit(phases, amplitudes)
        comp_times.append(time.time() - start_time)
    
    comp_mean = np.mean(comp_times)
    comp_std = np.std(comp_times)
    print(f"{comp_mean:.4f}±{comp_std:.4f}s")
    
    # For fair comparison, also time full pipeline
    print("  Full pipeline...", end=" ")
    full_times = []
    for _ in range(n_runs):
        start_time = time.time()
        phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
        amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
        _ = pac_tp.fit(phases, amplitudes)
        full_times.append(time.time() - start_time)
    
    full_mean = np.mean(full_times)
    full_std = np.std(full_times)
    print(f"{full_mean:.4f}±{full_std:.4f}s")
    
    return {
        'init_time': init_time,
        'comp_mean': comp_mean,
        'comp_std': comp_std,
        'full_mean': full_mean,
        'full_std': full_std
    }

def compare_multiple_resolutions():
    """Compare performance across multiple frequency resolutions."""
    print("🎯 FAIR PERFORMANCE COMPARISON (COMPUTATION ONLY)")
    print("=" * 70)
    
    signal, fs = create_test_signal()
    
    # Test different resolutions
    resolutions = [
        (20, 15, "Small"),
        (50, 30, "Medium"), 
        (100, 70, "Large"),
        (150, 100, "Very Large")
    ]
    
    results = []
    
    for pha_bands, amp_bands, size_name in resolutions:
        print(f"\n--- {size_name} Resolution: {pha_bands}×{amp_bands} = {pha_bands*amp_bands} frequency pairs ---")
        
        # Benchmark gPAC
        gpac_result = benchmark_gpac_computation(signal, fs, pha_bands, amp_bands)
        
        # Benchmark Tensorpac
        tp_result = benchmark_tensorpac_computation(signal, fs, pha_bands, amp_bands)
        
        # Compare results
        if tp_result is not None:
            print(f"\n  📊 Comparison:")
            print(f"    Initialization: gPAC {gpac_result['init_time']:.3f}s vs TP {tp_result['init_time']:.3f}s")
            
            # Compare GPU vs Tensorpac fit-only
            if gpac_result['gpu_mean'] is not None:
                gpu_vs_tp_fit = tp_result['comp_mean'] / gpac_result['gpu_mean']
                if gpu_vs_tp_fit > 1:
                    print(f"    Computation: gPAC GPU {gpu_vs_tp_fit:.1f}x faster than TP (fit only)")
                else:
                    print(f"    Computation: TP (fit only) {1/gpu_vs_tp_fit:.1f}x faster than gPAC GPU")
                
                # Compare GPU vs Tensorpac full pipeline
                gpu_vs_tp_full = tp_result['full_mean'] / gpac_result['gpu_mean']
                if gpu_vs_tp_full > 1:
                    print(f"    Full pipeline: gPAC GPU {gpu_vs_tp_full:.1f}x faster than TP (full)")
                else:
                    print(f"    Full pipeline: TP (full) {1/gpu_vs_tp_full:.1f}x faster than gPAC GPU")
            
            # Compare CPU vs Tensorpac
            cpu_vs_tp_full = tp_result['full_mean'] / gpac_result['cpu_mean']
            if cpu_vs_tp_full > 1:
                print(f"    CPU comparison: gPAC CPU {cpu_vs_tp_full:.1f}x faster than TP (full)")
            else:
                print(f"    CPU comparison: TP (full) {1/cpu_vs_tp_full:.1f}x faster than gPAC CPU")
        
        results.append({
            'resolution': f"{pha_bands}×{amp_bands}",
            'total_pairs': pha_bands * amp_bands,
            'size_name': size_name,
            'gpac': gpac_result,
            'tensorpac': tp_result
        })
    
    return results

def main():
    """Run fair benchmark analysis."""
    print("🚀 STARTING FAIR gPAC vs TENSORPAC BENCHMARK")
    print("=" * 70)
    print("This benchmark separates initialization from computation")
    print("to provide fair performance comparison.\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    else:
        print("❌ No GPU available - CPU only comparison")
    
    if not TENSORPAC_AVAILABLE:
        print("❌ Tensorpac not available - gPAC only analysis")
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run benchmarks
    results = compare_multiple_resolutions()
    
    # Summary
    print("\n🎯 SUMMARY OF FINDINGS")
    print("=" * 70)
    
    for result in results:
        gpac = result['gpac']
        tp = result['tensorpac']
        
        print(f"\n{result['size_name']} ({result['resolution']}):")
        print(f"  gPAC init: {gpac['init_time']:.3f}s")
        if tp:
            print(f"  TP init: {tp['init_time']:.3f}s")
        
        print(f"  gPAC CPU: {gpac['cpu_mean']:.4f}±{gpac['cpu_std']:.4f}s")
        if gpac['gpu_mean']:
            print(f"  gPAC GPU: {gpac['gpu_mean']:.4f}±{gpac['gpu_std']:.4f}s ({gpac['speedup']:.1f}x speedup)")
        
        if tp:
            print(f"  TP fit: {tp['comp_mean']:.4f}±{tp['comp_std']:.4f}s")
            print(f"  TP full: {tp['full_mean']:.4f}±{tp['full_std']:.4f}s")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()