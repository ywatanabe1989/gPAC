#!/usr/bin/env python3
"""
Scaling analysis to identify where gPAC performance bottlenecks occur.

This script tests different frequency resolutions to find the scaling behavior
and identify optimization targets.
"""

import time
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def benchmark_gpac_scaling():
    """Test gPAC performance across different frequency resolutions."""
    print("🔍 BENCHMARKING gPAC SCALING PERFORMANCE")
    print("=" * 60)
    
    signal, fs = create_test_signal()
    
    # Test different frequency resolutions
    resolutions = [
        (10, 8),    # Small
        (20, 15),   # Medium  
        (50, 30),   # Large
        (100, 70),  # Very Large (demo resolution)
        (150, 100), # Extreme
    ]
    
    results = {
        'resolutions': [],
        'total_bands': [],
        'gpac_cpu_times': [],
        'gpac_gpu_times': [],
        'tensorpac_times': []
    }
    
    for pha_bands, amp_bands in resolutions:
        total_bands = pha_bands * amp_bands
        print(f"\n--- Testing {pha_bands}×{amp_bands} = {total_bands} frequency pairs ---")
        
        # Test gPAC on CPU
        print("gPAC CPU...", end=" ")
        try:
            start_time = time.time()
            pac_cpu, _, _ = gpac.calculate_pac(
                signal, fs=fs,
                pha_n_bands=pha_bands, amp_n_bands=amp_bands,
                device='cpu', n_perm=None
            )
            cpu_time = time.time() - start_time
            print(f"{cpu_time:.3f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            cpu_time = None
        
        # Test gPAC on GPU  
        if torch.cuda.is_available():
            print("gPAC GPU...", end=" ")
            try:
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()
                pac_gpu, _, _ = gpac.calculate_pac(
                    signal, fs=fs,
                    pha_n_bands=pha_bands, amp_n_bands=amp_bands,
                    device='cuda', n_perm=None
                )
                gpu_time = time.time() - start_time
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                print(f"{gpu_time:.3f}s ({peak_memory:.1f}MB)")
            except Exception as e:
                print(f"FAILED: {e}")
                gpu_time = None
        else:
            gpu_time = None
        
        # Test Tensorpac
        if TENSORPAC_AVAILABLE:
            print("Tensorpac...", end=" ")
            try:
                signal_tp = signal[0, 0, 0, :].reshape(-1, 1)
                f_pha = np.linspace(2, 20, pha_bands)
                f_amp = np.linspace(60, 120, amp_bands)
                
                pac_tp = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='wavelet')
                pac_tp.idpac = (2, 0, 0)
                
                start_time = time.time()
                phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
                amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
                xpac = pac_tp.fit(phases, amplitudes)
                pac_tp_result = xpac.mean(axis=-1)
                tp_time = time.time() - start_time
                print(f"{tp_time:.3f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                tp_time = None
        else:
            tp_time = None
        
        # Store results
        results['resolutions'].append(f"{pha_bands}×{amp_bands}")
        results['total_bands'].append(total_bands)
        results['gpac_cpu_times'].append(cpu_time)
        results['gpac_gpu_times'].append(gpu_time)
        results['tensorpac_times'].append(tp_time)
    
    return results

def analyze_computational_complexity():
    """Analyze the computational complexity of gPAC components."""
    print("\n🔍 ANALYZING COMPUTATIONAL COMPLEXITY")
    print("=" * 60)
    
    signal, fs = create_test_signal()
    
    # Profile different components with moderate resolution
    pha_bands, amp_bands = 50, 30
    
    print(f"Profiling gPAC components at {pha_bands}×{amp_bands} resolution...")
    
    # Time the PAC model creation
    start_time = time.time()
    model = gpac.PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_n_bands=pha_bands,
        amp_n_bands=amp_bands,
        n_perm=None,
        trainable=False
    ).cuda() if torch.cuda.is_available() else gpac.PAC(
        seq_len=signal.shape[-1],
        fs=fs,
        pha_n_bands=pha_bands,
        amp_n_bands=amp_bands,
        n_perm=None,
        trainable=False
    )
    init_time = time.time() - start_time
    print(f"Model initialization: {init_time:.3f}s")
    
    # Time the forward pass
    if torch.cuda.is_available():
        signal_gpu = torch.tensor(signal, dtype=torch.float32).cuda()
    else:
        signal_gpu = torch.tensor(signal, dtype=torch.float32)
    
    # Warm up
    _ = model(signal_gpu)
    
    # Actual timing
    start_time = time.time()
    result = model(signal_gpu)
    forward_time = time.time() - start_time
    print(f"Forward pass: {forward_time:.3f}s")
    
    print(f"Total gPAC time: {init_time + forward_time:.3f}s")

def plot_scaling_results(results):
    """Plot the scaling analysis results."""
    print("\n📊 Creating scaling analysis plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time vs Resolution
    x = results['total_bands']
    
    if any(t is not None for t in results['gpac_cpu_times']):
        cpu_times = [t if t is not None else np.nan for t in results['gpac_cpu_times']]
        ax1.plot(x, cpu_times, 'b-o', label='gPAC CPU', linewidth=2, markersize=8)
    
    if any(t is not None for t in results['gpac_gpu_times']):
        gpu_times = [t if t is not None else np.nan for t in results['gpac_gpu_times']]
        ax1.plot(x, gpu_times, 'g-o', label='gPAC GPU', linewidth=2, markersize=8)
    
    if any(t is not None for t in results['tensorpac_times']):
        tp_times = [t if t is not None else np.nan for t in results['tensorpac_times']]
        ax1.plot(x, tp_times, 'r-o', label='Tensorpac', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Total Frequency Pairs')
    ax1.set_ylabel('Computation Time (seconds)')
    ax1.set_title('PAC Computation Time vs Frequency Resolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2: Speedup Analysis
    valid_indices = [i for i, (cpu, gpu, tp) in enumerate(zip(
        results['gpac_cpu_times'], results['gpac_gpu_times'], results['tensorpac_times']
    )) if cpu is not None and gpu is not None and tp is not None]
    
    if valid_indices:
        x_valid = [results['total_bands'][i] for i in valid_indices]
        cpu_valid = [results['gpac_cpu_times'][i] for i in valid_indices]
        gpu_valid = [results['gpac_gpu_times'][i] for i in valid_indices]
        tp_valid = [results['tensorpac_times'][i] for i in valid_indices]
        
        gpu_speedup = [cpu/gpu for cpu, gpu in zip(cpu_valid, gpu_valid)]
        tp_vs_gpac = [gpu/tp for gpu, tp in zip(gpu_valid, tp_valid)]
        
        ax2.plot(x_valid, gpu_speedup, 'g-o', label='GPU vs CPU Speedup', linewidth=2, markersize=8)
        ax2.plot(x_valid, tp_vs_gpac, 'r-o', label='Tensorpac vs gPAC Speed Ratio', linewidth=2, markersize=8)
        
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Baseline (1x)')
        ax2.set_xlabel('Total Frequency Pairs')
        ax2.set_ylabel('Speed Ratio')
        ax2.set_title('Performance Scaling Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
    
    plt.tight_layout()
    save_path = 'scaling_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scaling analysis to: {save_path}")
    plt.close()

def main():
    """Run complete scaling analysis."""
    print("🚀 STARTING gPAC SCALING ANALYSIS")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run benchmarks
    results = benchmark_gpac_scaling()
    
    # Analyze complexity
    analyze_computational_complexity()
    
    # Plot results
    plot_scaling_results(results)
    
    # Summary
    print("\n🎯 SCALING ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("Performance at different resolutions:")
    for i, res in enumerate(results['resolutions']):
        total = results['total_bands'][i]
        cpu_t = results['gpac_cpu_times'][i]
        gpu_t = results['gpac_gpu_times'][i]
        tp_t = results['tensorpac_times'][i]
        
        print(f"{res:>8} ({total:>5} pairs): ", end="")
        if cpu_t: print(f"CPU {cpu_t:.3f}s", end=" ")
        if gpu_t: print(f"GPU {gpu_t:.3f}s", end=" ")
        if tp_t: print(f"TP {tp_t:.3f}s", end=" ")
        
        if gpu_t and tp_t:
            ratio = gpu_t / tp_t
            if ratio < 1:
                print(f"(gPAC {1/ratio:.1f}x faster)")
            else:
                print(f"(TP {ratio:.1f}x faster)")
        else:
            print()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()