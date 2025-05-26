#!/usr/bin/env python3
"""
Comprehensive Performance Comparison: gPAC vs TensorPAC
Including initialization times and multiple scenarios
"""

import time
import numpy as np
import torch
import multiprocessing
from gpac import calculate_pac, PAC

# Try to import tensorpac
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("TensorPAC not available")
    exit(1)

# Get number of CPU cores
n_cores = multiprocessing.cpu_count()
print(f"System has {n_cores} CPU cores available")
print(f"GPU available: {torch.cuda.is_available()}")

# Test parameters
fs = 1000
duration = 5
n_samples = int(fs * duration)
n_trials = 5

# Generate test signal
t = np.linspace(0, duration, n_samples, endpoint=False)
phase_freq = 8
amp_freq = 80
signal = (np.sin(2 * np.pi * phase_freq * t) + 
          0.5 * np.sin(2 * np.pi * amp_freq * t) + 
          0.2 * np.random.randn(n_samples))

# Frequency parameters
pha_bands = 50
amp_bands = 30

print(f"\nTest Configuration:")
print(f"Signal: {duration}s @ {fs}Hz")
print(f"Frequency resolution: {pha_bands} x {amp_bands} = {pha_bands * amp_bands} pairs")
print(f"Running {n_trials} trials for timing\n")

# Store all results
results = {}

# 1. TensorPAC with initialization timing
print("=" * 60)
print("1. TensorPAC Performance (including initialization)")
print("=" * 60)

# Test single-core
print("\nTensorPAC with n_jobs=1:")
init_times_tp1 = []
comp_times_tp1 = []

for i in range(n_trials):
    # Time initialization
    init_start = time.time()
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=(2, 20, pha_bands), f_amp=(60, 160, amp_bands))
    init_time = time.time() - init_start
    init_times_tp1.append(init_time)
    
    # Time computation
    comp_start = time.time()
    pac_tp.filterfit(fs, signal, n_jobs=1)
    comp_time = time.time() - comp_start
    comp_times_tp1.append(comp_time)
    
    print(f"  Trial {i+1}: init={init_time:.3f}s, compute={comp_time:.3f}s, total={init_time+comp_time:.3f}s")

results['tp_single'] = {
    'init': np.mean(init_times_tp1),
    'compute': np.mean(comp_times_tp1),
    'total': np.mean(init_times_tp1) + np.mean(comp_times_tp1)
}

# Test multi-core
print(f"\nTensorPAC with n_jobs=-1 ({n_cores} cores):")
init_times_tp_multi = []
comp_times_tp_multi = []

for i in range(n_trials):
    # Time initialization
    init_start = time.time()
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=(2, 20, pha_bands), f_amp=(60, 160, amp_bands))
    init_time = time.time() - init_start
    init_times_tp_multi.append(init_time)
    
    # Time computation
    comp_start = time.time()
    pac_tp.filterfit(fs, signal, n_jobs=-1)
    comp_time = time.time() - comp_start
    comp_times_tp_multi.append(comp_time)
    
    print(f"  Trial {i+1}: init={init_time:.3f}s, compute={comp_time:.3f}s, total={init_time+comp_time:.3f}s")

results['tp_multi'] = {
    'init': np.mean(init_times_tp_multi),
    'compute': np.mean(comp_times_tp_multi),
    'total': np.mean(init_times_tp_multi) + np.mean(comp_times_tp_multi)
}

# 2. gPAC Performance
print("\n" + "=" * 60)
print("2. gPAC Performance (including initialization)")
print("=" * 60)

# Test CPU with function API (includes initialization)
print("\ngPAC CPU (function API - includes init):")
signal_torch = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)
total_times_gpac_cpu_func = []

for i in range(n_trials):
    start = time.time()
    pac_gpac_cpu, _, _ = calculate_pac(
        signal_torch,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=pha_bands,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=amp_bands,
        device="cpu"
    )
    total_time = time.time() - start
    total_times_gpac_cpu_func.append(total_time)
    print(f"  Trial {i+1}: total={total_time:.3f}s")

results['gpac_cpu_func'] = {
    'total': np.mean(total_times_gpac_cpu_func)
}

# Test CPU with class API (separate init and compute)
print("\ngPAC CPU (class API - separate init/compute):")
init_times_gpac_cpu = []
comp_times_gpac_cpu = []

for i in range(n_trials):
    # Time initialization
    init_start = time.time()
    pac_model = PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=pha_bands,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=amp_bands,
        trainable=False
    )
    pac_model.eval()
    init_time = time.time() - init_start
    init_times_gpac_cpu.append(init_time)
    
    # Time computation
    comp_start = time.time()
    result = pac_model(signal_torch)
    comp_time = time.time() - comp_start
    comp_times_gpac_cpu.append(comp_time)
    
    print(f"  Trial {i+1}: init={init_time:.3f}s, compute={comp_time:.3f}s, total={init_time+comp_time:.3f}s")

results['gpac_cpu_class'] = {
    'init': np.mean(init_times_gpac_cpu),
    'compute': np.mean(comp_times_gpac_cpu),
    'total': np.mean(init_times_gpac_cpu) + np.mean(comp_times_gpac_cpu)
}

# Test GPU if available
if torch.cuda.is_available():
    # GPU with function API
    print("\ngPAC GPU (function API - includes init):")
    signal_gpu = signal_torch.cuda()
    total_times_gpac_gpu_func = []
    
    for i in range(n_trials):
        torch.cuda.synchronize()
        start = time.time()
        pac_gpac_gpu, _, _ = calculate_pac(
            signal_gpu,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=pha_bands,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=amp_bands
        )
        torch.cuda.synchronize()
        total_time = time.time() - start
        total_times_gpac_gpu_func.append(total_time)
        print(f"  Trial {i+1}: total={total_time:.3f}s")
    
    results['gpac_gpu_func'] = {
        'total': np.mean(total_times_gpac_gpu_func)
    }
    
    # GPU with class API
    print("\ngPAC GPU (class API - separate init/compute):")
    init_times_gpac_gpu = []
    comp_times_gpac_gpu = []
    
    for i in range(n_trials):
        # Time initialization
        torch.cuda.synchronize()
        init_start = time.time()
        pac_model_gpu = PAC(
            seq_len=n_samples,
            fs=fs,
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=pha_bands,
            amp_start_hz=60,
            amp_end_hz=160,
            amp_n_bands=amp_bands,
            trainable=False
        ).cuda()
        pac_model_gpu.eval()
        torch.cuda.synchronize()
        init_time = time.time() - init_start
        init_times_gpac_gpu.append(init_time)
        
        # Time computation
        torch.cuda.synchronize()
        comp_start = time.time()
        result = pac_model_gpu(signal_gpu)
        torch.cuda.synchronize()
        comp_time = time.time() - comp_start
        comp_times_gpac_gpu.append(comp_time)
        
        print(f"  Trial {i+1}: init={init_time:.3f}s, compute={comp_time:.3f}s, total={init_time+comp_time:.3f}s")
    
    results['gpac_gpu_class'] = {
        'init': np.mean(init_times_gpac_gpu),
        'compute': np.mean(comp_times_gpac_gpu),
        'total': np.mean(init_times_gpac_gpu) + np.mean(comp_times_gpac_gpu)
    }

# 3. Summary
print("\n" + "=" * 60)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("=" * 60)

print("\nAverage Times (seconds):")
print("-" * 50)
print(f"{'Method':<30} {'Init':<10} {'Compute':<10} {'Total':<10}")
print("-" * 50)

# TensorPAC results
print(f"{'TensorPAC (1 CPU)':<30} {results['tp_single']['init']:<10.3f} {results['tp_single']['compute']:<10.3f} {results['tp_single']['total']:<10.3f}")
print(f"{'TensorPAC (' + str(n_cores) + ' CPUs)':<30} {results['tp_multi']['init']:<10.3f} {results['tp_multi']['compute']:<10.3f} {results['tp_multi']['total']:<10.3f}")

# gPAC CPU results
print(f"{'gPAC CPU (function API)':<30} {'N/A':<10} {'N/A':<10} {results['gpac_cpu_func']['total']:<10.3f}")
print(f"{'gPAC CPU (class API)':<30} {results['gpac_cpu_class']['init']:<10.3f} {results['gpac_cpu_class']['compute']:<10.3f} {results['gpac_cpu_class']['total']:<10.3f}")

# gPAC GPU results
if 'gpac_gpu_func' in results:
    print(f"{'gPAC GPU (function API)':<30} {'N/A':<10} {'N/A':<10} {results['gpac_gpu_func']['total']:<10.3f}")
    print(f"{'gPAC GPU (class API)':<30} {results['gpac_gpu_class']['init']:<10.3f} {results['gpac_gpu_class']['compute']:<10.3f} {results['gpac_gpu_class']['total']:<10.3f}")

print("\n" + "=" * 60)
print("SPEEDUP ANALYSIS")
print("=" * 60)

# Use TensorPAC multi-core as baseline
baseline_init = results['tp_multi']['init']
baseline_compute = results['tp_multi']['compute']
baseline_total = results['tp_multi']['total']

print(f"\nRelative to TensorPAC ({n_cores} CPUs):")
print("-" * 50)

# Compute-only comparison
print("\nComputation only (excluding initialization):")
if baseline_compute > 0:
    print(f"  gPAC CPU: {baseline_compute/results['gpac_cpu_class']['compute']:.1f}x")
    if 'gpac_gpu_class' in results:
        print(f"  gPAC GPU: {baseline_compute/results['gpac_gpu_class']['compute']:.1f}x")

# Total time comparison
print("\nTotal time (including initialization):")
print(f"  gPAC CPU (function): {baseline_total/results['gpac_cpu_func']['total']:.1f}x")
print(f"  gPAC CPU (class): {baseline_total/results['gpac_cpu_class']['total']:.1f}x")
if 'gpac_gpu_func' in results:
    print(f"  gPAC GPU (function): {baseline_total/results['gpac_gpu_func']['total']:.1f}x")
    print(f"  gPAC GPU (class): {baseline_total/results['gpac_gpu_class']['total']:.1f}x")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print("1. TensorPAC has faster initialization but slower computation")
print("2. gPAC initialization is slower due to filter design computation")
print("3. For single computations, initialization time matters")
print("4. For repeated computations, compute time dominates")
print("5. GPU advantage is mainly in computation, not initialization")