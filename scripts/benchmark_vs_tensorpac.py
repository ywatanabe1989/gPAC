#!/usr/bin/env python3
"""
Benchmark optimized gPAC vs TensorPAC.
"""

import torch
import numpy as np
import time
import sys
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')

from gpac import calculate_pac

# Try to import TensorPAC
try:
    from tensorpac import Pac as TensorPAC
    TENSORPAC_AVAILABLE = True
except ImportError:
    print("TensorPAC not available. Install with: pip install tensorpac")
    TENSORPAC_AVAILABLE = False

def benchmark_comparison():
    """Compare optimized gPAC with TensorPAC."""
    
    if not TENSORPAC_AVAILABLE:
        print("Skipping comparison - TensorPAC not installed")
        return
    
    # Test parameters
    fs = 512.0
    duration = 2.0
    n_times = int(fs * duration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Signal length: {n_times} samples")
    print("-" * 50)
    
    # Generate test signal (numpy for TensorPAC)
    np.random.seed(42)
    signal_np = np.random.randn(1, n_times).astype(np.float32)
    signal_torch = torch.from_numpy(signal_np).unsqueeze(0).unsqueeze(0)
    
    if device == 'cuda':
        signal_torch = signal_torch.cuda()
    
    # Test configurations
    configs = [
        {"name": "Low resolution", "pha_n": 10, "amp_n": 10},
        {"name": "Medium resolution", "pha_n": 20, "amp_n": 20},
        {"name": "High resolution", "pha_n": 50, "amp_n": 30},
    ]
    
    for config in configs:
        print(f"\n{config['name']} ({config['pha_n']}x{config['amp_n']} bands):")
        
        # TensorPAC
        print("  TensorPAC:")
        
        # Create frequency vectors
        f_pha = np.linspace(2, 20, config['pha_n'] + 1)
        f_amp = np.linspace(60, 160, config['amp_n'] + 1)
        
        # Initialize TensorPAC
        p = TensorPAC(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert')
        
        # Warmup
        _ = p.filterfit(sf=fs, x_pha=signal_np, x_amp=signal_np)
        
        # Time it
        start = time.time()
        pac_tp = p.filterfit(sf=fs, x_pha=signal_np, x_amp=signal_np)
        time_tp = (time.time() - start) * 1000
        print(f"    Time: {time_tp:.2f} ms")
        
        # gPAC (optimized)
        print("  gPAC (optimized):")
        
        # Warmup
        _ = calculate_pac(
            signal_torch,
            fs=fs,
            pha_n_bands=config['pha_n'],
            amp_n_bands=config['amp_n'],
            device=device,
            use_optimized_filter=True
        )
        
        # Time it
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        pac_gpac, _, _ = calculate_pac(
            signal_torch,
            fs=fs,
            pha_n_bands=config['pha_n'],
            amp_n_bands=config['amp_n'],
            device=device,
            use_optimized_filter=True
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        time_gpac = (time.time() - start) * 1000
        print(f"    Time: {time_gpac:.2f} ms")
        
        # Speedup
        speedup = time_tp / time_gpac
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✅ gPAC is {speedup:.1f}x faster!")
        else:
            print(f"  ❌ TensorPAC is {1/speedup:.1f}x faster")
    
    # Test batch processing
    print("\n" + "="*50)
    print("Batch processing test (8 signals):")
    
    batch_np = np.random.randn(8, n_times).astype(np.float32)
    batch_torch = torch.from_numpy(batch_np).unsqueeze(1).unsqueeze(1)
    if device == 'cuda':
        batch_torch = batch_torch.cuda()
    
    # TensorPAC (processes one at a time)
    print("  TensorPAC (sequential):")
    p = TensorPAC(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert')
    
    start = time.time()
    pac_tp_batch = []
    for i in range(8):
        pac = p.filterfit(sf=fs, x_pha=batch_np[i:i+1], x_amp=batch_np[i:i+1])
        pac_tp_batch.append(pac)
    time_tp_batch = (time.time() - start) * 1000
    print(f"    Time: {time_tp_batch:.2f} ms ({time_tp_batch/8:.2f} ms/sample)")
    
    # gPAC (batch processing)
    print("  gPAC (batch):")
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    pac_gpac_batch, _, _ = calculate_pac(
        batch_torch,
        fs=fs,
        pha_n_bands=config['pha_n'],
        amp_n_bands=config['amp_n'],
        device=device,
        use_optimized_filter=True
    )
    if device == 'cuda':
        torch.cuda.synchronize()
    time_gpac_batch = (time.time() - start) * 1000
    print(f"    Time: {time_gpac_batch:.2f} ms ({time_gpac_batch/8:.2f} ms/sample)")
    
    batch_speedup = time_tp_batch / time_gpac_batch
    print(f"  Batch speedup: {batch_speedup:.2f}x")
    
    if batch_speedup > 1:
        print(f"  ✅ gPAC batch processing is {batch_speedup:.1f}x faster!")

if __name__ == "__main__":
    print("gPAC vs TensorPAC Benchmark")
    print("=" * 50)
    benchmark_comparison()
    print("\n✅ Benchmark complete!")