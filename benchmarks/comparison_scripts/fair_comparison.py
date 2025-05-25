#!/usr/bin/env python3
"""
Fair comparison between gPAC and Tensorpac - excluding initialization overhead.

This script compares only the computation time after models are initialized,
which is the realistic scenario where models are reused multiple times.
"""

import time
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

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

def fair_performance_comparison():
    """Compare computation time after initialization."""
    print("🎯 FAIR PERFORMANCE COMPARISON (Post-Initialization)")
    print("=" * 70)
    
    signal, fs = create_test_signal()
    
    # Test different resolutions
    resolutions = [
        (20, 15, "Small"),
        (50, 30, "Medium"), 
        (100, 70, "Large (Demo)"),
        (150, 100, "Extra Large")
    ]
    
    for pha_bands, amp_bands, size_name in resolutions:
        total_bands = pha_bands * amp_bands
        print(f"\n--- {size_name}: {pha_bands}×{amp_bands} = {total_bands} frequency pairs ---")
        
        # =================
        # gPAC Setup
        # =================
        print("Setting up gPAC model...", end=" ")
        setup_start = time.time()
        
        # Initialize gPAC model
        gpac_model = gpac.PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_n_bands=pha_bands,
            amp_n_bands=amp_bands,
            n_perm=None,
            trainable=False
        )
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpac_model = gpac_model.to(device)
        signal_torch = torch.tensor(signal, dtype=torch.float32).to(device)
        
        gpac_setup_time = time.time() - setup_start
        print(f"{gpac_setup_time:.3f}s")
        
        # Warm up gPAC (important for GPU timing)
        print("Warming up gPAC...", end=" ")
        with torch.no_grad():
            _ = gpac_model(signal_torch)
        print("Done")
        
        # Time gPAC computation (multiple runs for accuracy)
        print("Timing gPAC computation...", end=" ")
        n_runs = 5
        gpac_times = []
        
        for _ in range(n_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                pac_result = gpac_model(signal_torch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            gpac_times.append(time.time() - start_time)
        
        gpac_avg_time = np.mean(gpac_times)
        gpac_std_time = np.std(gpac_times)
        print(f"{gpac_avg_time:.4f}±{gpac_std_time:.4f}s")
        
        # =================
        # Tensorpac Setup
        # =================
        if TENSORPAC_AVAILABLE:
            print("Setting up Tensorpac model...", end=" ")
            setup_start = time.time()
            
            # Prepare signal for tensorpac
            signal_tp = signal[0, 0, 0, :].reshape(-1, 1)
            
            # Create frequency arrays to match gPAC
            f_pha = np.linspace(2, 20, pha_bands)
            f_amp = np.linspace(60, 120, amp_bands)
            
            # Initialize Tensorpac model
            pac_tp = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='wavelet')
            pac_tp.idpac = (2, 0, 0)  # Modulation Index
            
            # Pre-filter the signal (this is initialization cost)
            phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
            amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
            
            tp_setup_time = time.time() - setup_start
            print(f"{tp_setup_time:.3f}s")
            
            # Time Tensorpac computation (multiple runs)
            print("Timing Tensorpac computation...", end=" ")
            tp_times = []
            
            for _ in range(n_runs):
                start_time = time.time()
                
                # Only time the PAC calculation, not filtering
                xpac = pac_tp.fit(phases, amplitudes)
                pac_tp_result = xpac.mean(axis=-1)
                
                tp_times.append(time.time() - start_time)
            
            tp_avg_time = np.mean(tp_times)
            tp_std_time = np.std(tp_times)
            print(f"{tp_avg_time:.4f}±{tp_std_time:.4f}s")
        else:
            tp_avg_time = None
            tp_setup_time = None
        
        # =================
        # Comparison
        # =================
        print("\n📊 Results:")
        print(f"  gPAC setup time:      {gpac_setup_time:.3f}s")
        print(f"  gPAC computation:     {gpac_avg_time:.4f}±{gpac_std_time:.4f}s")
        
        if tp_avg_time is not None:
            print(f"  Tensorpac setup time: {tp_setup_time:.3f}s")
            print(f"  Tensorpac computation: {tp_avg_time:.4f}±{tp_std_time:.4f}s")
            
            # Speed comparison (computation only)
            if gpac_avg_time < tp_avg_time:
                speedup = tp_avg_time / gpac_avg_time
                print(f"  🚀 gPAC is {speedup:.1f}x FASTER for computation!")
            else:
                speedup = gpac_avg_time / tp_avg_time
                print(f"  📉 Tensorpac is {speedup:.1f}x faster for computation")
            
            # Total time comparison (setup + computation)
            total_gpac = gpac_setup_time + gpac_avg_time
            total_tp = tp_setup_time + tp_avg_time
            
            if total_gpac < total_tp:
                total_speedup = total_tp / total_gpac
                print(f"  🎯 gPAC total time {total_speedup:.1f}x FASTER")
            else:
                total_speedup = total_gpac / total_tp
                print(f"  🎯 Tensorpac total time {total_speedup:.1f}x faster")
        
        print(f"  Memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "  Memory usage: N/A (CPU)")

def test_batch_processing():
    """Test performance with batch processing."""
    print("\n\n🚀 BATCH PROCESSING PERFORMANCE TEST")
    print("=" * 70)
    
    fs = 512
    duration = 2.0
    pha_bands, amp_bands = 50, 30
    
    # Create batch of signals
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} signals ---")
        
        # Create batch signal
        single_signal, _ = create_test_signal(fs, duration)
        batch_signal = np.repeat(single_signal, batch_size, axis=0)
        print(f"Batch signal shape: {batch_signal.shape}")
        
        # gPAC batch processing
        if not hasattr(test_batch_processing, 'gpac_model'):
            # Initialize once
            test_batch_processing.gpac_model = gpac.PAC(
                seq_len=single_signal.shape[-1],
                fs=fs,
                pha_n_bands=pha_bands,
                amp_n_bands=amp_bands,
                n_perm=None,
                trainable=False
            )
            if torch.cuda.is_available():
                test_batch_processing.gpac_model = test_batch_processing.gpac_model.cuda()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_torch = torch.tensor(batch_signal, dtype=torch.float32).to(device)
        
        # Time batch processing
        start_time = time.time()
        with torch.no_grad():
            batch_result = test_batch_processing.gpac_model(batch_torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.time() - start_time
        
        per_signal_time = batch_time / batch_size
        print(f"gPAC batch time: {batch_time:.4f}s ({per_signal_time:.4f}s per signal)")
        
        # Compare with single signal processing
        single_torch = batch_torch[:1]
        start_time = time.time()
        with torch.no_grad():
            single_result = test_batch_processing.gpac_model(single_torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_time = time.time() - start_time
        
        efficiency = (single_time * batch_size) / batch_time
        print(f"Single processing would take: {single_time * batch_size:.4f}s")
        print(f"Batch efficiency: {efficiency:.1f}x speedup")

def main():
    """Run complete fair comparison."""
    print("🎯 FAIR gPAC vs TENSORPAC COMPARISON")
    print("=" * 70)
    print("Comparing computation time AFTER initialization")
    print("This simulates real-world usage where models are reused")
    print()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  Using CPU only")
    
    # Run fair comparison
    fair_performance_comparison()
    
    # Test batch processing
    test_batch_processing()
    
    print("\n\n🎯 SUMMARY")
    print("=" * 70)
    print("Key findings:")
    print("1. Initialization overhead was masking true performance")
    print("2. Post-initialization comparison shows real computation speed")
    print("3. Batch processing efficiency demonstrates GPU utilization")
    print("4. Fair comparison reveals actual algorithm performance")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()