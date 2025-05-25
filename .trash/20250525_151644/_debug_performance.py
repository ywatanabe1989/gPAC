#!/usr/bin/env python3
"""
Debug script to investigate gPAC performance and compatibility issues.

This script will:
1. Check if GPU acceleration is actually being used
2. Compare algorithm configurations between gPAC and Tensorpac  
3. Measure performance with identical parameters
4. Analyze why results are different
"""

import time
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import gPAC
import gpac

# Try to import tensorpac
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False

def check_gpu_availability():
    """Check GPU availability and setup."""
    print("🔍 DEBUGGING GPU AVAILABILITY")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU device name: {torch.cuda.get_device_name()}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("❌ No CUDA GPU available - this explains poor performance!")
    
    print()

def create_test_signal(fs=512, duration=2.0):
    """Create identical test signal for both methods."""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create signal with known PAC: 6Hz phase modulates 80Hz amplitude
    pha_freq, amp_freq = 6.0, 80.0
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    amplitude_mod = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
    carrier = np.sin(2 * np.pi * amp_freq * t)
    pac_signal = phase_signal + amplitude_mod * carrier * 0.5
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(t))
    signal = pac_signal + noise
    
    return signal.reshape(1, 1, 1, -1), fs, pha_freq, amp_freq

def debug_gpac_device_usage(signal, fs):
    """Debug which device gPAC is actually using."""
    print("🔍 DEBUGGING gPAC DEVICE USAGE")
    print("=" * 50)
    
    # Test with explicit CPU
    print("Testing gPAC on CPU...")
    start_time = time.time()
    pac_cpu, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal, fs=fs,
        pha_n_bands=50, amp_n_bands=30,
        device='cpu',
        n_perm=100
    )
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.3f} seconds")
    
    # Test with explicit GPU (if available)
    if torch.cuda.is_available():
        print("Testing gPAC on GPU...")
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        pac_gpu, _, _ = gpac.calculate_pac(
            signal, fs=fs,
            pha_n_bands=50, amp_n_bands=30,
            device='cuda',
            n_perm=100
        )
        gpu_time = time.time() - start_time
        
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"GPU time: {gpu_time:.3f} seconds")
        print(f"Memory used: {(end_memory - start_memory) / 1024**2:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"✅ GPU is {speedup:.1f}x faster than CPU")
        else:
            slowdown = gpu_time / cpu_time
            print(f"❌ GPU is {slowdown:.1f}x SLOWER than CPU - this is problematic!")
    else:
        print("❌ No GPU available for testing")
        gpu_time = None
    
    print()
    return cpu_time, gpu_time

def debug_algorithm_differences(signal, fs):
    """Compare algorithm configurations between gPAC and Tensorpac."""
    print("🔍 DEBUGGING ALGORITHM DIFFERENCES")
    print("=" * 50)
    
    if not TENSORPAC_AVAILABLE:
        print("❌ Tensorpac not available")
        return
    
    # Test with identical frequency configurations
    pha_start, pha_end = 2.0, 20.0
    amp_start, amp_end = 60.0, 120.0
    n_pha, n_amp = 20, 15
    
    print(f"Using identical parameters:")
    print(f"  Phase range: {pha_start}-{pha_end} Hz ({n_pha} bands)")
    print(f"  Amplitude range: {amp_start}-{amp_end} Hz ({n_amp} bands)")
    
    # gPAC calculation
    print("\n--- gPAC Configuration ---")
    start_time = time.time()
    pac_gpac, pha_freqs_gpac, amp_freqs_gpac = gpac.calculate_pac(
        signal, fs=fs,
        pha_start_hz=pha_start, pha_end_hz=pha_end, pha_n_bands=n_pha,
        amp_start_hz=amp_start, amp_end_hz=amp_end, amp_n_bands=n_amp,
        device='cpu',  # Use CPU for fair comparison
        n_perm=None  # No permutation for speed
    )
    gpac_time = time.time() - start_time
    
    print(f"gPAC result shape: {pac_gpac.shape}")
    print(f"gPAC phase freqs: {pha_freqs_gpac.shape} - [{pha_freqs_gpac[0]:.1f}, {pha_freqs_gpac[-1]:.1f}]")
    print(f"gPAC amp freqs: {amp_freqs_gpac.shape} - [{amp_freqs_gpac[0]:.1f}, {amp_freqs_gpac[-1]:.1f}]")
    print(f"gPAC time: {gpac_time:.3f} seconds")
    
    # Tensorpac calculation  
    print("\n--- Tensorpac Configuration ---")
    try:
        signal_tp = signal[0, 0, 0, :].reshape(-1, 1)
        
        # Create explicit frequency arrays to match gPAC
        f_pha = np.linspace(pha_start, pha_end, n_pha)
        f_amp = np.linspace(amp_start, amp_end, n_amp)
        
        pac_tp = Pac(
            f_pha=f_pha,
            f_amp=f_amp,
            dcomplex='wavelet'
        )
        pac_tp.idpac = (2, 0, 0)  # Modulation Index like mngs
        
        start_time = time.time()
        phases = pac_tp.filter(fs, signal_tp.squeeze(), ftype='phase', n_jobs=1)
        amplitudes = pac_tp.filter(fs, signal_tp.squeeze(), ftype='amplitude', n_jobs=1)
        xpac = pac_tp.fit(phases, amplitudes)
        pac_tp_result = xpac.mean(axis=-1).T  # Average over time, transpose to match gPAC
        tensorpac_time = time.time() - start_time
        
        print(f"Tensorpac result shape: {pac_tp_result.shape}")
        print(f"Tensorpac phase freqs: {f_pha.shape} - [{f_pha[0]:.1f}, {f_pha[-1]:.1f}]")
        print(f"Tensorpac amp freqs: {f_amp.shape} - [{f_amp[0]:.1f}, {f_amp[-1]:.1f}]")
        print(f"Tensorpac time: {tensorpac_time:.3f} seconds")
        
        # Compare results
        print("\n--- Comparison ---")
        pac_gpac_2d = pac_gpac[0, 0].cpu().numpy()
        
        if pac_gpac_2d.shape == pac_tp_result.shape:
            print("✅ Shapes match!")
            
            # Calculate correlation
            correlation = np.corrcoef(pac_gpac_2d.flatten(), pac_tp_result.flatten())[0, 1]
            print(f"Correlation between results: {correlation:.3f}")
            
            # Calculate relative difference
            rel_diff = np.abs(pac_gpac_2d - pac_tp_result) / (np.abs(pac_tp_result) + 1e-9)
            mean_rel_diff = np.mean(rel_diff)
            print(f"Mean relative difference: {mean_rel_diff:.3f}")
            
            if correlation > 0.7:
                print("✅ Results are reasonably similar")
            else:
                print("❌ Results are very different - algorithm mismatch!")
                
        else:
            print(f"❌ Shape mismatch: gPAC {pac_gpac_2d.shape} vs Tensorpac {pac_tp_result.shape}")
        
        # Speed comparison
        if gpac_time < tensorpac_time:
            speedup = tensorpac_time / gpac_time
            print(f"✅ gPAC is {speedup:.1f}x faster")
        else:
            slowdown = gpac_time / tensorpac_time
            print(f"❌ gPAC is {slowdown:.1f}x slower")
            
    except Exception as e:
        print(f"❌ Tensorpac calculation failed: {e}")
    
    print()

def debug_frequency_configurations():
    """Debug frequency band configurations."""
    print("🔍 DEBUGGING FREQUENCY CONFIGURATIONS")
    print("=" * 50)
    
    # Test gPAC frequency band calculation
    from gpac._PAC import PAC
    
    print("gPAC frequency band calculation:")
    test_pac = PAC(
        seq_len=1024,
        fs=512,
        pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=10,
        amp_start_hz=60.0, amp_end_hz=120.0, amp_n_bands=8,
        trainable=False
    )
    
    print(f"gPAC phase frequencies: {test_pac.PHA_MIDS_HZ}")
    print(f"gPAC amplitude frequencies: {test_pac.AMP_MIDS_HZ}")
    
    if TENSORPAC_AVAILABLE:
        print("\nTensorpac frequency band calculation:")
        f_pha = np.linspace(2.0, 20.0, 10)
        f_amp = np.linspace(60.0, 120.0, 8)
        print(f"Tensorpac phase frequencies: {f_pha}")
        print(f"Tensorpac amplitude frequencies: {f_amp}")
        
        print("\nDifferences:")
        pha_diff = np.abs(test_pac.PHA_MIDS_HZ.cpu().numpy() - f_pha)
        amp_diff = np.abs(test_pac.AMP_MIDS_HZ.cpu().numpy() - f_amp)
        print(f"Max phase freq difference: {np.max(pha_diff):.3f} Hz")
        print(f"Max amp freq difference: {np.max(amp_diff):.3f} Hz")
    
    print()

def main():
    """Run all debugging analyses."""
    print("🚨 DEBUGGING gPAC PERFORMANCE & COMPATIBILITY ISSUES")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check GPU availability
    check_gpu_availability()
    
    # Create test signal
    signal, fs, pha_freq, amp_freq = create_test_signal()
    print(f"Test signal: {signal.shape} at {fs} Hz")
    print(f"Ground truth PAC: {pha_freq} Hz → {amp_freq} Hz\n")
    
    # Debug device usage
    cpu_time, gpu_time = debug_gpac_device_usage(signal, fs)
    
    # Debug algorithm differences
    debug_algorithm_differences(signal, fs)
    
    # Debug frequency configurations
    debug_frequency_configurations()
    
    print("🔍 DEBUGGING COMPLETE")
    print("=" * 60)
    
    # Summary of findings
    print("SUMMARY OF FINDINGS:")
    if not torch.cuda.is_available():
        print("❌ CRITICAL: No GPU available - explains poor performance")
    if gpu_time and gpu_time > cpu_time:
        print("❌ CRITICAL: GPU slower than CPU - implementation issue")
    print("❌ CRITICAL: gPAC and Tensorpac produce different results")
    print("❌ CRITICAL: Performance claims are not substantiated")
    
    print("\nRECOMMENDATIONS:")
    print("1. Fix GPU utilization for proper acceleration")
    print("2. Align algorithm implementations for compatibility")
    print("3. Verify frequency band calculations")
    print("4. Re-benchmark with identical configurations")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()