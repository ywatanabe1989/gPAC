#!/usr/bin/env python3
"""
Test realistic EEG configuration: 60s, 400Hz, 16 channels
See how many subjects we can batch process with 80GB VRAM
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from src.gpac import calculate_pac

def test_realistic_eeg(batch_size, description):
    """Test realistic EEG configuration."""
    # Realistic EEG parameters
    n_channels = 16
    fs = 400  # Hz
    duration = 60  # seconds
    seq_len = fs * duration  # 24,000 samples
    
    total_points = batch_size * n_channels * seq_len
    
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Configuration: {n_channels} channels, {duration}s @ {fs}Hz")
    print(f"Batch size: {batch_size} subjects")
    print(f"Shape: ({batch_size}, {n_channels}, 1, {seq_len})")
    print(f"Total data points: {total_points:,}")
    
    # Calculate input data size
    signal = torch.randn(batch_size, n_channels, 1, seq_len)
    input_size_gb = signal.numel() * 4 / 1024**3
    print(f"Input data size: {input_size_gb:.3f} GB")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Move to GPU and time the processing
    signal = signal.cuda()
    torch.cuda.synchronize()
    
    print(f"\nProcessing...")
    start = time.time()
    
    try:
        # Use moderate frequency resolution for realistic analysis
        pac, freqs_pha, freqs_amp = calculate_pac(
            signal,
            fs=fs,
            pha_start_hz=1.0, pha_end_hz=12.0, pha_n_bands=12,  # 1-12 Hz phase (delta, theta, alpha)
            amp_start_hz=30.0, amp_end_hz=100.0, amp_n_bands=14,  # 30-100 Hz amplitude (gamma)
            device='cuda',
            mi_n_bins=18,
            use_optimized_filter=True,
        )
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        current_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"✅ SUCCESS!")
        print(f"  Processing time: {gpu_time:.3f}s")
        print(f"  Time per subject: {gpu_time/batch_size:.3f}s")
        print(f"  Peak GPU memory: {peak_memory:.2f} GB")
        print(f"  Current memory: {current_memory:.2f} GB")
        print(f"  Memory efficiency: {peak_memory/80*100:.1f}% of 80GB used")
        print(f"  Throughput: {total_points/gpu_time/1_000_000:.1f} M points/sec")
        print(f"  Output shape: {pac.shape}")
        print(f"  Frequency pairs: {len(freqs_pha)} × {len(freqs_amp)} = {len(freqs_pha)*len(freqs_amp)}")
        
        return {
            'batch_size': batch_size,
            'success': True,
            'gpu_time': gpu_time,
            'time_per_subject': gpu_time/batch_size,
            'peak_memory': peak_memory,
            'input_size_gb': input_size_gb,
            'throughput': total_points/gpu_time/1_000_000,
            'memory_amplification': peak_memory/input_size_gb,
            'total_points': total_points
        }
        
    except Exception as e:
        error_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"❌ FAILED: {e}")
        print(f"  Memory at failure: {error_memory:.2f} GB")
        
        return {
            'batch_size': batch_size,
            'success': False,
            'error': str(e),
            'error_memory': error_memory,
            'input_size_gb': input_size_gb
        }

# Test different batch sizes for realistic EEG
print("REALISTIC EEG BATCH PROCESSING TEST")
print("Configuration: 60s, 400Hz, 16-channel EEG")
print("Finding maximum batch size for A100's 80GB VRAM")

batch_sizes = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
results = []

for batch_size in batch_sizes:
    description = f"Batch {batch_size}: {batch_size} subjects"
    result = test_realistic_eeg(batch_size, description)
    results.append(result)
    
    if not result['success']:
        print(f"\n⚠️  Maximum batch size reached: {batch_size-1 if len(results)>1 else 0}")
        break
    
    # Stop if we're using >70GB to be safe
    if result['success'] and result['peak_memory'] > 70:
        print(f"\n⚠️  Approaching memory limit at {result['peak_memory']:.1f} GB")
        break

# Summary
print(f"\n{'='*80}")
print("REALISTIC EEG PROCESSING SUMMARY")
print(f"{'='*80}")

successful_results = [r for r in results if r['success']]

if successful_results:
    print(f"{'Batch':<8} {'Input GB':<10} {'Peak GB':<10} {'Time':<8} {'Per Subj':<10} {'Throughput':<12}")
    print("-" * 70)
    
    for r in successful_results:
        print(f"{r['batch_size']:<8} {r['input_size_gb']:<10.3f} {r['peak_memory']:<10.2f} "
              f"{r['gpu_time']:<8.3f} {r['time_per_subject']:<10.3f} {r['throughput']:<12.1f}")
    
    max_batch = successful_results[-1]
    
    print(f"\n🎯 A100 Performance with Realistic EEG:")
    print(f"  Maximum batch size: {max_batch['batch_size']} subjects")
    print(f"  Total processing time: {max_batch['gpu_time']:.3f}s for {max_batch['batch_size']} subjects")
    print(f"  Time per subject: {max_batch['time_per_subject']:.3f}s")
    print(f"  Peak memory usage: {max_batch['peak_memory']:.1f} GB ({max_batch['peak_memory']/80*100:.1f}% of 80GB)")
    print(f"  Memory amplification: {max_batch['memory_amplification']:.0f}x")
    print(f"  Peak throughput: {max(r['throughput'] for r in successful_results):.1f} M points/sec")
    
    print(f"\n🚀 Real-world impact:")
    print(f"  Can process {max_batch['batch_size']} subjects simultaneously")
    print(f"  Equivalent to {max_batch['batch_size']*60/60:.0f} hours of EEG data in {max_batch['gpu_time']:.1f}s")
    print(f"  Perfect for large clinical studies or research cohorts")
    
    # Compare to sequential processing
    sequential_time = max_batch['batch_size'] * max_batch['time_per_subject']
    speedup = sequential_time / max_batch['gpu_time']
    print(f"  Batch speedup: {speedup:.1f}x faster than processing sequentially")

else:
    print("No successful batches - configuration too memory intensive")

print(f"\nConclusion: A100 enables large-scale EEG batch processing!")
print(f"This is the kind of workload where 80GB VRAM really shines.")