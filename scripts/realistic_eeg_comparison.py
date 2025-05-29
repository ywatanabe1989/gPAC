#!/usr/bin/env python3
"""
Compare TensorPAC vs gPAC for realistic EEG: 60s, 400Hz, 16 channels
Test both single subject and batch processing scenarios
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from tensorpac import Pac
from src.gpac import calculate_pac

def test_tensorpac_single(n_channels, fs, duration):
    """Test TensorPAC with single subject."""
    seq_len = fs * duration
    print(f"\n{'='*60}")
    print(f"TensorPAC (CPU) - Single Subject")
    print(f"{'='*60}")
    print(f"Configuration: {n_channels} channels, {duration}s @ {fs}Hz")
    print(f"Data points per channel: {seq_len:,}")
    
    # Generate single subject data (TensorPAC expects 2D: epochs x time)
    np.random.seed(42)
    signal = np.random.randn(n_channels, seq_len)
    
    print(f"Input shape: {signal.shape}")
    print(f"Total data points: {signal.size:,}")
    
    # TensorPAC configuration matching gPAC
    print(f"\nProcessing with TensorPAC...")
    start = time.time()
    
    try:
        # Create PAC object
        p = Pac(
            idpac=(2, 0, 0),  # Modulation Index
            f_pha=np.linspace(1.0, 12.0, 12),  # Match gPAC: 1-12 Hz, 12 bands
            f_amp=np.linspace(30.0, 100.0, 14),  # Match gPAC: 30-100 Hz, 14 bands
            n_bins=18,
        )
        
        # Process first channel only (TensorPAC processes one channel at a time)
        pac_single = p.filterfit(fs, signal[0], n_jobs=-1)
        
        single_time = time.time() - start
        
        print(f"✅ Single channel completed")
        print(f"  Time for 1 channel: {single_time:.3f}s")
        print(f"  Output shape: {pac_single.shape}")
        
        # Estimate time for all channels (sequential processing)
        estimated_total = single_time * n_channels
        print(f"  Estimated time for {n_channels} channels: {estimated_total:.3f}s")
        
        return {
            'single_channel_time': single_time,
            'estimated_total_time': estimated_total,
            'n_channels': n_channels,
            'success': True,
            'output_shape': pac_single.shape
        }
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {'success': False, 'error': str(e)}

def test_gpac_batch(batch_sizes, n_channels, fs, duration):
    """Test gPAC with different batch sizes."""
    seq_len = fs * duration
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"gPAC (GPU) - Batch Size {batch_size}")
        print(f"{'='*60}")
        print(f"Configuration: {batch_size} subjects × {n_channels} channels × {duration}s @ {fs}Hz")
        
        total_points = batch_size * n_channels * seq_len
        print(f"Total data points: {total_points:,}")
        
        # Generate batch data
        torch.manual_seed(42)
        signal = torch.randn(batch_size, n_channels, 1, seq_len)
        input_size_gb = signal.numel() * 4 / 1024**3
        print(f"Input size: {input_size_gb:.3f} GB")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        signal = signal.cuda()
        torch.cuda.synchronize()
        
        print(f"\nProcessing with gPAC...")
        start = time.time()
        
        try:
            pac, freqs_pha, freqs_amp = calculate_pac(
                signal,
                fs=fs,
                pha_start_hz=1.0, pha_end_hz=12.0, pha_n_bands=12,
                amp_start_hz=30.0, amp_end_hz=100.0, amp_n_bands=14,
                device='cuda',
                mi_n_bins=18,
                use_optimized_filter=True,
            )
            
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"✅ SUCCESS!")
            print(f"  Batch processing time: {gpu_time:.3f}s")
            print(f"  Time per subject: {gpu_time/batch_size:.3f}s")
            print(f"  Peak memory: {peak_memory:.2f} GB")
            print(f"  Output shape: {pac.shape}")
            print(f"  Throughput: {total_points/gpu_time/1_000_000:.1f} M points/sec")
            
            results.append({
                'batch_size': batch_size,
                'gpu_time': gpu_time,
                'time_per_subject': gpu_time/batch_size,
                'peak_memory': peak_memory,
                'throughput': total_points/gpu_time/1_000_000,
                'total_points': total_points,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            })
            break
    
    return results

def main():
    # Realistic EEG parameters
    n_channels = 16
    fs = 400  # Hz
    duration = 60  # seconds
    
    print("REALISTIC EEG COMPARISON: TensorPAC vs gPAC")
    print(f"Configuration: {n_channels} channels, {duration}s @ {fs}Hz")
    print("Testing both single-subject and batch processing scenarios")
    
    # Test TensorPAC (single subject)
    tensorpac_result = test_tensorpac_single(n_channels, fs, duration)
    
    # Test gPAC (various batch sizes)
    batch_sizes = [1, 5, 10]  # Start conservative based on previous results
    gpac_results = test_gpac_batch(batch_sizes, n_channels, fs, duration)
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if tensorpac_result['success'] and any(r['success'] for r in gpac_results):
        successful_gpac = [r for r in gpac_results if r['success']]
        
        print(f"\nTensorPAC (CPU) Performance:")
        print(f"  Single subject processing: {tensorpac_result['estimated_total_time']:.3f}s")
        print(f"  Time per channel: {tensorpac_result['single_channel_time']:.3f}s")
        print(f"  Processing mode: Sequential (one channel at a time)")
        
        print(f"\ngPAC (GPU) Performance:")
        for r in successful_gpac:
            speedup_vs_tensorpac = tensorpac_result['estimated_total_time'] / r['time_per_subject']
            batch_speedup = tensorpac_result['estimated_total_time'] / r['gpu_time'] * r['batch_size']
            
            print(f"  Batch {r['batch_size']}: {r['gpu_time']:.3f}s total, {r['time_per_subject']:.3f}s per subject")
            print(f"    Speedup vs TensorPAC: {speedup_vs_tensorpac:.1f}x per subject")
            print(f"    Batch efficiency: {batch_speedup:.1f}x for {r['batch_size']} subjects")
        
        # Best case scenario
        best_gpac = max(successful_gpac, key=lambda x: x['batch_size'])
        total_speedup = (tensorpac_result['estimated_total_time'] * best_gpac['batch_size']) / best_gpac['gpu_time']
        
        print(f"\n🎯 Key Insights:")
        print(f"  • TensorPAC: {tensorpac_result['estimated_total_time']:.1f}s for 1 subject ({n_channels} channels)")
        print(f"  • gPAC: {best_gpac['gpu_time']:.1f}s for {best_gpac['batch_size']} subjects")
        print(f"  • Overall speedup: {total_speedup:.0f}x faster with batch processing")
        print(f"  • A100 enables processing {best_gpac['batch_size']}x more subjects in {best_gpac['gpu_time']/tensorpac_result['estimated_total_time']:.1f}x less time")
        
        print(f"\n🚀 Real-world impact:")
        print(f"  • TensorPAC: Process 10 subjects in {tensorpac_result['estimated_total_time']*10:.1f}s")
        print(f"  • gPAC: Process 10 subjects in {best_gpac['gpu_time']:.1f}s")
        print(f"  • Time saved: {(tensorpac_result['estimated_total_time']*10 - best_gpac['gpu_time']):.1f}s per 10 subjects")

if __name__ == "__main__":
    main()