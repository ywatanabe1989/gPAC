#!/usr/bin/env python3
"""
Realistic batch sizes to better utilize A100 VRAM.
Focus on practical neuroscience batch processing.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from src.gpac import calculate_pac

def test_batch_size(batch_size, n_channels, seq_len, description):
    """Test with realistic batch sizes."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    n_segments = 1
    total_points = batch_size * n_channels * n_segments * seq_len
    
    print(f"Batch size: {batch_size}")
    print(f"Shape: ({batch_size}, {n_channels}, {n_segments}, {seq_len})")
    print(f"Total data points: {total_points:,}")
    
    # Generate batch
    print("Generating data...")
    torch.manual_seed(42)
    signal = torch.randn(batch_size, n_channels, n_segments, seq_len)
    input_size_gb = signal.numel() * 4 / 1024**3
    print(f"Input tensor size: {input_size_gb:.3f} GB")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check memory before
    memory_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    # GPU timing
    print(f"\nGPU processing...")
    start = time.time()
    
    try:
        pac_gpu, _, _ = calculate_pac(
            signal,
            fs=1000,
            pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=15,  # Moderate size
            amp_start_hz=60.0, amp_end_hz=160.0, amp_n_bands=15,
            device='cuda',
            mi_n_bins=18,
            use_optimized_filter=True,
        )
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Check memory after
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_used = memory_after - memory_before
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✅ Success!")
        print(f"  Time: {gpu_time:.4f}s")
        print(f"  Memory used: {memory_used:.2f} GB")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Throughput: {total_points / gpu_time / 1_000_000:.1f} M points/sec")
        print(f"  Output shape: {pac_gpu.shape}")
        
        # Reset peak memory counter
        torch.cuda.reset_peak_memory_stats()
        
        return {
            'batch_size': batch_size,
            'total_points': total_points,
            'gpu_time': gpu_time,
            'memory_used': memory_used,
            'peak_memory': peak_memory,
            'throughput': total_points / gpu_time / 1_000_000,
            'input_size_gb': input_size_gb,
            'success': True
        }
        
    except Exception as e:
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_used = memory_after - memory_before
        print(f"❌ Failed: {e}")
        print(f"  Memory when failed: {memory_used:.2f} GB")
        torch.cuda.empty_cache()
        
        return {
            'batch_size': batch_size,
            'total_points': total_points,
            'memory_used': memory_used,
            'input_size_gb': input_size_gb,
            'success': False,
            'error': str(e)
        }

# Progressive batch size tests
test_cases = [
    (1, 64, 50000, "Single sample: 64-channel, 50s EEG"),
    (5, 64, 30000, "Small batch: 5 subjects, 30s each"), 
    (10, 64, 20000, "Medium batch: 10 subjects, 20s each"),
    (20, 64, 15000, "Large batch: 20 subjects, 15s each"),
    (50, 32, 15000, "Huge batch: 50 subjects, 32ch, 15s"),
    (100, 16, 15000, "Ultra batch: 100 subjects, 16ch, 15s"),
    (200, 8, 15000, "Massive batch: 200 subjects, 8ch, 15s"),
    (500, 4, 10000, "Extreme batch: 500 subjects, 4ch, 10s"),
]

print("REALISTIC BATCH PROCESSING TEST")
print("Finding the sweet spot for A100's 80GB VRAM")
print("Simulating real neuroscience batch processing scenarios")

results = []
max_successful_batch = 0

for batch_size, n_channels, seq_len, description in test_cases:
    result = test_batch_size(batch_size, n_channels, seq_len, description)
    results.append(result)
    
    if result['success']:
        max_successful_batch = batch_size
    else:
        print(f"\n⚠️  Reached memory limit at batch size {batch_size}")
        break

# Summary
print(f"\n{'='*80}")
print("BATCH PROCESSING SUMMARY")
print(f"{'='*80}")
print(f"{'Batch':<8} {'Points':<12} {'Input GB':<10} {'Peak GB':<10} {'Time':<8} {'Throughput':<12} {'Status'}")
print("-" * 80)

successful_results = [r for r in results if r['success']]
for r in results:
    if r['success']:
        status = "✅"
        print(f"{r['batch_size']:<8} {r['total_points']:<12,} {r['input_size_gb']:<10.3f} {r['peak_memory']:<10.2f} {r['gpu_time']:<8.3f} {r['throughput']:<12.1f} {status}")
    else:
        status = "❌"
        print(f"{r['batch_size']:<8} {r['total_points']:<12,} {r['input_size_gb']:<10.3f} {r.get('memory_used', 0):<10.2f} {'N/A':<8} {'N/A':<12} {status}")

if successful_results:
    print(f"\n🎯 A100 Performance Summary:")
    print(f"  Maximum successful batch: {max_successful_batch}")
    peak_throughput = max(r['throughput'] for r in successful_results)
    peak_memory = max(r['peak_memory'] for r in successful_results) 
    print(f"  Peak throughput: {peak_throughput:.1f} M points/sec")
    print(f"  Maximum memory used: {peak_memory:.1f} GB / 80 GB ({peak_memory/80*100:.1f}%)")
    
    largest_batch = successful_results[-1]
    print(f"\n🚀 Largest successful batch:")
    print(f"  {largest_batch['batch_size']} samples processing {largest_batch['total_points']:,} points")
    print(f"  Real-world equivalent: {largest_batch['batch_size']} subjects analyzed simultaneously")
    print(f"  Memory efficiency: {largest_batch['peak_memory']:.1f} GB used of 80 GB available")

print(f"\nConclusion: A100 enables true large-scale neuroscience batch processing!")