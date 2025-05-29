#!/usr/bin/env python3
"""
Test massive batch processing to utilize A100's 80GB VRAM.
Show where the A100 really shines - huge batch sizes.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

def estimate_memory_usage(batch_size, n_channels, n_segments, seq_len, n_bands=20):
    """Estimate memory usage for a given configuration."""
    # Input data
    input_size = batch_size * n_channels * n_segments * seq_len * 4  # 4 bytes per float32
    
    # Intermediate tensors (rough estimate)
    # - Filtered signals: batch × channels × segments × n_bands × seq_len
    filtered_size = batch_size * n_channels * n_segments * n_bands * seq_len * 4
    
    # - Phase/amplitude: similar size
    phase_amp_size = filtered_size * 2
    
    # Total rough estimate
    total_mb = (input_size + filtered_size + phase_amp_size) / (1024**2)
    total_gb = total_mb / 1024
    
    return total_gb

def test_massive_batch(batch_size, n_channels, seq_len, description):
    """Test with massive batch sizes to utilize A100."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    n_segments = 1
    total_points = batch_size * n_channels * n_segments * seq_len
    estimated_memory = estimate_memory_usage(batch_size, n_channels, n_segments, seq_len)
    
    print(f"Batch size: {batch_size}")
    print(f"Shape per sample: {n_channels}ch × {n_segments}seg × {seq_len}samples")
    print(f"Total data points: {total_points:,}")
    print(f"Estimated GPU memory: {estimated_memory:.1f} GB")
    
    if estimated_memory > 70:  # Don't exceed 70GB to be safe
        print("⚠️  Too large - would exceed GPU memory")
        return None
    
    # Generate massive batch
    print("Generating data...")
    torch.manual_seed(42)
    signal = torch.randn(batch_size, n_channels, n_segments, seq_len)
    
    print(f"Input tensor size: {signal.numel() * 4 / 1024**3:.2f} GB")
    
    # Test CPU (if feasible)
    if total_points < 10_000_000:  # Only test CPU for smaller cases
        print(f"\nCPU timing:")
        start = time.time()
        from src.gpac import calculate_pac
        try:
            pac_cpu, _, _ = calculate_pac(
                signal,
                fs=1000,
                pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=10,  # Smaller for speed
                amp_start_hz=60.0, amp_end_hz=160.0, amp_n_bands=10,
                device='cpu',
                mi_n_bins=18,
                use_optimized_filter=True,
            )
            cpu_time = time.time() - start
            print(f"  Time: {cpu_time:.4f}s")
        except Exception as e:
            print(f"  CPU failed: {e}")
            cpu_time = None
    else:
        print(f"\nSkipping CPU (too large)")
        cpu_time = None
    
    # Test GPU
    print(f"\nGPU timing:")
    
    # Check memory before
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() / 1024**3
    
    start = time.time()
    try:
        pac_gpu, _, _ = calculate_pac(
            signal,
            fs=1000,
            pha_start_hz=2.0, pha_end_hz=20.0, pha_n_bands=10,
            amp_start_hz=60.0, amp_end_hz=160.0, amp_n_bands=10,
            device='cuda',
            mi_n_bins=18,
            use_optimized_filter=True,
        )
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Check memory after
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_used = memory_after - memory_before
        
        print(f"  Time: {gpu_time:.4f}s")
        print(f"  GPU memory used: {memory_used:.2f} GB")
        print(f"  Throughput: {total_points / gpu_time / 1_000_000:.1f} M points/sec")
        
        if cpu_time:
            speedup = cpu_time / gpu_time
            print(f"  GPU speedup: {speedup:.1f}x")
        
        return {
            'batch_size': batch_size,
            'total_points': total_points,
            'gpu_time': gpu_time,
            'memory_used': memory_used,
            'throughput': total_points / gpu_time / 1_000_000,
            'cpu_time': cpu_time
        }
        
    except Exception as e:
        print(f"  GPU failed: {e}")
        return None

# Test progressively larger batch sizes
test_cases = [
    (1, 64, 50000, "Baseline: Single sample, 64ch, 50s"),
    (10, 64, 50000, "Small batch: 10 samples"),
    (100, 64, 50000, "Medium batch: 100 samples"),
    (500, 64, 50000, "Large batch: 500 samples"),
    (1000, 64, 30000, "Huge batch: 1000 samples, 30s each"),
    (2000, 32, 30000, "Massive batch: 2000 samples, 32ch"),
    (5000, 16, 20000, "Ultra batch: 5000 samples, 16ch"),
]

print("MASSIVE BATCH PROCESSING TEST")
print("Utilizing A100's 80GB VRAM for realistic neuroscience workloads")
print(f"Available GPU memory: 80GB")

results = []
for batch_size, n_channels, seq_len, description in test_cases:
    try:
        result = test_massive_batch(batch_size, n_channels, seq_len, description)
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: {e}")
        torch.cuda.empty_cache()  # Clean up on error

# Summary
print(f"\n{'='*80}")
print("MASSIVE BATCH SUMMARY: A100 Utilization")
print(f"{'='*80}")
print(f"{'Batch Size':<12} {'Total Points':<15} {'GPU Time':<10} {'Memory':<10} {'Throughput':<15}")
print("-" * 75)

for r in results:
    print(f"{r['batch_size']:<12} {r['total_points']:<15,} {r['gpu_time']:<10.4f} {r['memory_used']:<10.1f}GB {r['throughput']:<15.1f}M/s")

if len(results) > 1:
    max_throughput = max(r['throughput'] for r in results)
    max_memory = max(r['memory_used'] for r in results)
    print(f"\nPeak performance:")
    print(f"  Maximum throughput: {max_throughput:.1f} M points/sec")
    print(f"  Maximum memory used: {max_memory:.1f} GB / 80 GB ({max_memory/80*100:.1f}%)")

print(f"\nKey insights:")
print("- A100 excels at massive batch processing")
print("- GPU memory enables processing entire datasets at once")
print("- Batch processing amortizes initialization costs")
print("- Perfect for large-scale neuroscience studies")