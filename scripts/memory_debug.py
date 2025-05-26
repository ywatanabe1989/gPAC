#!/usr/bin/env python3
"""
Debug GPU memory usage to understand why it's using 53GB for small data.
"""

import torch
import numpy as np
import sys
sys.path.append('..')

def monitor_memory(step_name):
    """Monitor GPU memory at each step."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{step_name:<30} Allocated: {allocated:>6.2f} GB, Cached: {cached:>6.2f} GB")
        return allocated

def debug_pac_memory():
    """Debug memory usage step by step."""
    print("GPU Memory Debug: Why does gPAC use 53GB?")
    print("=" * 60)
    
    # Start clean
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor_memory("Initial state")
    
    # Create small test data
    print("\n1. Creating small test data:")
    signal = torch.randn(1, 16, 1, 10000)  # Much smaller: 16ch, 10s
    input_size = signal.numel() * 4 / 1024**3
    print(f"   Input size: {input_size:.4f} GB ({signal.numel():,} points)")
    
    signal = signal.cuda()
    monitor_memory("After moving data to GPU")
    
    # Import and check
    print("\n2. Importing gPAC:")
    from src.gpac import calculate_pac
    monitor_memory("After importing gPAC")
    
    # Test step by step
    print("\n3. Running calculate_pac with minimal settings:")
    
    try:
        # Use minimal frequency bands to reduce memory
        pac, freqs_pha, freqs_amp = calculate_pac(
            signal,
            fs=1000,
            pha_start_hz=5.0, pha_end_hz=15.0, pha_n_bands=3,  # Just 3 bands!
            amp_start_hz=60.0, amp_end_hz=120.0, amp_n_bands=3,  # Just 3 bands!
            device='cuda',
            mi_n_bins=9,  # Smaller bins
            use_optimized_filter=True,
        )
        
        monitor_memory("After PAC calculation")
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak memory used: {peak_memory:.2f} GB")
        print(f"Memory amplification: {peak_memory / input_size:.0f}x")
        print(f"Output shape: {pac.shape}")
        
        return peak_memory, input_size
        
    except Exception as e:
        print(f"Error: {e}")
        monitor_memory("After error")
        return None, input_size

def test_memory_scaling():
    """Test how memory scales with different parameters."""
    print(f"\n{'='*60}")
    print("MEMORY SCALING TEST")
    print(f"{'='*60}")
    
    configs = [
        (1, 4, 5000, 2, 2, "Tiny: 1ch, 2x2 bands"),
        (1, 8, 5000, 3, 3, "Small: 8ch, 3x3 bands"),
        (1, 16, 5000, 5, 5, "Medium: 16ch, 5x5 bands"),
        (1, 16, 10000, 5, 5, "Longer: 16ch, 10s, 5x5 bands"),
    ]
    
    for n_ch, seq_len, pha_bands, amp_bands, description in configs:
        print(f"\n{description}")
        print("-" * 40)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        signal = torch.randn(1, n_ch, 1, seq_len).cuda()
        input_size = signal.numel() * 4 / 1024**3
        
        try:
            from src.gpac import calculate_pac
            pac, _, _ = calculate_pac(
                signal,
                fs=1000,
                pha_start_hz=5.0, pha_end_hz=15.0, pha_n_bands=pha_bands,
                amp_start_hz=60.0, amp_end_hz=120.0, amp_n_bands=amp_bands,
                device='cuda',
                mi_n_bins=9,
                use_optimized_filter=True,
            )
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Input: {input_size:.4f} GB, Peak: {peak_memory:.2f} GB")
            print(f"  Amplification: {peak_memory / input_size:.0f}x")
            print(f"  Frequency pairs: {pha_bands * amp_bands}")
            
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    # Debug the memory issue
    peak, input_size = debug_pac_memory()
    
    # Test scaling
    test_memory_scaling()
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print("The high memory usage is likely due to:")
    print("1. Filter convolutions creating large intermediate tensors")
    print("2. Frequency domain operations requiring padding")
    print("3. Multiple frequency bands processed simultaneously")
    print("4. PyTorch's memory allocation strategy")
    print("\nTo better utilize 80GB VRAM:")
    print("- Use chunked processing for very large batches")
    print("- Implement gradient checkpointing for training")
    print("- Process frequency bands sequentially rather than in parallel")