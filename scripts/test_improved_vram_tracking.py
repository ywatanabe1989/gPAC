#!/usr/bin/env python3
"""Test script to verify improved VRAM tracking in the Profiler."""

import torch
import numpy as np
from gpac._Profiler import Profiler

def test_vram_tracking():
    """Test VRAM tracking with PyTorch operations."""
    
    # Create profiler
    profiler = Profiler(enable_gpu=True)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available. Cannot test VRAM tracking.")
        return
    
    print("Testing improved VRAM tracking...\n")
    
    # Profile different operations
    with profiler.profile("Small Tensor Creation"):
        x = torch.randn(1000, 1000, device='cuda')
        
    with profiler.profile("Large Tensor Creation"):
        y = torch.randn(5000, 5000, device='cuda')
        
    with profiler.profile("Matrix Multiplication"):
        z = torch.matmul(y, y)
        
    with profiler.profile("Memory Intensive Operation"):
        # Create multiple tensors
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1000, 1000, device='cuda'))
        
        # Perform operations
        result = torch.zeros_like(tensors[0])
        for t in tensors:
            result += t
            
    # Clear some memory
    del x, y, z, tensors, result
    
    with profiler.profile("After Cleanup"):
        # Force garbage collection
        torch.cuda.empty_cache()
        # Small operation to measure memory after cleanup
        final = torch.randn(100, 100, device='cuda')
    
    # Print summary
    profiler.print_summary()
    
    # Get dictionary summary
    summary = profiler.get_summary_dict()
    print("\n📊 Dictionary Summary:")
    print(f"Peak VRAM Allocated: {summary.get('peak_vram_allocated_gb', 'N/A'):.3f} GB")
    print(f"Peak VRAM Reserved: {summary.get('peak_vram_reserved_gb', 'N/A'):.3f} GB")
    
    # Show memory fragmentation info
    print("\n💡 Memory Info:")
    print("- 'Allocated' memory is actually used by tensors")
    print("- 'Reserved' memory is cached by PyTorch for future allocations")
    print("- The difference shows PyTorch's memory caching behavior")

if __name__ == "__main__":
    test_vram_tracking()