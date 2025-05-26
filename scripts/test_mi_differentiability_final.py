#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test ModulationIndex differentiability - Final version

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from gpac._ModulationIndex import ModulationIndex
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex

def test_gradient_flow():
    """Test if gradients flow through ModulationIndex."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test dimensions
    batch_size, n_channels = 2, 3
    n_pha_freq, n_amp_freq = 4, 5
    n_segments, segment_length = 10, 100
    
    print("1. Testing Standard ModulationIndex:")
    print("-" * 40)
    
    # Create data on correct device
    pha = torch.randn(batch_size, n_channels, n_pha_freq, n_segments, segment_length, 
                      device=device, requires_grad=True)
    amp = torch.randn(batch_size, n_channels, n_amp_freq, n_segments, segment_length, 
                      device=device, requires_grad=True).abs()
    
    standard_mi = ModulationIndex(n_bins=18).to(device)
    output = standard_mi(pha, amp)
    mi_values = output['mi']
    loss = mi_values.mean()
    
    # Check gradient computation
    loss.backward()
    
    print(f"   MI mean: {mi_values.mean().item():.6f}")
    print(f"   MI std: {mi_values.std().item():.6f}")
    
    if pha.grad is not None:
        print(f"   Phase gradient norm: {pha.grad.norm().item():.6f}")
        print(f"   Phase gradient max: {pha.grad.abs().max().item():.6f}")
    else:
        print("   Phase gradient: None")
        
    if amp.grad is not None:
        print(f"   Amplitude gradient norm: {amp.grad.norm().item():.6f}")
        print(f"   Amplitude gradient max: {amp.grad.abs().max().item():.6f}")
    else:
        print("   Amplitude gradient: None")
    
    # Analyze gradient issue
    if pha.grad is None or amp.grad is None:
        print("\n   ⚠️  No gradients computed - likely due to discrete operations")
    elif pha.grad.abs().max() < 1e-8 and amp.grad.abs().max() < 1e-8:
        print("\n   ⚠️  Gradients are zero - discrete binning blocks gradient flow")
    else:
        print("\n   ✓ Non-zero gradients detected")
    
    print("\n2. Testing Differentiable ModulationIndex:")
    print("-" * 40)
    
    # Create fresh data
    pha_diff = torch.randn(batch_size, n_channels, n_pha_freq, n_segments, segment_length, 
                           device=device, requires_grad=True)
    amp_diff = torch.randn(batch_size, n_channels, n_amp_freq, n_segments, segment_length, 
                           device=device, requires_grad=True).abs()
    
    diff_mi = DifferentiableModulationIndex(n_bins=18, temperature=1.0).to(device)
    output_diff = diff_mi(pha_diff, amp_diff)
    mi_values_diff = output_diff['mi']
    loss_diff = mi_values_diff.mean()
    
    loss_diff.backward()
    
    print(f"   MI mean: {mi_values_diff.mean().item():.6f}")
    print(f"   MI std: {mi_values_diff.std().item():.6f}")
    
    if pha_diff.grad is not None:
        print(f"   Phase gradient norm: {pha_diff.grad.norm().item():.6f}")
        print(f"   Phase gradient max: {pha_diff.grad.abs().max().item():.6f}")
    else:
        print("   Phase gradient: None")
        
    if amp_diff.grad is not None:
        print(f"   Amplitude gradient norm: {amp_diff.grad.norm().item():.6f}")
        print(f"   Amplitude gradient max: {amp_diff.grad.abs().max().item():.6f}")
    else:
        print("   Amplitude gradient: None")
    
    if pha_diff.grad is not None and amp_diff.grad is not None and \
       (pha_diff.grad.abs().max() > 1e-8 or amp_diff.grad.abs().max() > 1e-8):
        print("\n   ✓ Non-zero gradients detected - soft binning preserves gradient flow!")
    else:
        print("\n   ⚠️  Gradient issue detected")
    
    # Compare MI values
    print(f"\n3. Comparing MI values:")
    print("-" * 40)
    print(f"   Standard MI mean: {mi_values.mean().item():.6f}")
    print(f"   Differentiable MI mean: {mi_values_diff.mean().item():.6f}")
    print(f"   Difference: {abs(mi_values.mean().item() - mi_values_diff.mean().item()):.6f}")

def test_direct_bucketize():
    """Directly test bucketize gradient blocking."""
    print("\n4. Direct test of torch.bucketize:")
    print("-" * 40)
    
    x = torch.tensor([0.1, 0.5, 1.5, 2.5], requires_grad=True)
    boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0])
    
    # Bucketize operation
    indices = torch.bucketize(x, boundaries)
    print(f"   Input: {x}")
    print(f"   Boundaries: {boundaries}")
    print(f"   Indices: {indices}")
    print(f"   Indices requires_grad: {indices.requires_grad}")
    
    # Try to backprop through one-hot encoding
    one_hot = nn.functional.one_hot(indices, num_classes=5).float()
    loss = one_hot.sum()
    
    try:
        loss.backward()
        print("   ✓ Backward succeeded")
        if x.grad is not None:
            print(f"   x.grad: {x.grad}")
        else:
            print("   ✗ x.grad is None")
    except RuntimeError as e:
        print(f"   ✗ Backward failed: {str(e)[:100]}...")
    
    print("\n   Conclusion: torch.bucketize produces discrete indices that")
    print("   break the computation graph, preventing gradient flow.")

if __name__ == "__main__":
    print("="*60)
    print("ModulationIndex Differentiability Analysis")
    print("="*60 + "\n")
    
    test_gradient_flow()
    test_direct_bucketize()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("\n1. Standard ModulationIndex uses torch.bucketize which creates")
    print("   discrete bin indices, breaking gradient flow.")
    print("\n2. DifferentiableModulationIndex uses soft binning (softmax or")
    print("   gaussian weights) to maintain differentiability.")
    print("\n3. For gradient-based optimization (training), use the")
    print("   DifferentiableModulationIndex implementation.")
    print("\n4. For evaluation only, either implementation works, but")
    print("   standard MI may be slightly faster.")
    print("="*60)