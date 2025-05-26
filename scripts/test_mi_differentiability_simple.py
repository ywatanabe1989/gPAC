#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test ModulationIndex differentiability

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from gpac._ModulationIndex import ModulationIndex
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex

def test_gradient_flow():
    """Test if gradients flow through ModulationIndex."""
    print("Testing gradient flow through ModulationIndex implementations\n")
    
    # Create synthetic data
    batch_size, n_channels = 2, 3
    n_pha_freq, n_amp_freq = 4, 5
    n_segments, segment_length = 10, 100
    
    # Test standard ModulationIndex
    print("1. Testing Standard ModulationIndex:")
    standard_mi = ModulationIndex(n_bins=18)
    
    # Generate fresh phase and amplitude for this test
    pha = torch.randn(batch_size, n_channels, n_pha_freq, n_segments, segment_length, requires_grad=True)
    amp = torch.randn(batch_size, n_channels, n_amp_freq, n_segments, segment_length, requires_grad=True).abs()
    
    try:
        output = standard_mi(pha, amp)
        mi_values = output['mi'] if isinstance(output, dict) else output
        loss = mi_values.mean()
        loss.backward()
        
        # Check if gradients exist
        if pha.grad is not None and amp.grad is not None:
            print("   ✓ Gradients computed successfully")
            print(f"   - Phase gradient norm: {pha.grad.norm().item():.6f}")
            print(f"   - Amplitude gradient norm: {amp.grad.norm().item():.6f}")
            
            # Check if gradients are non-zero
            if pha.grad.abs().max() > 1e-8 or amp.grad.abs().max() > 1e-8:
                print("   ✓ Non-zero gradients detected")
            else:
                print("   ✗ Gradients are effectively zero")
        else:
            print("   ✗ No gradients computed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test differentiable ModulationIndex
    print("\n2. Testing Differentiable ModulationIndex:")
    diff_mi = DifferentiableModulationIndex(n_bins=18, temperature=1.0)
    
    # Generate fresh phase and amplitude for this test
    pha = torch.randn(batch_size, n_channels, n_pha_freq, n_segments, segment_length, requires_grad=True)
    amp = torch.randn(batch_size, n_channels, n_amp_freq, n_segments, segment_length, requires_grad=True).abs()
    
    try:
        output = diff_mi(pha, amp)
        mi_values = output['mi']
        loss = mi_values.mean()
        loss.backward()
        
        if pha.grad is not None and amp.grad is not None:
            print("   ✓ Gradients computed successfully")
            print(f"   - Phase gradient norm: {pha.grad.norm().item():.6f}")
            print(f"   - Amplitude gradient norm: {amp.grad.norm().item():.6f}")
            
            if pha.grad.abs().max() > 1e-8 and amp.grad.abs().max() > 1e-8:
                print("   ✓ Non-zero gradients detected")
            else:
                print("   ✗ Gradients are effectively zero")
        else:
            print("   ✗ No gradients computed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test bucketize operation directly
    print("\n3. Testing torch.bucketize directly:")
    x = torch.tensor([0.1, 0.5, 1.5, 2.5], requires_grad=True)
    boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0])
    
    indices = torch.bucketize(x, boundaries)
    one_hot = nn.functional.one_hot(indices, num_classes=4).float()
    loss = one_hot.sum()
    
    try:
        loss.backward()
        print("   ✓ Backward pass succeeded")
        if x.grad is not None:
            print(f"   Gradient: {x.grad}")
        else:
            print("   ✗ No gradient computed")
    except Exception as e:
        print(f"   ✗ Bucketize blocks gradients: {e}")

if __name__ == "__main__":
    print("="*60)
    print("ModulationIndex Differentiability Test")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    test_gradient_flow()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("- Standard MI appears to compute gradients but they may be zero")
    print("- This is because bucketize creates discrete indices")
    print("- DifferentiableModulationIndex uses soft binning instead")
    print("- Soft binning preserves gradient flow for training")
    print("="*60)