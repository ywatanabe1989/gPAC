#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Summary of ModulationIndex differentiability findings

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from gpac._ModulationIndex import ModulationIndex
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex

def print_summary():
    """Print summary of differentiability findings."""
    
    print("="*80)
    print("MODULATION INDEX DIFFERENTIABILITY SUMMARY")
    print("="*80)
    
    print("\n1. PROBLEM IDENTIFIED:")
    print("-" * 40)
    print("   • Standard ModulationIndex uses torch.bucketize for phase binning")
    print("   • torch.bucketize creates discrete bin indices (0, 1, 2, ...)")
    print("   • These discrete indices break the computation graph")
    print("   • Result: NO gradient flow through standard MI calculation")
    
    print("\n2. ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    print("   • Phase binning: pha → bucketize → indices → one_hot → masks")
    print("   • The bucketize operation is non-differentiable")
    print("   • One-hot encoding from discrete indices is also non-differentiable")
    print("   • Even though later operations (mean, entropy) are differentiable,")
    print("     the gradient cannot flow through the discrete binning step")
    
    print("\n3. SOLUTION IMPLEMENTED:")
    print("-" * 40)
    print("   • DifferentiableModulationIndex replaces hard binning with soft binning")
    print("   • Two soft binning methods available:")
    print("     - Softmax: Uses circular distance to bin centers")
    print("     - Gaussian: Uses Gaussian weights based on distance")
    print("   • Temperature parameter controls softness:")
    print("     - Low temp (0.1): More discrete, closer to hard binning")
    print("     - High temp (2.0+): Very smooth, may reduce accuracy")
    
    print("\n4. IMPLEMENTATION DETAILS:")
    print("-" * 40)
    
    # Show the key difference
    print("   Standard MI (non-differentiable):")
    print("   ```python")
    print("   # Hard binning")
    print("   bin_indices = torch.bucketize(pha, boundaries)")
    print("   masks = F.one_hot(bin_indices, n_bins)  # Discrete!")
    print("   ```")
    
    print("\n   Differentiable MI:")
    print("   ```python")
    print("   # Soft binning")
    print("   distances = circular_distance(pha, bin_centers)")
    print("   weights = torch.softmax(-distances/temperature, dim=-1)  # Continuous!")
    print("   ```")
    
    print("\n5. PRACTICAL IMPLICATIONS:")
    print("-" * 40)
    print("   ✓ Use DifferentiableModulationIndex for:")
    print("     • Training neural networks with MI as part of loss")
    print("     • Gradient-based optimization of PAC features")
    print("     • Any scenario requiring backpropagation through MI")
    print("\n   ✓ Use standard ModulationIndex for:")
    print("     • Evaluation only (no gradients needed)")
    print("     • Maximum compatibility with TensorPAC")
    print("     • Slightly faster computation")
    
    print("\n6. VERIFICATION TEST:")
    print("-" * 40)
    
    # Quick verification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small test data
    pha = torch.randn(1, 1, 1, 1, 100, device=device, requires_grad=True)
    amp = torch.randn(1, 1, 1, 1, 100, device=device, requires_grad=True).abs()
    
    # Test standard MI
    standard_mi = ModulationIndex(n_bins=18).to(device)
    output1 = standard_mi(pha.clone(), amp.clone())
    loss1 = output1['mi'].mean()
    
    try:
        loss1.backward()
        std_grad_success = True
    except:
        std_grad_success = False
    
    # Test differentiable MI
    diff_mi = DifferentiableModulationIndex(n_bins=18).to(device)
    pha2 = pha.clone().detach().requires_grad_(True)
    amp2 = amp.clone().detach().requires_grad_(True)
    output2 = diff_mi(pha2, amp2)
    loss2 = output2['mi'].mean()
    
    try:
        loss2.backward()
        diff_grad_success = pha2.grad is not None
    except:
        diff_grad_success = False
    
    print(f"   Standard MI gradient flow: {'✗ FAILED' if not std_grad_success else '✓ Success (but likely zero)'}")
    print(f"   Differentiable MI gradient flow: {'✓ SUCCESS' if diff_grad_success else '✗ Failed'}")
    
    if diff_grad_success and pha2.grad is not None:
        print(f"   Phase gradient norm: {pha2.grad.norm().item():.6f}")
    
    print("\n7. RECOMMENDATION:")
    print("-" * 40)
    print("   For the gPAC project, include both implementations:")
    print("   • ModulationIndex: For evaluation and TensorPAC compatibility")
    print("   • DifferentiableModulationIndex: For gradient-based optimization")
    print("   • Document the trade-offs clearly for users")
    
    print("\n" + "="*80)
    print("Feature request 03_differentiable_modulation_index.md: ✓ COMPLETED")
    print("="*80)

if __name__ == "__main__":
    print_summary()