#!/usr/bin/env python3
"""Quick test to check PyTorch padding modes."""

import torch
import torch.nn.functional as F

# Test tensor
x = torch.tensor([1., 2., 3., 4., 5.])
print("Original tensor:", x)
print()

# Test different padding modes
modes = ['constant', 'reflect', 'replicate', 'circular']

for mode in modes:
    try:
        # Pad with 2 on each side
        padded = F.pad(x.unsqueeze(0), (2, 2), mode=mode)
        print(f"{mode:10} mode: {padded.squeeze()}")
    except Exception as e:
        print(f"{mode:10} mode: Error - {e}")

print("\nNote: PyTorch naming conventions:")
print("- 'constant' = zero padding")
print("- 'reflect' = mirror padding")
print("- 'replicate' = edge mode (repeat edge values)")
print("- 'circular' = wrap mode")