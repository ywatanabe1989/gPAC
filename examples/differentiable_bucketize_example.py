#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using Differentiable Bucketize for Gradient-Based Optimization

This example demonstrates how to use the differentiable bucketize functions
to maintain gradient flow through discretization operations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add the source directory to path
import sys
sys.path.append('../src')

from gpac._differentiable_bucketize import (
    differentiable_bucketize,
    differentiable_bucketize_indices,
    differentiable_phase_binning,
)


def example_1_basic_usage():
    """Example 1: Basic usage of differentiable bucketize."""
    print("Example 1: Basic Differentiable Bucketize")
    print("-" * 50)
    
    # Create some data and boundaries
    x = torch.tensor([0.5, 1.2, 2.7, 3.1, 4.5], requires_grad=True)
    boundaries = torch.tensor([0., 1., 2., 3., 4., 5.])
    
    # Standard (non-differentiable) bucketize
    hard_bins = torch.bucketize(x, boundaries, right=False) - 1
    print(f"Hard bins: {hard_bins}")
    
    # Differentiable bucketize
    soft_bins = differentiable_bucketize(x, boundaries, temperature=0.1)
    soft_indices = differentiable_bucketize_indices(x, boundaries, temperature=0.1)
    
    print(f"Soft indices: {soft_indices.detach().numpy()}")
    print(f"Soft bins shape: {soft_bins.shape}")
    print(f"Soft bins (first element):\n{soft_bins[0].detach().numpy()}")
    
    # Test gradient flow
    loss = soft_indices.sum()
    loss.backward()
    print(f"\nGradients: {x.grad}")
    print()


def example_2_temperature_effect():
    """Example 2: Effect of temperature on soft binning."""
    print("Example 2: Temperature Effect on Soft Binning")
    print("-" * 50)
    
    x = torch.tensor([1.5])  # Value between bins 1 and 2
    boundaries = torch.tensor([0., 1., 2., 3.])
    
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0]
    
    plt.figure(figsize=(12, 4))
    for i, temp in enumerate(temperatures):
        soft_bins = differentiable_bucketize(x, boundaries, temperature=temp)
        
        plt.subplot(1, 5, i+1)
        plt.bar(range(3), soft_bins[0].detach().numpy())
        plt.title(f'T = {temp}')
        plt.xlabel('Bin')
        plt.ylabel('Weight')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('temperature_effect.png')
    print("Saved temperature effect visualization to 'temperature_effect.png'")
    print()


def example_3_phase_binning():
    """Example 3: Circular binning for phase data."""
    print("Example 3: Phase Binning (Circular)")
    print("-" * 50)
    
    # Phase values near -π and π (should be in similar bins)
    phases = torch.tensor([-3.14, -1.57, 0., 1.57, 3.14], requires_grad=True)
    
    # Use phase binning
    soft_bins = differentiable_phase_binning(phases, n_bins=8, temperature=0.1)
    
    print(f"Phase values: {phases.detach().numpy()}")
    print(f"Soft bins shape: {soft_bins.shape}")
    
    # Show that -π and π are in similar bins
    print(f"\nBins for -π: {soft_bins[0].detach().numpy()}")
    print(f"Bins for π: {soft_bins[4].detach().numpy()}")
    print(f"Similarity: {torch.cosine_similarity(soft_bins[0], soft_bins[4], dim=0):.4f}")
    print()


def example_4_optimization():
    """Example 4: Using differentiable bucketize in optimization."""
    print("Example 4: Optimization with Differentiable Bucketize")
    print("-" * 50)
    
    # Create a simple optimization problem:
    # Find x values that maximize entropy of the bin distribution
    
    class EntropyMaximizer(nn.Module):
        def __init__(self, n_values, boundaries):
            super().__init__()
            self.x = nn.Parameter(torch.randn(n_values))
            self.boundaries = boundaries
            
        def forward(self):
            # Get soft bin assignments
            soft_bins = differentiable_bucketize(
                self.x, self.boundaries, temperature=0.5
            )
            
            # Compute distribution over bins
            bin_probs = soft_bins.mean(dim=0)
            
            # Compute entropy
            entropy = -(bin_probs * torch.log(bin_probs + 1e-9)).sum()
            
            return entropy, self.x, bin_probs
    
    # Setup
    boundaries = torch.linspace(-3, 3, 7)  # 6 bins
    model = EntropyMaximizer(30, boundaries)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Initial state
    entropy, x, probs = model()
    print(f"Initial entropy: {entropy.item():.4f}")
    print(f"Initial bin distribution: {probs.detach().numpy()}")
    
    # Optimize
    for step in range(100):
        optimizer.zero_grad()
        entropy, x, probs = model()
        loss = -entropy  # Maximize entropy
        loss.backward()
        optimizer.step()
    
    # Final state
    print(f"\nFinal entropy: {entropy.item():.4f}")
    print(f"Final bin distribution: {probs.detach().numpy()}")
    print(f"Max possible entropy: {np.log(6):.4f}")
    print()


def example_5_comparison_methods():
    """Example 5: Compare different soft binning methods."""
    print("Example 5: Comparison of Soft Binning Methods")
    print("-" * 50)
    
    x = torch.linspace(-2, 5, 100)
    boundaries = torch.tensor([0., 1., 2., 3., 4.])
    
    methods = ['softmax', 'sigmoid', 'gaussian']
    
    plt.figure(figsize=(15, 4))
    for i, method in enumerate(methods):
        soft_bins = differentiable_bucketize(
            x, boundaries, temperature=0.3, method=method
        )
        
        plt.subplot(1, 3, i+1)
        for bin_idx in range(4):
            plt.plot(x.numpy(), soft_bins[:, bin_idx].detach().numpy(), 
                    label=f'Bin {bin_idx}')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=2, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=3, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=4, color='k', linestyle='--', alpha=0.3)
        plt.title(f'{method.capitalize()} Method')
        plt.xlabel('Input Value')
        plt.ylabel('Bin Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('binning_methods_comparison.png')
    print("Saved methods comparison to 'binning_methods_comparison.png'")
    print()


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_temperature_effect()
    example_3_phase_binning()
    example_4_optimization()
    example_5_comparison_methods()
    
    print("All examples completed!")

# EOF