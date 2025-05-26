#!/usr/bin/env python3
"""Verify the MI fix is working correctly."""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Force reload of modules to get the fix
import importlib
import gpac._ModulationIndex
importlib.reload(gpac._ModulationIndex)
from gpac._ModulationIndex import ModulationIndex

def test_mi_fix():
    """Test if MI calculation now returns TensorPAC-compatible values."""
    
    print("="*60)
    print("VERIFYING MI FIX")
    print("="*60)
    
    # Create test data
    n_samples = 1000
    n_bins = 18
    
    # Test 1: Uniform distribution (no coupling)
    print("\n1. Uniform distribution (no coupling):")
    phase_uniform = torch.linspace(-np.pi, np.pi, n_samples).reshape(1, 1, 1, 1, -1)
    amp_uniform = torch.ones(n_samples).reshape(1, 1, 1, 1, -1)
    
    mi_calc = ModulationIndex(n_bins=n_bins)
    result = mi_calc(phase_uniform, amp_uniform)
    mi_uniform = result['mi'].item()
    
    print(f"  MI value: {mi_uniform:.6f}")
    print(f"  Expected: ~2.0 (no coupling in TensorPAC)")
    print(f"  Is close to 2.0? {abs(mi_uniform - 2.0) < 0.1}")
    
    # Test 2: Perfect coupling
    print("\n2. Perfect coupling (amplitude at specific phase):")
    # Create amplitude that's high only at phase=0
    phase = torch.linspace(-np.pi, np.pi, n_samples)
    amplitude = torch.exp(-phase**2 / 0.1)  # Gaussian centered at phase=0
    
    phase_coupled = phase.reshape(1, 1, 1, 1, -1)
    amp_coupled = amplitude.reshape(1, 1, 1, 1, -1)
    
    result = mi_calc(phase_coupled, amp_coupled)
    mi_coupled = result['mi'].item()
    
    print(f"  MI value: {mi_coupled:.6f}")
    print(f"  Expected: <1.0 (strong coupling in TensorPAC)")
    print(f"  Is less than 1.0? {mi_coupled < 1.0}")
    
    # Test 3: Check the actual formula
    print("\n3. Manual calculation check:")
    
    # Get amplitude probability distribution
    amp_prob = result['amp_prob'].squeeze()
    
    # Calculate MI manually
    epsilon = 1e-10
    log_n_bins = np.log(n_bins)
    entropy_part = (amp_prob * torch.log(amp_prob + epsilon)).sum()
    mi_manual = 1.0 + entropy_part.item() / log_n_bins
    
    print(f"  MI from module: {mi_coupled:.6f}")
    print(f"  MI manual calc: {mi_manual:.6f}")
    print(f"  Match? {abs(mi_coupled - mi_manual) < 0.001}")
    
    # Check source code
    print("\n4. Checking source code:")
    import inspect
    source = inspect.getsource(mi_calc.forward)
    if "1.0 + entropy_part / log_n_bins" in source:
        print("  ✅ Fix is present in source code")
    else:
        print("  ❌ Fix NOT found in source code")
        print("  May need to restart Python or clear cache")
    
    return mi_uniform, mi_coupled

if __name__ == "__main__":
    test_mi_fix()