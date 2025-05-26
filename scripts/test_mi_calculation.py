#!/usr/bin/env python3
"""Test Modulation Index calculation to identify scaling issue."""

import numpy as np
import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def test_mi_calculation():
    """Compare MI calculation methods."""
    
    # Create synthetic phase and amplitude data
    n_samples = 1000
    n_bins = 18
    
    # Generate phase: uniform distribution from -pi to pi
    phase = torch.linspace(-np.pi, np.pi, n_samples)
    
    # Generate amplitude with phase-dependent modulation
    # Strong coupling: amplitude higher at phase=0
    base_amplitude = 1.0
    modulation_depth = 0.8
    amplitude = base_amplitude * (1 + modulation_depth * torch.cos(phase))
    
    print("="*60)
    print("MODULATION INDEX CALCULATION TEST")
    print("="*60)
    print(f"Samples: {n_samples}")
    print(f"Bins: {n_bins}")
    print(f"Amplitude range: [{amplitude.min():.3f}, {amplitude.max():.3f}]")
    
    # Method 1: gPAC-style binning
    print("\n1. gPAC-style MI calculation:")
    
    # Compute phase bins
    phase_bin_edges = torch.linspace(-np.pi, np.pi, n_bins + 1)
    phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2
    
    # Assign phases to bins
    phase_bins = torch.searchsorted(phase_bin_edges[:-1], phase, right=False)
    phase_bins = torch.clamp(phase_bins, 0, n_bins - 1)
    
    # Calculate mean amplitude per phase bin
    amp_per_bin = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = phase_bins == i
        if mask.any():
            amp_per_bin[i] = amplitude[mask].mean()
    
    # Normalize to probability distribution
    amp_prob_gpac = amp_per_bin / amp_per_bin.sum()
    
    # Calculate MI (gPAC style)
    epsilon = 1e-10
    log_n_bins = torch.log(torch.tensor(n_bins, dtype=torch.float32))
    kl_div = (amp_prob_gpac * torch.log(amp_prob_gpac + epsilon)).sum()
    mi_gpac = (log_n_bins + kl_div) / log_n_bins
    
    print(f"  Amplitude per bin (first 5): {amp_per_bin[:5].numpy()}")
    print(f"  Probability sum: {amp_prob_gpac.sum():.6f}")
    print(f"  MI value: {mi_gpac:.6f}")
    
    # Method 2: TensorPAC-style calculation
    print("\n2. TensorPAC-style MI calculation:")
    
    # Create one-hot encoding for phase bins
    idx = torch.zeros((n_bins, n_samples))
    for i in range(n_samples):
        idx[phase_bins[i], i] = 1
    
    # Calculate mean amplitude per bin (einsum style)
    m = idx.sum(dim=1)
    amp_per_bin_tp = torch.zeros(n_bins)
    for i in range(n_bins):
        if m[i] > 0:
            amp_per_bin_tp[i] = (amplitude * idx[i]).sum() / m[i]
    
    # Normalize differently - this might be the key
    amp_prob_tp = amp_per_bin_tp / amp_per_bin_tp.sum()
    
    # Calculate MI (TensorPAC style)
    p_j = amp_prob_tp.numpy()
    p_j = np.ma.masked_array(p_j, mask=(p_j == 0))
    h_p = -p_j * np.ma.log(p_j).filled(0.)
    mi_tp = 1 + h_p.sum() / np.log(n_bins)
    
    print(f"  Amplitude per bin (first 5): {amp_per_bin_tp[:5].numpy()}")
    print(f"  Probability sum: {amp_prob_tp.sum():.6f}")
    print(f"  MI value: {mi_tp:.6f}")
    
    # Compare
    print(f"\n3. Comparison:")
    print(f"  Ratio (gPAC/TensorPAC): {mi_gpac/mi_tp:.4f}")
    print(f"  Difference: {abs(mi_gpac - mi_tp):.6f}")
    
    # Test with uniform distribution (should give MI ≈ 0)
    print("\n4. Test with uniform amplitude (no coupling):")
    amplitude_uniform = torch.ones_like(amplitude)
    
    # gPAC style
    amp_per_bin_uniform = torch.ones(n_bins) / n_bins
    amp_prob_uniform = amp_per_bin_uniform / amp_per_bin_uniform.sum()
    kl_div_uniform = (amp_prob_uniform * torch.log(amp_prob_uniform + epsilon)).sum()
    mi_uniform_gpac = (log_n_bins + kl_div_uniform) / log_n_bins
    
    # TensorPAC style  
    p_j_uniform = np.ones(n_bins) / n_bins
    h_p_uniform = -p_j_uniform * np.log(p_j_uniform)
    mi_uniform_tp = 1 + h_p_uniform.sum() / np.log(n_bins)
    
    print(f"  gPAC MI (uniform): {mi_uniform_gpac:.6f}")
    print(f"  TensorPAC MI (uniform): {mi_uniform_tp:.6f}")
    
    # The issue might be in amplitude extraction
    print("\n5. Key insight:")
    print("  Both methods should give identical results for same input")
    print("  The 22x difference suggests amplitude extraction differs")
    print("  Check Hilbert transform amplitude calculation!")
    
    return mi_gpac, mi_tp

if __name__ == "__main__":
    test_mi_calculation()