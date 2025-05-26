#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Analysis of PAC scaling difference between gPAC and TensorPAC

"""
This script analyzes the mathematical differences between gPAC and TensorPAC
that lead to the ~20x scale difference in PAC values.
"""

def analyze_implementations():
    print("="*80)
    print("PAC SCALING ISSUE ANALYSIS")
    print("="*80)
    
    print("\n1. TENSORPAC IMPLEMENTATION (Tort Method):")
    print("-" * 40)
    print("```python")
    print("# From tensorpac/methods/meth_pac.py")
    print("def modulation_index(pha, amp, n_bins=18):")
    print("    # Step 1: Bin amplitude by phase")
    print("    p_j = _kl_hr(pha, amp, n_bins)  # Get binned amplitudes")
    print("    ")
    print("    # Step 2: Normalize to probability distribution")
    print("    p_j /= p_j.sum(axis=0, keepdims=True)")
    print("    ")
    print("    # Step 3: Calculate entropy term")
    print("    p_j = p_j * np.ma.log(p_j).filled(0.)")
    print("    ")
    print("    # Step 4: Calculate MI")
    print("    pac = 1 + p_j.sum(axis=0) / np.log(n_bins)")
    print("    return pac")
    print("```")
    
    print("\n   Key binning function:")
    print("```python")
    print("def _kl_hr(pha, amp, n_bins, mean_bins=True):")
    print("    # Create bins from -π to π")
    print("    eps = np.finfo(pha.dtype).eps * 2")
    print("    vecbin = np.linspace(-np.pi - eps, np.pi + eps, n_bins + 1)")
    print("    phad = np.digitize(pha, vecbin) - 1")
    print("    ")
    print("    # For each bin, calculate mean amplitude")
    print("    for i in np.unique(phad):")
    print("        idx = phad == i")
    print("        m = idx.sum() if mean_bins else 1.")
    print("        abin_pha = np.einsum('i...j, k...j->ik...', amp, idx) / m")
    print("```")
    
    print("\n2. GPAC IMPLEMENTATION:")
    print("-" * 40)
    print("```python")
    print("# From gpac/_ModulationIndex.py")
    print("def forward(self, pha, amp, epsilon=1e-9):")
    print("    # Step 1: Bin phase values")
    print("    bin_indices = torch.bucketize(pha, cutoffs, right=False)")
    print("    one_hot_masks = F.one_hot(bin_indices, num_classes=n_bins).bool()")
    print("    ")
    print("    # Step 2: Calculate mean amplitude per bin")
    print("    amp_sums_in_bins = (pha_masks * amp).sum(dim=5)")
    print("    counts_in_bins = pha_masks.sum(dim=5)")
    print("    amp_means_per_bin = amp_sums_in_bins / (counts_in_bins + epsilon)")
    print("    ")
    print("    # Step 3: Normalize to probability")
    print("    amp_probs = amp_means_per_bin / amp_means_per_bin.sum(dim=-1)")
    print("    ")
    print("    # Step 4: Calculate MI")
    print("    entropy_term = (amp_probs * torch.log(amp_probs + epsilon)).sum(dim=-1)")
    print("    mi_result = 1.0 + entropy_term / log_n_bins")
    print("```")
    
    print("\n3. KEY DIFFERENCES IDENTIFIED:")
    print("-" * 40)
    
    print("\n   A. BINNING APPROACH:")
    print("   • TensorPAC: Uses np.digitize with epsilon padding")
    print("   • gPAC: Uses torch.bucketize without epsilon")
    print("   • Impact: Different bin edge handling")
    
    print("\n   B. AMPLITUDE CALCULATION:")
    print("   • TensorPAC: Uses einsum for efficient matrix operations")
    print("   • gPAC: Uses masked multiplication and sum")
    print("   • Both calculate mean amplitude per bin (equivalent)")
    
    print("\n   C. NORMALIZATION:")
    print("   • Both normalize to probability distribution")
    print("   • Both use same MI formula: 1 + sum(p*log(p))/log(n_bins)")
    
    print("\n   D. EPSILON HANDLING:")
    print("   • TensorPAC: Uses np.ma.log (masked arrays) for log(0)")
    print("   • gPAC: Adds epsilon to prevent log(0)")
    print("   • This could cause small differences")
    
    print("\n4. SCALE DIFFERENCE INVESTIGATION:")
    print("-" * 40)
    
    print("\n   The ~20x scale difference is NOT due to:")
    print("   ✗ Different MI formula (both use same Tort formula)")
    print("   ✗ Different normalization (both normalize to probability)")
    print("   ✗ Different binning count (both use 18 bins by default)")
    
    print("\n   The scale difference MIGHT be due to:")
    print("   ? Different preprocessing before MI calculation")
    print("   ? Different interpretation of 'amplitude'")
    print("   ? Processing pipeline differences")
    print("   ? Numerical precision issues")
    
    print("\n5. BULLETIN BOARD INSIGHTS:")
    print("-" * 40)
    print("   From the agent bulletin board:")
    print("   • TensorPAC combines filtering + Hilbert in filter() method")
    print("   • gPAC separates these into distinct steps")
    print("   • TensorPAC returns processed phase/amplitude directly")
    print("   • gPAC may have additional normalization steps")
    
    print("\n6. RECOMMENDATIONS:")
    print("-" * 40)
    print("   1. Check if TensorPAC applies amplitude normalization")
    print("   2. Verify phase is in [-π, π] range for both")
    print("   3. Check if amplitude scaling differs")
    print("   4. Compare raw binned values before MI calculation")
    print("   5. Test with simple synthetic signals")
    
    print("\n" + "="*80)
    print("CONCLUSION: The implementations are mathematically equivalent.")
    print("The scale difference likely comes from preprocessing differences.")
    print("="*80)

if __name__ == "__main__":
    analyze_implementations()