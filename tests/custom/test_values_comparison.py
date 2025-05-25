#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ywatanabe/proj/mngs_repo/src')

import mngs
import gpac
import numpy as np
import torch

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate demo signal
xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
pha_n_bands = 50
amp_n_bands = 30

print("Input shape:", xx.shape)
print("Using xx[:2, :2] for comparison")

# Test with mngs
pac, pha_mids_hz, amp_mids_hz = mngs.dsp.pac(
    xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
)

# Test with gpac
pac_g, pha_mids_hz_g, amp_mids_hz_g = gpac.calculate_pac(
    xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
)

print("Shapes:")
print("  mngs:", pac.shape)
print("  gpac:", pac_g.shape)

print("\nFrequency arrays:")
print("  mngs pha_mids_hz shape:", pha_mids_hz.shape)
print("  gpac pha_mids_hz shape:", pha_mids_hz_g.shape)
print("  mngs amp_mids_hz shape:", amp_mids_hz.shape)
print("  gpac amp_mids_hz shape:", amp_mids_hz_g.shape)

print("\nFrequency ranges:")
print("  mngs pha range:", pha_mids_hz.min(), "to", pha_mids_hz.max())
print("  gpac pha range:", pha_mids_hz_g.min(), "to", pha_mids_hz_g.max())
print("  mngs amp range:", amp_mids_hz.min(), "to", amp_mids_hz.max())
print("  gpac amp range:", amp_mids_hz_g.min(), "to", amp_mids_hz_g.max())

# Convert to numpy for easier comparison
if isinstance(pac_g, torch.Tensor):
    pac_g_np = pac_g.detach().cpu().numpy()
else:
    pac_g_np = pac_g

if isinstance(pac, torch.Tensor):
    pac_np = pac.detach().cpu().numpy()
else:
    pac_np = pac

print("\nValue statistics:")
print("  mngs PAC - min:", pac_np.min(), "max:", pac_np.max(), "mean:", pac_np.mean())
print("  gpac PAC - min:", pac_g_np.min(), "max:", pac_g_np.max(), "mean:", pac_g_np.mean())

# Check if values are close
print("\nValue comparison:")
if pac_np.shape == pac_g_np.shape:
    print("  Shapes match:", pac_np.shape)
    
    # Check absolute difference
    abs_diff = np.abs(pac_np - pac_g_np)
    print("  Max absolute difference:", abs_diff.max())
    print("  Mean absolute difference:", abs_diff.mean())
    
    # Check relative difference where values are not too small
    mask = np.abs(pac_np) > 1e-6
    if mask.any():
        rel_diff = np.abs((pac_np - pac_g_np) / pac_np)[mask]
        print("  Max relative difference (where |mngs| > 1e-6):", rel_diff.max())
        print("  Mean relative difference (where |mngs| > 1e-6):", rel_diff.mean())
    
    # Check if they're approximately equal
    is_close = np.allclose(pac_np, pac_g_np, rtol=1e-3, atol=1e-6)
    print("  Values are close (rtol=1e-3, atol=1e-6):", is_close)
    
    if not is_close:
        print("  WARNING: Values differ significantly!")
        # Show some example differences
        print("  Example differences at first few positions:")
        for i in range(min(2, pac_np.shape[0])):
            for j in range(min(2, pac_np.shape[1])):
                for k in range(min(3, pac_np.shape[2])):
                    for l in range(min(3, pac_np.shape[3])):
                        print(f"    [{i},{j},{k},{l}]: mngs={pac_np[i,j,k,l]:.6f}, gpac={pac_g_np[i,j,k,l]:.6f}, diff={abs_diff[i,j,k,l]:.6f}")
else:
    print("  Shapes don't match!")
    print("  mngs shape:", pac_np.shape)
    print("  gpac shape:", pac_g_np.shape)