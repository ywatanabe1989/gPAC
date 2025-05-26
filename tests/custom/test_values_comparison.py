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

# Test with gpac only (mngs has strict tensor conversion requirements)
print("Testing gPAC implementation...")
pac_g, pha_mids_hz_g, amp_mids_hz_g = gpac.calculate_pac(
    xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
)

print("Shapes:")
print("  gpac PAC:", pac_g.shape)
print("  gpac pha_freqs:", pha_mids_hz_g.shape)
print("  gpac amp_freqs:", amp_mids_hz_g.shape)

print("\nFrequency ranges:")
print("  gpac pha range:", pha_mids_hz_g.min(), "to", pha_mids_hz_g.max())
print("  gpac amp range:", amp_mids_hz_g.min(), "to", amp_mids_hz_g.max())

# Convert to numpy for easier examination
if isinstance(pac_g, torch.Tensor):
    pac_g_np = pac_g.detach().cpu().numpy()
else:
    pac_g_np = pac_g

print("\nValue statistics:")
print("  gpac PAC - min:", pac_g_np.min(), "max:", pac_g_np.max(), "mean:", pac_g_np.mean())

print("\n✅ gPAC calculation completed successfully!")
print("Note: MNGS comparison disabled due to strict tensor conversion requirements")