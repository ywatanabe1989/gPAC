#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ywatanabe/proj/mngs_repo/src')

import mngs
import gpac

xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
pha_n_bands = 50
amp_n_bands = 30

pac, pha_mids_hz, amp_mids_hz = mngs.dsp.pac(
    xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
)

pac_g, pha_mids_hz_g, amp_mids_hz_g = gpac.calculate_pac(
    xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
)

print("mngs shape:", pac.shape)
print("gpac shape:", pac_g.shape)
assert pac.shape == pac_g.shape
print("✓ Shape assertion passed!")