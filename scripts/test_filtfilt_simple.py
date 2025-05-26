#!/usr/bin/env python3
"""Test v01 filtfilt approach to improve TensorPAC correlation."""

import sys
import numpy as np
import torch
import torch.nn.functional as F

# Add project path
sys.path.insert(0, '/home/ywatanabe/proj/gPAC/src')

from gpac._tensorpac_fir1 import design_filter_tensorpac

def test_filtfilt_approaches():
    """Compare filtfilt implementations."""
    
    # Parameters
    fs = 512
    seq_len = 1024
    
    # Generate test signal
    t = np.arange(seq_len) / fs
    test_signal = (np.sin(2 * np.pi * 10 * t) + 
                   0.5 * np.sin(2 * np.pi * 50 * t) +
                   0.3 * np.sin(2 * np.pi * 100 * t))
    
    # Design filter for 40-60 Hz band
    kernel = design_filter_tensorpac(seq_len, fs, low_hz=40, high_hz=60, cycle=6)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    signal_tensor = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    print("="*60)
    print("TESTING FILTFILT APPROACHES")
    print("="*60)
    print(f"Signal length: {seq_len}")
    print(f"Filter length: {len(kernel)}")
    print(f"Filter band: 40-60 Hz")
    
    # V01 Sequential approach (simple)
    print("\n1. V01 Sequential Approach:")
    v01_filtered = F.conv1d(signal_tensor, kernel_tensor, padding="same")
    v01_filtered = F.conv1d(v01_filtered.flip(-1), kernel_tensor, padding="same").flip(-1)
    print(f"   Output shape: {v01_filtered.shape}")
    print(f"   Output range: [{v01_filtered.min():.4f}, {v01_filtered.max():.4f}]")
    
    # Current approach (complex odd extension)
    print("\n2. Current Odd Extension Approach:")
    # This is simplified - the real implementation is more complex
    padlen = min(3 * len(kernel), seq_len - 1)
    print(f"   Padlen: {padlen}")
    
    # Extract signal
    sig = signal_tensor[0, 0]
    
    # Odd extension padding
    left_pad = -sig[1:padlen+1].flip(0)
    right_pad = -sig[-padlen-1:-1].flip(0)
    padded = torch.cat([left_pad, sig, right_pad])
    padded_tensor = padded.unsqueeze(0).unsqueeze(0)
    
    # Apply filtering
    current_filtered = F.conv1d(padded_tensor, kernel_tensor, padding='same')
    current_filtered = F.conv1d(current_filtered.flip(-1), kernel_tensor, padding='same').flip(-1)
    
    # Remove padding
    current_filtered = current_filtered[:, :, padlen:-padlen]
    print(f"   Output shape: {current_filtered.shape}")
    print(f"   Output range: [{current_filtered.min():.4f}, {current_filtered.max():.4f}]")
    
    # Compare results
    v01_result = v01_filtered.squeeze().numpy()
    current_result = current_filtered.squeeze().numpy()
    
    # Calculate correlation
    correlation = np.corrcoef(v01_result, current_result)[0, 1]
    rms_diff = np.sqrt(np.mean((v01_result - current_result)**2))
    
    print(f"\n3. Comparison:")
    print(f"   Correlation between methods: {correlation:.6f}")
    print(f"   RMS difference: {rms_diff:.6f}")
    
    # Energy comparison
    v01_energy = np.sum(v01_result**2)
    current_energy = np.sum(current_result**2)
    print(f"\n4. Signal Energy:")
    print(f"   V01:     {v01_energy:.2f}")
    print(f"   Current: {current_energy:.2f}")
    print(f"   Ratio:   {v01_energy/current_energy:.4f}")
    
    print("\n" + "="*60)
    if correlation > 0.99:
        print("✅ Both methods produce very similar results")
    else:
        print("⚠️  Methods differ significantly - this could affect PAC correlation")
    
    return correlation

if __name__ == "__main__":
    test_filtfilt_approaches()