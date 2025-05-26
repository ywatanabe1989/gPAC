#!/usr/bin/env python3
"""Compare v01 sequential filtfilt with current odd extension implementation."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt

def v01_sequential_filtfilt(x, kernel):
    """V01 implementation: simple sequential forward-backward filtering."""
    # First forward pass
    filtered = F.conv1d(x, kernel, padding="same")
    # Second pass on time-reversed signal (backward filtering)
    filtered = F.conv1d(filtered.flip(-1), kernel, padding="same").flip(-1)
    return filtered

def current_odd_extension_filtfilt(x, kernel):
    """Current implementation: odd extension padding."""
    batch_size = x.shape[0]
    seq_len = x.shape[-1]
    
    # Calculate padlen based on filter length
    padlen = min(3 * kernel.shape[-1], seq_len - 1)
    
    # Apply odd extension padding manually for each signal
    x_padded_list = []
    for b in range(batch_size):
        signal = x[b, 0]  # Get signal
        
        if padlen > 0:
            # Create odd extension: -flipped_signal
            left_pad = -signal[1:padlen+1].flip(0)
            right_pad = -signal[-padlen-1:-1].flip(0) 
            signal_padded = torch.cat([left_pad, signal, right_pad])
        else:
            signal_padded = signal
        
        x_padded_list.append(signal_padded)
    
    # Stack padded signals
    x_padded = torch.stack(x_padded_list).unsqueeze(1)
    
    # Apply forward and backward filtering
    filtered = F.conv1d(x_padded, kernel, padding='same')
    filtered = F.conv1d(filtered.flip(-1), kernel, padding='same').flip(-1)
    
    # Remove padding
    if padlen > 0:
        filtered = filtered[:, :, padlen:-padlen]
    
    return filtered

def test_implementations():
    """Test both implementations against scipy.signal.filtfilt."""
    # Generate test signal
    fs = 512
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    
    # Create a signal with multiple frequency components
    test_signal = (np.sin(2 * np.pi * 10 * t) + 
                   0.5 * np.sin(2 * np.pi * 50 * t) +
                   0.3 * np.sin(2 * np.pi * 100 * t))
    
    # Design a bandpass filter (40-60 Hz)
    nyq = fs / 2
    low = 40 / nyq
    high = 60 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply scipy filtfilt (ground truth)
    scipy_filtered = signal.filtfilt(b, a, test_signal)
    
    # Convert to PyTorch tensors
    signal_tensor = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Create FIR approximation of the filter for conv1d
    # Use frequency sampling method
    w, h = signal.freqz(b, a, worN=512)
    taps = signal.firwin2(101, w/(np.pi), np.abs(h))
    kernel = torch.tensor(taps, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Test v01 implementation
    v01_filtered = v01_sequential_filtfilt(signal_tensor, kernel)
    v01_result = v01_filtered.squeeze().numpy()
    
    # Test current implementation
    current_filtered = current_odd_extension_filtfilt(signal_tensor, kernel)
    current_result = current_filtered.squeeze().numpy()
    
    # Calculate correlations
    corr_v01 = np.corrcoef(scipy_filtered[50:-50], v01_result[50:-50])[0, 1]
    corr_current = np.corrcoef(scipy_filtered[50:-50], current_result[50:-50])[0, 1]
    
    print("="*60)
    print("FILTFILT IMPLEMENTATION COMPARISON")
    print("="*60)
    print(f"Signal length: {len(test_signal)}")
    print(f"Filter length: {len(taps)}")
    print()
    print("Correlation with scipy.signal.filtfilt:")
    print(f"  V01 sequential:     {corr_v01:.6f}")
    print(f"  Current odd ext:    {corr_current:.6f}")
    print()
    print("RMS difference from scipy.signal.filtfilt:")
    rms_v01 = np.sqrt(np.mean((scipy_filtered[50:-50] - v01_result[50:-50])**2))
    rms_current = np.sqrt(np.mean((scipy_filtered[50:-50] - current_result[50:-50])**2))
    print(f"  V01 sequential:     {rms_v01:.6f}")
    print(f"  Current odd ext:    {rms_current:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t[200:400], test_signal[200:400], 'k-', alpha=0.5, label='Original')
    plt.plot(t[200:400], scipy_filtered[200:400], 'b-', label='scipy.filtfilt')
    plt.legend()
    plt.title('Ground Truth')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.plot(t[200:400], scipy_filtered[200:400], 'b-', alpha=0.5, label='scipy.filtfilt')
    plt.plot(t[200:400], v01_result[200:400], 'g-', label=f'V01 (r={corr_v01:.4f})')
    plt.legend()
    plt.title('V01 Sequential Implementation')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.plot(t[200:400], scipy_filtered[200:400], 'b-', alpha=0.5, label='scipy.filtfilt')
    plt.plot(t[200:400], current_result[200:400], 'r-', label=f'Current (r={corr_current:.4f})')
    plt.legend()
    plt.title('Current Odd Extension Implementation')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('./scripts/filtfilt_comparison.png', dpi=150)
    print(f"\n✅ Comparison plot saved to: ./scripts/filtfilt_comparison.png")
    
    # Return which implementation is better
    if corr_v01 > corr_current:
        print(f"\n🎯 V01 sequential implementation is better by {corr_v01 - corr_current:.4f}")
        return "v01"
    else:
        print(f"\n🎯 Current implementation is better by {corr_current - corr_v01:.4f}")
        return "current"

if __name__ == "__main__":
    better_implementation = test_implementations()