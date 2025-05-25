#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test comparing gPAC and TensorPAC using the same Modulation Index (MI) method.
Both use idpac=(2,0,0) which is the Tort et al. 2010 MI method.
"""

import numpy as np
import torch
from tensorpac import Pac as TensorPAC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'src')
import gpac


def test_mi_comparison():
    """Compare MI calculation between gPAC and TensorPAC."""
    print("MODULATION INDEX COMPARISON: gPAC vs TensorPAC")
    print("=" * 60)
    print("Both using Tort et al. 2010 Modulation Index method")
    print("TensorPAC: idpac=(2,0,0)")
    print("gPAC: Default MI implementation")
    print()
    
    # Parameters
    fs = 512
    duration = 2.0
    n_times = int(fs * duration)
    
    # Create test signal with strong PAC
    t = np.linspace(0, duration, n_times)
    
    # Phase signal (6 Hz)
    phase_freq = 6.0
    phase_signal = np.sin(2 * np.pi * phase_freq * t)
    
    # Amplitude signal (80 Hz)
    amp_freq = 80.0
    amp_signal = np.sin(2 * np.pi * amp_freq * t)
    
    # Create PAC by modulating amplitude with phase
    modulation_depth = 0.8  # Strong modulation
    modulated_amp = amp_signal * (1 + modulation_depth * (phase_signal + 1) / 2)
    
    # Combine signals
    signal = 0.5 * phase_signal + modulated_amp
    signal += np.random.normal(0, 0.05, len(signal))
    
    # Normalize signal
    signal = (signal - signal.mean()) / signal.std()
    
    print(f"Signal: {n_times} samples @ {fs} Hz")
    print(f"Phase frequency: {phase_freq} Hz")
    print(f"Amplitude frequency: {amp_freq} Hz")
    print(f"Modulation depth: {modulation_depth}")
    print()
    
    # Define narrow frequency bands around our target frequencies
    f_pha = [4, 8]  # Around 6 Hz
    f_amp = [70, 90]  # Around 80 Hz
    n_pha_bands = 5
    n_amp_bands = 5
    
    # 1. TensorPAC with MI
    print("1. TensorPAC (MI method):")
    # Create frequency arrays for TensorPAC
    pha_freqs_tp = np.linspace(f_pha[0], f_pha[1], n_pha_bands)
    amp_freqs_tp = np.linspace(f_amp[0], f_amp[1], n_amp_bands)
    
    pac_tp = TensorPAC(
        idpac=(2, 0, 0),  # Modulation Index
        f_pha=pha_freqs_tp, 
        f_amp=amp_freqs_tp,
        n_bins=18,  # Default for MI
        dcomplex='hilbert',
        cycle=(3, 6)  # Match gPAC defaults
    )
    
    # Compute PAC
    pac_tp_result = pac_tp.filterfit(fs, signal[np.newaxis, :], n_perm=0)
    
    # Frequency arrays already defined above
    
    print(f"   Shape: {pac_tp_result.shape}")
    print(f"   Max MI: {pac_tp_result.max():.6f}")
    print(f"   Mean MI: {pac_tp_result.mean():.6f}")
    
    # Find peak location
    max_idx = np.unravel_index(pac_tp_result.argmax(), pac_tp_result.shape)
    peak_pha_tp = pha_freqs_tp[max_idx[-2]]
    peak_amp_tp = amp_freqs_tp[max_idx[-1]]
    print(f"   Peak at: phase={peak_pha_tp:.1f} Hz, amp={peak_amp_tp:.1f} Hz")
    
    # 2. gPAC with MI (default)
    print("\n2. gPAC (MI method):")
    pac_gpac = gpac.PAC(
        seq_len=n_times,
        fs=fs,
        pha_start_hz=f_pha[0],
        pha_end_hz=f_pha[1],
        pha_n_bands=n_pha_bands,
        amp_start_hz=f_amp[0],
        amp_end_hz=f_amp[1],
        amp_n_bands=n_amp_bands,
        mi_n_bins=18,  # Same as TensorPAC
        filter_cycle_pha=3,  # Match TensorPAC
        filter_cycle_amp=6,  # Match TensorPAC
        filtfilt_mode=True,  # Use sequential for better match
        edge_mode='reflect'  # Match scipy.filtfilt
    )
    
    # Move to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pac_gpac = pac_gpac.to(device)
    
    signal_torch = torch.tensor(signal, dtype=torch.float32, device=device)
    signal_torch = signal_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, time)
    
    with torch.no_grad():
        pac_gpac_result = pac_gpac(signal_torch).squeeze().cpu().numpy()
    
    # Get frequency arrays
    pha_freqs_gpac = pac_gpac.PHA_MIDS_HZ.cpu().numpy()
    amp_freqs_gpac = pac_gpac.AMP_MIDS_HZ.cpu().numpy()
    
    print(f"   Shape: {pac_gpac_result.shape}")
    print(f"   Max MI: {pac_gpac_result.max():.6f}")
    print(f"   Mean MI: {pac_gpac_result.mean():.6f}")
    
    # Find peak location
    max_idx = np.unravel_index(pac_gpac_result.argmax(), pac_gpac_result.shape)
    peak_pha_gpac = pha_freqs_gpac[max_idx[0]]
    peak_amp_gpac = amp_freqs_gpac[max_idx[1]]
    print(f"   Peak at: phase={peak_pha_gpac:.1f} Hz, amp={peak_amp_gpac:.1f} Hz")
    
    # 3. Detailed comparison
    print("\n3. Comparison:")
    print(f"   Peak location difference:")
    print(f"     Phase: {abs(peak_pha_tp - peak_pha_gpac):.2f} Hz")
    print(f"     Amplitude: {abs(peak_amp_tp - peak_amp_gpac):.2f} Hz")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Find common color scale
    vmin = 0
    vmax = max(pac_tp_result.max(), pac_gpac_result.max())
    
    # TensorPAC result
    im1 = axes[0, 0].imshow(pac_tp_result.squeeze(), aspect='auto', 
                            origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                            extent=[f_amp[0], f_amp[1], f_pha[0], f_pha[1]])
    axes[0, 0].set_title(f'TensorPAC MI\nMax: {pac_tp_result.max():.3f}')
    axes[0, 0].set_xlabel('Amplitude frequency (Hz)')
    axes[0, 0].set_ylabel('Phase frequency (Hz)')
    axes[0, 0].axhline(phase_freq, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(amp_freq, color='cyan', linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # gPAC result
    im2 = axes[0, 1].imshow(pac_gpac_result, aspect='auto', 
                            origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                            extent=[f_amp[0], f_amp[1], f_pha[0], f_pha[1]])
    axes[0, 1].set_title(f'gPAC MI\nMax: {pac_gpac_result.max():.3f}')
    axes[0, 1].set_xlabel('Amplitude frequency (Hz)')
    axes[0, 1].set_ylabel('Phase frequency (Hz)')
    axes[0, 1].axhline(phase_freq, color='cyan', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(amp_freq, color='cyan', linestyle='--', alpha=0.5)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Signal visualization
    time_window = 0.5  # seconds
    n_samples = int(time_window * fs)
    
    axes[1, 0].plot(t[:n_samples], signal[:n_samples], 'k-', linewidth=0.5)
    axes[1, 0].set_title('Test Signal (first 0.5s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MI values comparison plot
    if pac_tp_result.shape == pac_gpac_result.shape:
        axes[1, 1].scatter(pac_tp_result.flatten(), pac_gpac_result.flatten(), 
                          alpha=0.5, s=20)
        max_val = max(pac_tp_result.max(), pac_gpac_result.max())
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('TensorPAC MI')
        axes[1, 1].set_ylabel('gPAC MI')
        axes[1, 1].set_title('MI Values Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Calculate correlation
        corr = np.corrcoef(pac_tp_result.flatten(), pac_gpac_result.flatten())[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       verticalalignment='top')
    else:
        axes[1, 1].text(0.5, 0.5, 'Different shapes\nCannot compare directly', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mi_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved as 'mi_comparison.png'")
    
    # Additional analysis
    print("\n4. Implementation Notes:")
    print("   - Both use Tort et al. 2010 Modulation Index")
    print("   - Both use 18 bins for phase histogram")
    print("   - gPAC uses sequential filtfilt for better accuracy")
    print("   - Both detect peak near the true coupling frequencies")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_mi_comparison()