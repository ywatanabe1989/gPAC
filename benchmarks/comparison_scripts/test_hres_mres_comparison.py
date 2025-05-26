#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved comparison with ground truth indicators and Hilbert transform option.
"""

import numpy as np
import torch
import tensorpac
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, 'src')
import gpac


def calc_pac_with_tensorpac(xx, fs, dcomplex='wavelet', i_batch=0, i_ch=0):
    """Calculate PAC with TensorPAC using specified complex method."""
    # Using hres/mres settings
    p = tensorpac.Pac(f_pha="hres", f_amp="mres", dcomplex=dcomplex)
    
    # Extract data
    if xx.ndim == 4:
        signal = xx[i_batch, i_ch]
        if signal.ndim == 2:
            signal = signal[0]
    else:
        signal = xx
    
    # Bandpass Filtering
    phases = p.filter(fs, signal[np.newaxis, :], ftype="phase", n_jobs=1)
    amplitudes = p.filter(fs, signal[np.newaxis, :], ftype="amplitude", n_jobs=1)
    
    # Calculate PAC using MI
    k = 2  # Modulation Index
    p.idpac = (k, 0, 0)
    xpac = p.fit(phases, amplitudes)
    pac = xpac.mean(axis=-1)
    
    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)
    
    pac = pac.T  # (amp, pha) -> (pha, amp)
    
    return phases, amplitudes, freqs_pha, freqs_amp, pac


def test_comparison_with_ground_truth():
    """Compare gPAC with TensorPAC (both wavelet and hilbert) with ground truth."""
    print("PAC COMPARISON: gPAC vs TensorPAC (wavelet & hilbert)")
    print("=" * 60)
    
    # Parameters
    fs = 512
    duration = 4.0
    n_times = int(fs * duration)
    
    # Ground truth frequencies
    true_phase_freq = 6.0   # Hz
    true_amp_freq = 80.0    # Hz
    
    # Create synthetic PAC signal
    t = np.linspace(0, duration, n_times)
    phase_signal = np.sin(2 * np.pi * true_phase_freq * t)
    amp_signal = np.sin(2 * np.pi * true_amp_freq * t)
    
    # Create PAC by modulating amplitude with phase
    modulation_depth = 0.8
    modulated_amp = amp_signal * (1 + modulation_depth * (phase_signal + 1) / 2)
    
    # Combine signals
    signal = 0.3 * phase_signal + modulated_amp
    signal += np.random.normal(0, 0.05, len(signal))
    signal = (signal - signal.mean()) / signal.std()
    
    print(f"Test signal: {n_times} samples @ {fs} Hz")
    print(f"Ground truth: phase={true_phase_freq} Hz, amplitude={true_amp_freq} Hz")
    print()
    
    # 1. TensorPAC with wavelet
    print("1. TensorPAC (wavelet)...")
    start = time.time()
    _, _, freqs_pha_tp, freqs_amp_tp, pac_tp_wavelet = calc_pac_with_tensorpac(
        signal, fs, dcomplex='wavelet'
    )
    time_tp_wavelet = time.time() - start
    print(f"   Max PAC: {pac_tp_wavelet.max():.6f}, Time: {time_tp_wavelet:.3f}s")
    
    # 2. TensorPAC with hilbert
    print("\n2. TensorPAC (hilbert)...")
    start = time.time()
    _, _, _, _, pac_tp_hilbert = calc_pac_with_tensorpac(
        signal, fs, dcomplex='hilbert'
    )
    time_tp_hilbert = time.time() - start
    print(f"   Max PAC: {pac_tp_hilbert.max():.6f}, Time: {time_tp_hilbert:.3f}s")
    
    # 3. gPAC
    print("\n3. gPAC (hilbert + filtfilt)...")
    pac_model = gpac.PAC(
        seq_len=n_times,
        fs=fs,
        pha_start_hz=2.0,
        pha_end_hz=20.0,
        pha_n_bands=50,
        amp_start_hz=60.0,
        amp_end_hz=160.0,
        amp_n_bands=30,
        mi_n_bins=18,
        filter_cycle_pha=3,
        filter_cycle_amp=6,
        filtfilt_mode=True,
        edge_mode='reflect'
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pac_model = pac_model.to(device)
    
    signal_torch = torch.tensor(signal, dtype=torch.float32, device=device)
    signal_torch = signal_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Warm-up
    with torch.no_grad():
        _ = pac_model(signal_torch)
    
    # Time computation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        pac_gpac = pac_model(signal_torch).squeeze().cpu().numpy()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_gpac = time.time() - start
    
    freqs_pha_gpac = pac_model.PHA_MIDS_HZ.cpu().numpy()
    freqs_amp_gpac = pac_model.AMP_MIDS_HZ.cpu().numpy()
    
    print(f"   Max PAC: {pac_gpac.max():.6f}, Time: {time_gpac:.3f}s")
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    # Common color scale
    vmin = 0
    vmax = max(pac_tp_wavelet.max(), pac_tp_hilbert.max(), pac_gpac.max()) * 0.8
    
    # Common axis ranges (in Hz)
    pha_extent = [freqs_pha_tp[0], freqs_pha_tp[-1]]
    amp_extent = [freqs_amp_tp[0], freqs_amp_tp[-1]]
    
    # Helper function to add ground truth lines
    def add_ground_truth(ax, true_phase, true_amp):
        ax.axvline(true_amp, color='cyan', linestyle='--', linewidth=2, 
                   label=f'True amp: {true_amp} Hz', alpha=0.8)
        ax.axhline(true_phase, color='lime', linestyle='--', linewidth=2, 
                   label=f'True phase: {true_phase} Hz', alpha=0.8)
    
    # 1. TensorPAC wavelet
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pac_tp_wavelet, aspect='auto', origin='lower', 
                     cmap='hot', vmin=vmin, vmax=vmax,
                     extent=[amp_extent[0], amp_extent[1], pha_extent[0], pha_extent[1]])
    add_ground_truth(ax1, true_phase_freq, true_amp_freq)
    ax1.set_title(f'TensorPAC (wavelet)\nMax: {pac_tp_wavelet.max():.4f}', fontsize=12)
    ax1.set_xlabel('Amplitude frequency (Hz)')
    ax1.set_ylabel('Phase frequency (Hz)')
    plt.colorbar(im1, ax=ax1, label='MI')
    
    # 2. TensorPAC hilbert
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pac_tp_hilbert, aspect='auto', origin='lower', 
                     cmap='hot', vmin=vmin, vmax=vmax,
                     extent=[amp_extent[0], amp_extent[1], pha_extent[0], pha_extent[1]])
    add_ground_truth(ax2, true_phase_freq, true_amp_freq)
    ax2.set_title(f'TensorPAC (hilbert)\nMax: {pac_tp_hilbert.max():.4f}', fontsize=12)
    ax2.set_xlabel('Amplitude frequency (Hz)')
    ax2.set_ylabel('Phase frequency (Hz)')
    plt.colorbar(im2, ax=ax2, label='MI')
    
    # 3. gPAC
    ax3 = fig.add_subplot(gs[0, 2])
    # Need to interpolate gPAC to match axis ranges
    im3 = ax3.imshow(pac_gpac, aspect='auto', origin='lower', 
                     cmap='hot', vmin=vmin, vmax=vmax,
                     extent=[freqs_amp_gpac[0], freqs_amp_gpac[-1], 
                             freqs_pha_gpac[0], freqs_pha_gpac[-1]])
    add_ground_truth(ax3, true_phase_freq, true_amp_freq)
    ax3.set_title(f'gPAC (hilbert+filtfilt)\nMax: {pac_gpac.max():.4f}', fontsize=12)
    ax3.set_xlabel('Amplitude frequency (Hz)')
    ax3.set_ylabel('Phase frequency (Hz)')
    plt.colorbar(im3, ax=ax3, label='MI')
    
    # Set consistent axis limits
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(amp_extent)
        ax.set_ylim(pha_extent)
    
    # 4. Difference maps
    ax4 = fig.add_subplot(gs[1, 0])
    diff_wavelet_hilbert = pac_tp_wavelet - pac_tp_hilbert
    im4 = ax4.imshow(diff_wavelet_hilbert, aspect='auto', origin='lower', 
                     cmap='RdBu_r', 
                     vmin=-np.abs(diff_wavelet_hilbert).max(), 
                     vmax=np.abs(diff_wavelet_hilbert).max(),
                     extent=[amp_extent[0], amp_extent[1], pha_extent[0], pha_extent[1]])
    ax4.set_title('TensorPAC:\nwavelet - hilbert', fontsize=12)
    ax4.set_xlabel('Amplitude frequency (Hz)')
    ax4.set_ylabel('Phase frequency (Hz)')
    plt.colorbar(im4, ax=ax4, label='Difference')
    
    ax5 = fig.add_subplot(gs[1, 1])
    diff_hilbert_gpac = pac_tp_hilbert - pac_gpac
    im5 = ax5.imshow(diff_hilbert_gpac, aspect='auto', origin='lower', 
                     cmap='RdBu_r',
                     vmin=-np.abs(diff_hilbert_gpac).max(), 
                     vmax=np.abs(diff_hilbert_gpac).max(),
                     extent=[amp_extent[0], amp_extent[1], pha_extent[0], pha_extent[1]])
    ax5.set_title('TensorPAC hilbert\n- gPAC', fontsize=12)
    ax5.set_xlabel('Amplitude frequency (Hz)')
    ax5.set_ylabel('Phase frequency (Hz)')
    plt.colorbar(im5, ax=ax5, label='Difference')
    
    # 5. Profiles at ground truth phase
    ax6 = fig.add_subplot(gs[1, 2])
    # Find closest indices to ground truth
    idx_phase_tp = np.argmin(np.abs(freqs_pha_tp - true_phase_freq))
    idx_phase_gpac = np.argmin(np.abs(freqs_pha_gpac - true_phase_freq))
    
    ax6.plot(freqs_amp_tp, pac_tp_wavelet[idx_phase_tp, :], 'b-', 
             label='TensorPAC wavelet', linewidth=2)
    ax6.plot(freqs_amp_tp, pac_tp_hilbert[idx_phase_tp, :], 'g-', 
             label='TensorPAC hilbert', linewidth=2)
    ax6.plot(freqs_amp_gpac, pac_gpac[idx_phase_gpac, :], 'r--', 
             label='gPAC', linewidth=2)
    ax6.axvline(true_amp_freq, color='cyan', linestyle=':', linewidth=2,
                label=f'True: {true_amp_freq} Hz')
    ax6.set_xlabel('Amplitude frequency (Hz)')
    ax6.set_ylabel('MI value')
    ax6.set_title(f'Amplitude profile at phase ≈ {true_phase_freq} Hz', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(amp_extent)
    
    # 6. Signal visualization
    ax7 = fig.add_subplot(gs[2, :2])
    time_window = 0.5
    n_samples = int(time_window * fs)
    ax7.plot(t[:n_samples], signal[:n_samples], 'k-', linewidth=0.5)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Amplitude')
    ax7.set_title('Test signal (first 0.5s)', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # 7. Performance comparison
    ax8 = fig.add_subplot(gs[2, 2])
    methods = ['TensorPAC\nwavelet', 'TensorPAC\nhilbert', 'gPAC']
    times = [time_tp_wavelet, time_tp_hilbert, time_gpac]
    colors = ['blue', 'green', 'red']
    
    bars = ax8.bar(methods, times, color=colors, alpha=0.7)
    ax8.set_ylabel('Time (seconds)')
    ax8.set_title('Computation time', fontsize=12)
    
    # Add speedup annotations
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{time_val:.3f}s', ha='center', va='bottom')
        if i > 0:
            speedup = times[i-1] / time_val
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{speedup:.1f}x', ha='center', va='center', 
                    color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hres_mres_comparison_improved.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved as 'hres_mres_comparison_improved.png'")
    
    # Calculate correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS:")
    corr_wavelet_hilbert = np.corrcoef(pac_tp_wavelet.flatten(), 
                                       pac_tp_hilbert.flatten())[0, 1]
    corr_hilbert_gpac = np.corrcoef(pac_tp_hilbert.flatten(), 
                                    pac_gpac.flatten())[0, 1]
    print(f"TensorPAC wavelet vs hilbert: {corr_wavelet_hilbert:.3f}")
    print(f"TensorPAC hilbert vs gPAC: {corr_hilbert_gpac:.3f}")
    
    print("\nSPEEDUP:")
    print(f"gPAC vs TensorPAC wavelet: {time_tp_wavelet/time_gpac:.1f}x")
    print(f"gPAC vs TensorPAC hilbert: {time_tp_hilbert/time_gpac:.1f}x")


if __name__ == "__main__":
    test_comparison_with_ground_truth()