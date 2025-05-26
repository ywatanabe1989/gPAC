#!/usr/bin/env python3
"""
Analyze filter implementation differences between gPAC and TensorPAC.
This script focuses specifically on the bandpass filter comparison without joblib.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

# Import both implementations
import gpac

# Import TensorPAC modules directly
tensorpac_path = os.path.join(os.path.dirname(__file__), '../../tensorpac_source')
if tensorpac_path not in sys.path:
    sys.path.insert(0, tensorpac_path)

from tensorpac.spectral import fir1, fir_order
from scipy.signal import filtfilt


def analyze_filter_coefficients():
    """Compare filter design between gPAC and TensorPAC."""
    
    print("="*60)
    print("FILTER COEFFICIENT COMPARISON")
    print("="*60)
    
    # Parameters
    fs = 1000.0
    seq_len = 2000
    pha_freqs = (4, 8)
    amp_freqs = (75, 85)
    cycle_pha = 3
    cycle_amp = 6
    
    # gPAC filter design (using TensorPAC method)
    print("\n1. gPAC Filter Design (TensorPAC method):")
    from gpac._tensorpac_fir1 import design_filter_tensorpac, fir_order as gpac_fir_order
    
    # Phase filter
    pha_filter_order_gpac = gpac_fir_order(fs, seq_len, pha_freqs[0], cycle=cycle_pha)
    pha_kernel_gpac = design_filter_tensorpac(
        seq_len, fs, low_hz=pha_freqs[0], high_hz=pha_freqs[1], cycle=cycle_pha
    )
    print(f"   Phase filter order: {pha_filter_order_gpac}")
    print(f"   Phase filter length: {len(pha_kernel_gpac)}")
    
    # Amplitude filter  
    amp_filter_order_gpac = gpac_fir_order(fs, seq_len, amp_freqs[0], cycle=cycle_amp)
    amp_kernel_gpac = design_filter_tensorpac(
        seq_len, fs, low_hz=amp_freqs[0], high_hz=amp_freqs[1], cycle=cycle_amp
    )
    print(f"   Amplitude filter order: {amp_filter_order_gpac}")
    print(f"   Amplitude filter length: {len(amp_kernel_gpac)}")
    
    # TensorPAC filter design
    print("\n2. TensorPAC Filter Design (direct):")
    
    # Phase filter
    pha_filter_order_tp = fir_order(fs, seq_len, pha_freqs[0], cycle=cycle_pha)
    pha_wn = np.array([pha_freqs[0], pha_freqs[1]]) / (fs / 2.0)
    pha_kernel_tp, _ = fir1(pha_filter_order_tp, pha_wn)
    print(f"   Phase filter order: {pha_filter_order_tp}")
    print(f"   Phase filter length: {len(pha_kernel_tp)}")
    
    # Amplitude filter
    amp_filter_order_tp = fir_order(fs, seq_len, amp_freqs[0], cycle=cycle_amp)
    amp_wn = np.array([amp_freqs[0], amp_freqs[1]]) / (fs / 2.0)
    amp_kernel_tp, _ = fir1(amp_filter_order_tp, amp_wn)
    print(f"   Amplitude filter order: {amp_filter_order_tp}")
    print(f"   Amplitude filter length: {len(amp_kernel_tp)}")
    
    # Compare filter coefficients
    print("\n3. Filter Coefficient Comparison:")
    
    # Convert gPAC tensors to numpy
    pha_kernel_gpac_np = pha_kernel_gpac.numpy()
    amp_kernel_gpac_np = amp_kernel_gpac.numpy()
    
    # Ensure same length for comparison
    min_pha_len = min(len(pha_kernel_gpac_np), len(pha_kernel_tp))
    min_amp_len = min(len(amp_kernel_gpac_np), len(amp_kernel_tp))
    
    pha_corr = np.corrcoef(pha_kernel_gpac_np[:min_pha_len], pha_kernel_tp[:min_pha_len])[0, 1]
    amp_corr = np.corrcoef(amp_kernel_gpac_np[:min_amp_len], amp_kernel_tp[:min_amp_len])[0, 1]
    
    print(f"   Phase filter correlation: {pha_corr:.6f}")
    print(f"   Amplitude filter correlation: {amp_corr:.6f}")
    
    # Check if filters are identical
    pha_identical = np.allclose(pha_kernel_gpac_np, pha_kernel_tp, rtol=1e-5)
    amp_identical = np.allclose(amp_kernel_gpac_np, amp_kernel_tp, rtol=1e-5)
    
    print(f"   Phase filters identical: {pha_identical}")
    print(f"   Amplitude filters identical: {amp_identical}")
    
    return {
        'gpac_pha': pha_kernel_gpac_np,
        'gpac_amp': amp_kernel_gpac_np,
        'tensorpac_pha': pha_kernel_tp,
        'tensorpac_amp': amp_kernel_tp,
        'pha_corr': pha_corr,
        'amp_corr': amp_corr
    }


def analyze_filtering_output():
    """Compare actual filtering output between implementations."""
    
    print("\n" + "="*60)
    print("FILTERING OUTPUT COMPARISON")
    print("="*60)
    
    # Generate test signal
    fs = 1000.0
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Simple sine wave
    test_signal = np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 80 * t)
    
    # Parameters
    pha_freqs = (4, 8)
    amp_freqs = (75, 85)
    
    print("\n1. Applying gPAC filtering:")
    # gPAC uses conv1d internally
    pac_model = gpac.PAC(
        seq_len=len(test_signal),
        fs=fs,
        pha_start_hz=pha_freqs[0],
        pha_end_hz=pha_freqs[1],
        pha_n_bands=1,
        amp_start_hz=amp_freqs[0], 
        amp_end_hz=amp_freqs[1],
        amp_n_bands=1,
        filtfilt_mode=True
    )
    
    # Prepare signal for gPAC
    signal_tensor = torch.from_numpy(test_signal.astype(np.float32)).reshape(1, 1, -1)
    
    with torch.no_grad():
        filtered_gpac = pac_model.bandpass_filter(signal_tensor)
        # Extract phase and amplitude bands
        pha_filtered_gpac = filtered_gpac[0, 0, 0, :].numpy()
        amp_filtered_gpac = filtered_gpac[0, 0, 1, :].numpy()
    
    print("   ✅ gPAC filtering complete")
    
    print("\n2. Applying TensorPAC filtering:")
    # Get filter coefficients
    from gpac._tensorpac_fir1 import fir_order as gpac_fir_order
    
    pha_filter_order = gpac_fir_order(fs, len(test_signal), pha_freqs[0], cycle=3)
    amp_filter_order = gpac_fir_order(fs, len(test_signal), amp_freqs[0], cycle=6)
    
    pha_wn = np.array([pha_freqs[0], pha_freqs[1]]) / (fs / 2.0)
    amp_wn = np.array([amp_freqs[0], amp_freqs[1]]) / (fs / 2.0)
    
    pha_b, pha_a = fir1(pha_filter_order, pha_wn)
    amp_b, amp_a = fir1(amp_filter_order, amp_wn)
    
    # Apply filtfilt (like TensorPAC does)
    pha_filtered_tp = filtfilt(pha_b, pha_a, test_signal, padlen=pha_filter_order)
    amp_filtered_tp = filtfilt(amp_b, amp_a, test_signal, padlen=amp_filter_order)
    
    print("   ✅ TensorPAC filtering complete")
    
    # Compare outputs
    print("\n3. Comparing filtered signals:")
    
    pha_output_corr = np.corrcoef(pha_filtered_gpac, pha_filtered_tp)[0, 1]
    amp_output_corr = np.corrcoef(amp_filtered_gpac, amp_filtered_tp)[0, 1]
    
    print(f"   Phase output correlation: {pha_output_corr:.6f}")
    print(f"   Amplitude output correlation: {amp_output_corr:.6f}")
    
    # Check RMS difference
    pha_rms_diff = np.sqrt(np.mean((pha_filtered_gpac - pha_filtered_tp)**2))
    amp_rms_diff = np.sqrt(np.mean((amp_filtered_gpac - amp_filtered_tp)**2))
    
    print(f"   Phase RMS difference: {pha_rms_diff:.6f}")
    print(f"   Amplitude RMS difference: {amp_rms_diff:.6f}")
    
    return {
        'test_signal': test_signal,
        'gpac_pha': pha_filtered_gpac,
        'gpac_amp': amp_filtered_gpac,
        'tensorpac_pha': pha_filtered_tp,
        'tensorpac_amp': amp_filtered_tp,
        'pha_corr': pha_output_corr,
        'amp_corr': amp_output_corr
    }


def plot_comparison(filter_results, output_results):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot filter coefficients
    ax = axes[0, 0]
    ax.plot(filter_results['gpac_pha'], label='gPAC', alpha=0.7)
    ax.plot(filter_results['tensorpac_pha'], label='TensorPAC', alpha=0.7)
    ax.set_title(f"Phase Filter Coefficients (r={filter_results['pha_corr']:.3f})")
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(filter_results['gpac_amp'], label='gPAC', alpha=0.7)
    ax.plot(filter_results['tensorpac_amp'], label='TensorPAC', alpha=0.7)
    ax.set_title(f"Amplitude Filter Coefficients (r={filter_results['amp_corr']:.3f})")
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    
    # Plot filtered outputs (first 500 samples)
    ax = axes[1, 0]
    ax.plot(output_results['gpac_pha'][:500], label='gPAC', alpha=0.7)
    ax.plot(output_results['tensorpac_pha'][:500], label='TensorPAC', alpha=0.7)
    ax.set_title(f"Phase Filtered Output (r={output_results['pha_corr']:.3f})")
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    
    ax = axes[1, 1]
    ax.plot(output_results['gpac_amp'][:500], label='gPAC', alpha=0.7)
    ax.plot(output_results['tensorpac_amp'][:500], label='TensorPAC', alpha=0.7)
    ax.set_title(f"Amplitude Filtered Output (r={output_results['amp_corr']:.3f})")
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('./scripts/vs_tensorpac/results/figures/filter_comparison.png', dpi=150)
    plt.close()
    
    print("\n✅ Comparison plot saved to: ./scripts/vs_tensorpac/results/figures/filter_comparison.png")


def analyze_conv1d_vs_filtfilt():
    """Analyze the difference between conv1d and filtfilt implementations."""
    
    print("\n" + "="*60)
    print("CONV1D vs FILTFILT ANALYSIS")
    print("="*60)
    
    # Generate test signal
    fs = 1000.0
    n_samples = 2000
    t = np.arange(n_samples) / fs
    test_signal = np.sin(2 * np.pi * 6 * t)
    
    # Get filter
    from gpac._tensorpac_fir1 import design_filter_tensorpac, fir1, fir_order as gpac_fir_order
    
    filter_order = gpac_fir_order(fs, n_samples, 4.0, cycle=3)
    wn = np.array([4.0, 8.0]) / (fs / 2.0)
    b_coeff, a_coeff = fir1(filter_order, wn)
    
    print(f"\nFilter order: {filter_order}")
    print(f"Filter length: {len(b_coeff)}")
    
    # Method 1: scipy filtfilt
    filtered_filtfilt = filtfilt(b_coeff, a_coeff, test_signal, padlen=filter_order)
    
    # Method 2: PyTorch conv1d with filtfilt mode
    signal_tensor = torch.from_numpy(test_signal.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(b_coeff.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    filtered_conv1d = torch.nn.functional.conv1d(signal_tensor, kernel_tensor, padding='same')
    # Backward pass (time reversal)
    filtered_conv1d = torch.nn.functional.conv1d(
        filtered_conv1d.flip(-1), kernel_tensor, padding='same'
    ).flip(-1)
    
    filtered_conv1d_np = filtered_conv1d.squeeze().numpy()
    
    # Compare
    correlation = np.corrcoef(filtered_filtfilt, filtered_conv1d_np)[0, 1]
    rms_diff = np.sqrt(np.mean((filtered_filtfilt - filtered_conv1d_np)**2))
    
    print(f"\nCorrelation: {correlation:.6f}")
    print(f"RMS difference: {rms_diff:.6f}")
    
    if correlation > 0.999:
        print("✅ conv1d implementation matches filtfilt well")
    else:
        print("❌ conv1d implementation differs from filtfilt")
    
    return correlation


def main():
    """Run comprehensive filter analysis."""
    
    print("="*70)
    print("gPAC vs TensorPAC FILTER IMPLEMENTATION ANALYSIS")
    print("="*70)
    
    # Create output directory
    os.makedirs('./scripts/vs_tensorpac/results/figures', exist_ok=True)
    
    # 1. Compare filter coefficients
    filter_results = analyze_filter_coefficients()
    
    # 2. Compare filtering outputs
    output_results = analyze_filtering_output()
    
    # 3. Analyze conv1d vs filtfilt
    conv1d_correlation = analyze_conv1d_vs_filtfilt()
    
    # 4. Create comparison plots
    plot_comparison(filter_results, output_results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nFilter Coefficient Agreement:")
    print(f"  - Phase filters: r={filter_results['pha_corr']:.3f}")
    print(f"  - Amplitude filters: r={filter_results['amp_corr']:.3f}")
    
    print("\nFiltered Output Agreement:")
    print(f"  - Phase output: r={output_results['pha_corr']:.3f}")
    print(f"  - Amplitude output: r={output_results['amp_corr']:.3f}")
    
    print("\nImplementation Analysis:")
    print(f"  - conv1d vs filtfilt: r={conv1d_correlation:.3f}")
    
    if filter_results['pha_corr'] > 0.99 and filter_results['amp_corr'] > 0.99:
        print("\n✅ Filter coefficients match perfectly - using same algorithm")
    else:
        print("\n❌ Filter coefficients differ - different implementations")
    
    if output_results['pha_corr'] < 0.9 or output_results['amp_corr'] < 0.9:
        print("❌ Filtered outputs show poor agreement - likely an implementation issue")
        print("   Possible causes:")
        print("   - Different padding strategies")
        print("   - Different convolution methods")
        print("   - Edge handling differences")
    else:
        print("✅ Filtered outputs show good agreement")


if __name__ == "__main__":
    main()