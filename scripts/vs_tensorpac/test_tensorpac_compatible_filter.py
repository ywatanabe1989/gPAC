#!/usr/bin/env python3
"""
Test the new TensorPAC-compatible filter implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import time
from scipy.signal import filtfilt

# Import both implementations
from gpac._BandPassFilter_TensorPACCompatible import BandPassFilterTensorPACCompatible
from gpac._tensorpac_fir1 import fir1, fir_order

# Import TensorPAC
tensorpac_path = os.path.join(os.path.dirname(__file__), '../../tensorpac_source')
if tensorpac_path not in sys.path:
    sys.path.insert(0, tensorpac_path)
from tensorpac import Pac


def test_filter_compatibility():
    """Test if the new filter achieves >95% correlation with TensorPAC."""
    
    print("="*70)
    print("TESTING TENSORPAC-COMPATIBLE FILTER")
    print("="*70)
    
    # Parameters
    fs = 1000.0
    duration = 2.0
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Generate test signal
    signal = (np.sin(2 * np.pi * 6 * t) + 
              0.5 * np.sin(2 * np.pi * 80 * t) + 
              0.1 * np.random.randn(n_samples))
    
    # Frequency bands to test
    pha_bands = torch.tensor([[4.0, 8.0], [8.0, 13.0]])
    amp_bands = torch.tensor([[60.0, 80.0], [80.0, 100.0]])
    
    print(f"\nTest Configuration:")
    print(f"  Signal: {duration}s @ {fs}Hz")
    print(f"  Phase bands: {pha_bands.tolist()}")
    print(f"  Amplitude bands: {amp_bands.tolist()}")
    
    # 1. Apply TensorPAC filtering
    print("\n1. TENSORPAC FILTERING")
    tp_results = []
    
    for band in pha_bands:
        order = fir_order(fs, n_samples, band[0].item(), cycle=3)
        wn = band.numpy() / (fs / 2.0)
        b, _ = fir1(order, wn)
        filtered = filtfilt(b, 1.0, signal, padlen=order)
        tp_results.append(filtered)
    
    for band in amp_bands:
        order = fir_order(fs, n_samples, band[0].item(), cycle=6)
        wn = band.numpy() / (fs / 2.0)
        b, _ = fir1(order, wn)
        filtered = filtfilt(b, 1.0, signal, padlen=order)
        tp_results.append(filtered)
    
    tp_results = np.array(tp_results)
    print(f"   Output shape: {tp_results.shape}")
    
    # 2. Apply new gPAC compatible filter
    print("\n2. GPAC COMPATIBLE FILTERING")
    filter_module = BandPassFilterTensorPACCompatible(
        pha_bands=pha_bands,
        amp_bands=amp_bands,
        fs=fs,
        seq_len=n_samples,
        cycle_pha=3,
        cycle_amp=6
    )
    
    # Prepare input
    signal_tensor = torch.from_numpy(signal.astype(np.float32))
    signal_input = signal_tensor.reshape(1, 1, -1)  # (batch, 1, time)
    
    # Apply filter
    with torch.no_grad():
        gpac_output = filter_module(signal_input)
    
    gpac_results = gpac_output.squeeze().numpy()  # (n_bands, time)
    print(f"   Output shape: {gpac_results.shape}")
    
    # 3. Compare results
    print("\n3. CORRELATION ANALYSIS")
    correlations = []
    for i in range(len(tp_results)):
        corr = np.corrcoef(tp_results[i], gpac_results[i])[0, 1]
        correlations.append(corr)
        band_type = "Phase" if i < len(pha_bands) else "Amplitude"
        band_idx = i if i < len(pha_bands) else i - len(pha_bands)
        print(f"   {band_type} band {band_idx}: r = {corr:.6f}")
    
    avg_corr = np.mean(correlations)
    min_corr = np.min(correlations)
    
    print(f"\n   Average correlation: {avg_corr:.6f}")
    print(f"   Minimum correlation: {min_corr:.6f}")
    
    # 4. Performance comparison
    print("\n4. PERFORMANCE COMPARISON")
    
    # Time TensorPAC approach
    start = time.time()
    for _ in range(10):
        for band in pha_bands:
            order = fir_order(fs, n_samples, band[0].item(), cycle=3)
            wn = band.numpy() / (fs / 2.0)
            b, _ = fir1(order, wn)
            _ = filtfilt(b, 1.0, signal, padlen=order)
        for band in amp_bands:
            order = fir_order(fs, n_samples, band[0].item(), cycle=6)
            wn = band.numpy() / (fs / 2.0)
            b, _ = fir1(order, wn)
            _ = filtfilt(b, 1.0, signal, padlen=order)
    tp_time = (time.time() - start) / 10
    
    # Time gPAC approach
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = filter_module(signal_input)
    gpac_time = (time.time() - start) / 10
    
    print(f"   TensorPAC time: {tp_time*1000:.2f}ms")
    print(f"   gPAC time: {gpac_time*1000:.2f}ms")
    print(f"   Speedup: {tp_time/gpac_time:.2f}x")
    
    # 5. Test with full PAC pipeline
    print("\n5. FULL PAC PIPELINE TEST")
    
    # Create 5 test signals
    test_signals = []
    for i in range(5):
        sig = (np.sin(2 * np.pi * (5 + i) * t) + 
               0.5 * np.sin(2 * np.pi * (70 + i*10) * t) + 
               0.1 * np.random.randn(n_samples))
        test_signals.append(sig)
    test_signals = np.array(test_signals)
    
    # TensorPAC
    pac_tp = Pac(idpac=(2, 0, 0), f_pha=[4, 16], f_amp=[60, 120], dcomplex='hilbert')
    pac_tp_values = pac_tp.filterfit(fs, test_signals, n_jobs=1)
    
    print(f"   TensorPAC PAC shape: {pac_tp_values.shape}")
    print(f"   TensorPAC PAC values: {pac_tp_values.flatten()[:5]}")
    
    # Final verdict
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if min_corr >= 0.95:
        print("✅ SUCCESS: Achieved >95% correlation with TensorPAC!")
        print(f"   Minimum correlation: {min_corr:.4f}")
        print(f"   Average correlation: {avg_corr:.4f}")
    else:
        print("❌ FAILED: Did not achieve 95% correlation")
        print(f"   Minimum correlation: {min_corr:.4f}")
        print(f"   Need to investigate further")
    
    return min_corr >= 0.95


if __name__ == "__main__":
    success = test_filter_compatibility()
    sys.exit(0 if success else 1)