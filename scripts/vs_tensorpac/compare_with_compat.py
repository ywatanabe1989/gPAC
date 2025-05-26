#!/usr/bin/env python3
"""Compare gPAC with TensorPAC using the compatibility layer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch

# Import compatibility function
from gpac._calculate_gpac_tensorpac_compat import calculate_pac_tensorpac_compat

# Try to import TensorPAC
try:
    tensorpac_path = os.path.join(os.path.dirname(__file__), '../../tensorpac_source')
    sys.path.insert(0, tensorpac_path)
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False

def compare_with_compatibility():
    """Compare using compatibility layer."""
    
    if not TENSORPAC_AVAILABLE:
        print("TensorPAC not available")
        return
    
    # Generate test signals
    np.random.seed(42)
    torch.manual_seed(42)
    
    fs = 1000.0
    duration = 2.0
    n_trials = 5
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    signals = []
    for trial in range(n_trials):
        # Create PAC signal
        pha_freq = 5.0 + np.random.normal(0, 0.5)
        amp_freq = 80.0 + np.random.normal(0, 5)
        coupling_strength = 0.3 + np.random.uniform(0, 0.4)
        
        phase = np.sin(2 * np.pi * pha_freq * t + np.random.uniform(0, 2*np.pi))
        amplitude_envelope = 1 + coupling_strength * np.sin(2 * np.pi * pha_freq * t)
        amplitude = amplitude_envelope * np.sin(2 * np.pi * amp_freq * t)
        
        signal = 0.5 * phase + 0.3 * amplitude + 0.2 * np.random.randn(n_samples)
        signals.append(signal)
    
    signals = np.array(signals)
    
    print("="*60)
    print("gPAC vs TensorPAC with Compatibility Layer")
    print("="*60)
    
    # Configuration
    pha_freqs = (4, 16)
    amp_freqs = (60, 120)
    pha_n_bands = 8
    amp_n_bands = 8
    
    # Run gPAC with compatibility
    print("\n1. Running gPAC (compatibility mode)...")
    signal_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
    
    pac_gpac, pha_mids_gpac, amp_mids_gpac = calculate_pac_tensorpac_compat(
        signal=signal_tensor,
        fs=fs,
        pha_start_hz=pha_freqs[0],
        pha_end_hz=pha_freqs[1],
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_freqs[0],
        amp_end_hz=amp_freqs[1],
        amp_n_bands=amp_n_bands,
        n_perm=None
    )
    
    # Average over trials
    pac_gpac_mean = pac_gpac.mean(dim=0).squeeze().cpu().numpy()
    
    print(f"  Shape: {pac_gpac_mean.shape}")
    print(f"  Range: [{pac_gpac_mean.min():.6f}, {pac_gpac_mean.max():.6f}]")
    print(f"  Mean: {pac_gpac_mean.mean():.6f}")
    
    # Run TensorPAC
    print("\n2. Running TensorPAC...")
    
    # Create frequency bands
    pha_vec = np.linspace(pha_freqs[0], pha_freqs[1], pha_n_bands + 1)
    amp_vec = np.linspace(amp_freqs[0], amp_freqs[1], amp_n_bands + 1)
    pha_bands = [[pha_vec[i], pha_vec[i+1]] for i in range(pha_n_bands)]
    amp_bands = [[amp_vec[i], amp_vec[i+1]] for i in range(amp_n_bands)]
    
    pac_obj = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands)
    pac_tp = pac_obj.filterfit(fs, signals)
    
    # Average over trials
    pac_tp_mean = pac_tp.mean(axis=2)
    
    print(f"  Shape: {pac_tp_mean.shape}")
    print(f"  Range: [{pac_tp_mean.min():.6f}, {pac_tp_mean.max():.6f}]")
    print(f"  Mean: {pac_tp_mean.mean():.6f}")
    
    # Compare
    print("\n3. Comparison:")
    
    # Reshape TensorPAC output to match gPAC
    pac_tp_reshaped = pac_tp_mean.T  # (amp, pha) -> (pha, amp)
    
    # Calculate correlation
    correlation = np.corrcoef(pac_gpac_mean.flatten(), pac_tp_reshaped.flatten())[0, 1]
    
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Max absolute error: {np.abs(pac_gpac_mean - pac_tp_reshaped).max():.6f}")
    print(f"  Mean absolute error: {np.abs(pac_gpac_mean - pac_tp_reshaped).mean():.6f}")
    print(f"  Relative error: {np.abs(pac_gpac_mean - pac_tp_reshaped).mean() / pac_tp_reshaped.mean() * 100:.1f}%")
    
    # Success criteria
    print("\n4. Success Criteria:")
    print(f"  ✅ Correlation > 0.9: {'YES' if correlation > 0.9 else 'NO'}")
    print(f"  ✅ Mean values similar: {'YES' if abs(pac_gpac_mean.mean() - pac_tp_mean.mean()) < 0.01 else 'NO'}")
    print(f"  ✅ Max values similar: {'YES' if abs(pac_gpac_mean.max() - pac_tp_mean.max()) < 0.05 else 'NO'}")
    
    print("\n" + "="*60)
    if correlation > 0.9:
        print("🎉 SUCCESS! High correlation achieved with compatibility layer!")
    else:
        print(f"⚠️  Correlation {correlation:.3f} - needs further tuning")

if __name__ == "__main__":
    compare_with_compatibility()