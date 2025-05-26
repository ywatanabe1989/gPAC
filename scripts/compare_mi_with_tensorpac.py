#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Compare MI implementations with TensorPAC

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "tensorpac_source"))

from gpac._ModulationIndex import ModulationIndex
from gpac._DifferentiableModulationIndex import DifferentiableModulationIndex
from tensorpac.methods import compute_pac

def create_pac_signal(n_samples=1000, f_pha=10, f_amp=60, pac_strength=0.5):
    """Create synthetic PAC signal."""
    t = np.linspace(0, 1, n_samples)
    
    # Low frequency phase signal
    pha_signal = np.sin(2 * np.pi * f_pha * t)
    
    # High frequency amplitude signal modulated by phase
    amp_envelope = 1 + pac_strength * np.sin(2 * np.pi * f_pha * t)
    amp_signal = amp_envelope * np.sin(2 * np.pi * f_amp * t)
    
    return pha_signal, amp_signal, t

def test_mi_comparison():
    """Compare MI values between implementations."""
    print("Comparing Modulation Index implementations\n")
    
    # Create synthetic PAC data
    n_samples = 1000
    pha_signal, amp_signal, t = create_pac_signal(n_samples, pac_strength=0.8)
    
    # Extract phase and amplitude  
    from scipy.signal import hilbert
    pha = np.angle(hilbert(pha_signal))
    amp = np.abs(hilbert(amp_signal))
    
    # Reshape for our format: (B, C, F, Seg, Time)
    pha_tensor = torch.tensor(pha).reshape(1, 1, 1, 1, -1).float()
    amp_tensor = torch.tensor(amp).reshape(1, 1, 1, 1, -1).float()
    
    # 1. TensorPAC MI (Tort method)
    print("1. TensorPAC Modulation Index (Tort method):")
    print("-" * 50)
    tensorpac_mi = compute_pac(
        pha.reshape(1, 1, -1), 
        amp.reshape(1, 1, -1),
        method='tort',
        n_bins=18
    )
    print(f"   MI value: {tensorpac_mi[0, 0]:.6f}")
    
    # 2. Standard gPAC MI
    print("\n2. gPAC Standard ModulationIndex:")
    print("-" * 50)
    standard_mi = ModulationIndex(n_bins=18)
    with torch.no_grad():
        output = standard_mi(pha_tensor, amp_tensor)
        gpac_mi = output['mi']
        print(f"   MI value: {gpac_mi.item():.6f}")
        print(f"   Difference from TensorPAC: {abs(gpac_mi.item() - tensorpac_mi[0, 0]):.6f}")
    
    # 3. Differentiable gPAC MI
    print("\n3. gPAC Differentiable ModulationIndex:")
    print("-" * 50)
    
    temperatures = [0.1, 0.5, 1.0, 2.0]
    for temp in temperatures:
        diff_mi = DifferentiableModulationIndex(n_bins=18, temperature=temp)
        with torch.no_grad():
            output = diff_mi(pha_tensor, amp_tensor)
            diff_mi_val = output['mi']
            print(f"   Temperature {temp}: MI = {diff_mi_val.item():.6f}, "
                  f"Diff = {abs(diff_mi_val.item() - tensorpac_mi[0, 0]):.6f}")
    
    # Test gradient flow
    print("\n4. Gradient Flow Test:")
    print("-" * 50)
    
    # Standard MI
    pha_grad = pha_tensor.clone().requires_grad_(True)
    amp_grad = amp_tensor.clone().requires_grad_(True)
    
    output = standard_mi(pha_grad, amp_grad)
    loss = output['mi'].mean()
    
    try:
        loss.backward()
        print("   Standard MI: Backward succeeded")
        if pha_grad.grad is not None:
            print(f"     Phase grad exists: {pha_grad.grad.abs().max().item() > 0}")
        if amp_grad.grad is not None:
            print(f"     Amp grad exists: {amp_grad.grad.abs().max().item() > 0}")
    except Exception as e:
        print(f"   Standard MI: Backward failed - {str(e)[:50]}...")
    
    # Differentiable MI  
    pha_grad2 = pha_tensor.clone().requires_grad_(True)
    amp_grad2 = amp_tensor.clone().requires_grad_(True)
    
    diff_mi = DifferentiableModulationIndex(n_bins=18, temperature=0.5)
    output2 = diff_mi(pha_grad2, amp_grad2)
    loss2 = output2['mi'].mean()
    
    try:
        loss2.backward()
        print("\n   Differentiable MI: Backward succeeded")
        if pha_grad2.grad is not None:
            print(f"     Phase grad norm: {pha_grad2.grad.norm().item():.6f}")
            print(f"     Phase grad max: {pha_grad2.grad.abs().max().item():.6f}")
        if amp_grad2.grad is not None:
            print(f"     Amp grad norm: {amp_grad2.grad.norm().item():.6f}")
            print(f"     Amp grad max: {amp_grad2.grad.abs().max().item():.6f}")
    except Exception as e:
        print(f"   Differentiable MI: Backward failed - {str(e)[:50]}...")

if __name__ == "__main__":
    print("="*60)
    print("ModulationIndex TensorPAC Comparison")
    print("="*60 + "\n")
    
    test_mi_comparison()
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    print("\n1. Standard gPAC MI should match TensorPAC Tort method closely")
    print("2. Differentiable MI trades accuracy for gradient flow")
    print("3. Lower temperature = closer to discrete binning")
    print("4. Only differentiable MI allows gradient-based optimization")
    print("="*60)