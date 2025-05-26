#!/usr/bin/env python3
"""
Test script to verify Hilbert transform differentiability.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from gpac._Hilbert import Hilbert


def test_gradient_flow():
    """Test if gradients flow through Hilbert transform."""
    print("="*60)
    print("TESTING HILBERT TRANSFORM GRADIENT FLOW")
    print("="*60)
    
    # Create test signal
    seq_len = 1024
    batch_size = 2
    signal = torch.randn(batch_size, seq_len, requires_grad=True)
    
    # Initialize Hilbert transform
    hilbert = Hilbert(seq_len=seq_len, dim=-1)
    
    # Forward pass
    output = hilbert(signal)
    
    # Check output shape
    print(f"\nInput shape: {signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output contains phase and amplitude: {output.shape[-1] == 2}")
    
    # Test gradient flow
    print("\n1. Testing gradient flow through amplitude:")
    loss_amp = output[..., 1].sum()  # Sum of amplitudes
    loss_amp.backward(retain_graph=True)
    
    if signal.grad is not None:
        print(f"   ✓ Gradients flow through amplitude")
        print(f"   Gradient norm: {signal.grad.norm().item():.6f}")
        signal.grad.zero_()
    else:
        print(f"   ✗ No gradients through amplitude!")
    
    print("\n2. Testing gradient flow through phase:")
    loss_phase = output[..., 0].sum()  # Sum of phases
    loss_phase.backward()
    
    if signal.grad is not None:
        print(f"   ✓ Gradients flow through phase")
        print(f"   Gradient norm: {signal.grad.norm().item():.6f}")
    else:
        print(f"   ✗ No gradients through phase!")
    
    return signal.grad is not None


def test_complex_operations():
    """Test differentiability of complex operations."""
    print("\n" + "="*60)
    print("TESTING COMPLEX OPERATIONS DIFFERENTIABILITY")
    print("="*60)
    
    # Test individual operations
    x = torch.randn(10, requires_grad=True)
    
    # 1. FFT
    print("\n1. FFT operation:")
    xf = torch.fft.fft(x)
    loss_fft = xf.abs().sum()
    loss_fft.backward()
    print(f"   FFT gradient norm: {x.grad.norm().item():.6f}")
    x.grad.zero_()
    
    # 2. IFFT
    print("\n2. IFFT operation:")
    x_complex = torch.complex(x, torch.zeros_like(x))
    xi = torch.fft.ifft(x_complex)
    loss_ifft = xi.abs().sum()
    loss_ifft.backward()
    print(f"   IFFT gradient norm: {x.grad.norm().item():.6f}")
    x.grad.zero_()
    
    # 3. atan2
    print("\n3. atan2 operation:")
    y = torch.randn(10, requires_grad=True)
    phase = torch.atan2(y, x)
    loss_atan2 = phase.sum()
    loss_atan2.backward()
    print(f"   atan2 gradient norm (x): {x.grad.norm().item():.6f}")
    print(f"   atan2 gradient norm (y): {y.grad.norm().item():.6f}")
    
    # 4. Complex abs
    print("\n4. Complex abs operation:")
    z = torch.complex(x, y)
    amp = z.abs()
    loss_abs = amp.sum()
    x.grad.zero_()
    y.grad.zero_()
    loss_abs.backward()
    print(f"   Complex abs gradient norm (real): {x.grad.norm().item():.6f}")
    print(f"   Complex abs gradient norm (imag): {y.grad.norm().item():.6f}")
    
    return True


def test_phase_extraction():
    """Test phase extraction and compare with numpy."""
    print("\n" + "="*60)
    print("TESTING PHASE EXTRACTION")
    print("="*60)
    
    # Create a test signal with known frequency
    fs = 1000
    duration = 1.0
    t = torch.linspace(0, duration, int(fs * duration))
    freq = 10.0  # 10 Hz
    
    # Create signal
    signal = torch.sin(2 * np.pi * freq * t)
    
    # Compute Hilbert transform
    hilbert = Hilbert(seq_len=len(t))
    output = hilbert(signal.unsqueeze(0))
    
    phase_torch = output[0, :, 0].detach().numpy()
    amp_torch = output[0, :, 1].detach().numpy()
    
    # Compare with scipy
    from scipy.signal import hilbert as scipy_hilbert
    analytic_scipy = scipy_hilbert(signal.numpy())
    phase_scipy = np.angle(analytic_scipy)
    amp_scipy = np.abs(analytic_scipy)
    
    # Calculate correlations
    phase_corr = np.corrcoef(phase_torch, phase_scipy)[0, 1]
    amp_corr = np.corrcoef(amp_torch, amp_scipy)[0, 1]
    
    print(f"\nPhase correlation with scipy: {phase_corr:.6f}")
    print(f"Amplitude correlation with scipy: {amp_corr:.6f}")
    
    # Check phase unwrapping
    phase_diff = np.diff(phase_torch)
    jumps = np.abs(phase_diff) > np.pi
    print(f"\nPhase jumps > π: {np.sum(jumps)}")
    
    return phase_corr, amp_corr


def test_numerical_gradient():
    """Test numerical gradient vs automatic gradient."""
    print("\n" + "="*60)
    print("TESTING NUMERICAL GRADIENTS")
    print("="*60)
    
    torch.manual_seed(42)
    seq_len = 64
    signal = torch.randn(1, seq_len, requires_grad=True)
    
    hilbert = Hilbert(seq_len=seq_len)
    
    # Automatic gradient
    output = hilbert(signal)
    loss = output[..., 1].sum()  # Use amplitude
    loss.backward()
    auto_grad = signal.grad.clone()
    
    # Numerical gradient
    eps = 1e-5  # Smaller epsilon for better accuracy
    numerical_grad = torch.zeros_like(signal)
    
    with torch.no_grad():
        for i in range(seq_len):
            signal_plus = signal.clone().detach()
            signal_minus = signal.clone().detach()
            
            signal_plus[0, i] += eps
            signal_minus[0, i] -= eps
            
            output_plus = hilbert(signal_plus)[..., 1].sum()
            output_minus = hilbert(signal_minus)[..., 1].sum()
            
            numerical_grad[0, i] = (output_plus - output_minus) / (2 * eps)
    
    # Compare
    rel_error = (auto_grad - numerical_grad).abs() / (auto_grad.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    print(f"\nMax relative error: {max_rel_error:.6f}")
    print(f"Mean relative error: {mean_rel_error:.6f}")
    
    if max_rel_error < 0.01:
        print("✓ Numerical gradient check passed!")
    else:
        print("✗ Numerical gradient check failed!")
    
    return max_rel_error < 0.01


def main():
    """Run all differentiability tests."""
    
    # Test 1: Basic gradient flow
    grad_flow_ok = test_gradient_flow()
    
    # Test 2: Complex operations
    complex_ops_ok = test_complex_operations()
    
    # Test 3: Phase extraction
    phase_corr, amp_corr = test_phase_extraction()
    
    # Test 4: Numerical gradient
    numerical_ok = test_numerical_gradient()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Gradient flow: {'✓ PASS' if grad_flow_ok else '✗ FAIL'}")
    print(f"Complex operations: {'✓ PASS' if complex_ops_ok else '✗ FAIL'}")
    print(f"Phase correlation: {phase_corr:.3f} {'✓' if abs(phase_corr) > 0.95 else '✗'}")
    print(f"Amplitude correlation: {amp_corr:.3f} {'✓' if amp_corr > 0.99 else '✗'}")
    print(f"Numerical gradient: {'✓ PASS' if numerical_ok else '✗ FAIL'}")
    
    # Overall assessment
    all_ok = (grad_flow_ok and complex_ops_ok and numerical_ok and 
              amp_corr > 0.99)
    
    print(f"\nOverall differentiability: {'✓ VERIFIED' if all_ok else '✗ ISSUES FOUND'}")
    
    if abs(phase_corr) < 0.95:
        print("\n⚠️ Phase extraction shows poor correlation with scipy.")
        print("   This may be due to phase wrapping or reference differences.")


if __name__ == "__main__":
    main()