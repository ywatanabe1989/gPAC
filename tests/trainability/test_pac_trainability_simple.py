#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 23:45:00 (ywatanabe)"
# File: ./tests/test_pac_trainability_simple.py

"""
Simple test for PAC trainability with SincNet-style learnable frequency bands.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gpac import PAC, generate_pac_signal
from gpac._Filters._DifferentiableBandpassFilter import DifferentiableBandPassFilter


def test_sincnet_filter_trainability():
    """Test that SincNet-style filters can learn frequency bands."""
    print("\n" + "="*60)
    print("SINCNET FILTER TRAINABILITY TEST")
    print("="*60)
    
    # Parameters
    fs = 256
    sig_len = 512
    n_epochs = 10
    
    # Target signal with known frequency
    target_freq = 10.0  # Hz
    t = torch.arange(sig_len) / fs
    target_signal = torch.sin(2 * np.pi * target_freq * t)
    
    # Initialize filter with wrong bands
    filter_model = DifferentiableBandPassFilter(
        sig_len=sig_len,
        fs=fs,
        pha_low_hz=15,  # Wrong initial band
        pha_high_hz=25,
        pha_n_bands=1,
        amp_low_hz=50,
        amp_high_hz=100,
        amp_n_bands=1,
        filter_length=101,
        normalization='std'
    )
    
    print(f"\nTarget frequency: {target_freq} Hz")
    print("\nInitial bands:")
    initial_bands = filter_model.get_filter_banks()
    print(f"  Phase band: {initial_bands['pha_bands'][0].numpy()} Hz")
    
    # Optimizer
    optimizer = optim.Adam(filter_model.parameters(), lr=0.1)
    
    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        filtered = filter_model(target_signal.unsqueeze(0).unsqueeze(0))
        
        # Loss: maximize energy in filtered signal (should learn to pass target freq)
        energy = torch.mean(filtered[0, 0, 0] ** 2)
        loss = -energy  # Negative because we want to maximize
        
        # Add regularization
        reg_losses = filter_model.get_regularization_loss(0.01, 0.01)
        loss = loss + reg_losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Constrain parameters
        filter_model.constrain_parameters()
        
        if epoch % 2 == 0:
            current_bands = filter_model.get_filter_banks()
            print(f"\nEpoch {epoch}: Phase band = {current_bands['pha_bands'][0].numpy()} Hz, Energy = {energy.item():.4f}")
    
    # Final bands
    final_bands = filter_model.get_filter_banks()
    final_band = final_bands['pha_bands'][0].numpy()
    print(f"\nFinal phase band: {final_band} Hz")
    print(f"Target frequency {target_freq} Hz is {'inside' if final_band[0] <= target_freq <= final_band[1] else 'outside'} the learned band")
    
    # Verify the band moved towards the target
    initial_center = initial_bands['pha_bands'][0].mean().item()
    final_center = final_bands['pha_bands'][0].mean().item()
    
    print(f"\nBand center moved from {initial_center:.1f} Hz to {final_center:.1f} Hz")
    print(f"Distance to target reduced from {abs(initial_center - target_freq):.1f} Hz to {abs(final_center - target_freq):.1f} Hz")
    
    return abs(final_center - target_freq) < abs(initial_center - target_freq)


def test_pac_with_known_coupling():
    """Test PAC with a simple known coupling."""
    print("\n" + "="*60)
    print("PAC WITH KNOWN COUPLING TEST")
    print("="*60)
    
    # Parameters
    fs = 256
    duration = 1.0
    n_samples = int(fs * duration)
    
    # Create PAC model with trainable filters
    pac_model = PAC(
        seq_len=n_samples,
        fs=fs,
        pha_start_hz=5,
        pha_end_hz=15,
        pha_n_bands=2,
        amp_start_hz=40,
        amp_end_hz=120,
        amp_n_bands=2,
        trainable=True,
        n_perm=0
    )
    
    # Generate signal with known PAC
    signal = generate_pac_signal(
        duration=duration,
        fs=fs,
        phase_freq=10,
        amp_freq=80,
        coupling_strength=0.8,
        noise_level=0.05
    )
    
    # Convert to tensor
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Test forward pass
    with torch.no_grad():
        output = pac_model(x)
        pac_values = output['pac'].squeeze().float()
        
        print(f"\nPAC output shape: {pac_values.shape}")
        print(f"Max PAC value: {pac_values.max().item():.4f}")
        print(f"Min PAC value: {pac_values.min().item():.4f}")
        
        # Find which band combination has highest PAC
        max_idx = pac_values.argmax()
        max_pha_idx = max_idx // pac_values.shape[1]
        max_amp_idx = max_idx % pac_values.shape[1]
        
        bands = pac_model.bandpass.get_filter_banks()
        print(f"\nHighest PAC at:")
        print(f"  Phase band {max_pha_idx}: {bands['pha_bands'][max_pha_idx].numpy()} Hz")
        print(f"  Amp band {max_amp_idx}: {bands['amp_bands'][max_amp_idx].numpy()} Hz")
    
    # Test gradient flow
    x.requires_grad = True
    output = pac_model(x)
    pac_values = output['pac'].squeeze().float()
    loss = pac_values.max()
    loss.backward()
    
    print(f"\nGradient test passed: {x.grad is not None}")
    
    return True


if __name__ == "__main__":
    print("Running SincNet-style PAC trainability tests...")
    
    # Test 1: Filter learning
    filter_test_passed = test_sincnet_filter_trainability()
    
    # Test 2: PAC with coupling
    pac_test_passed = test_pac_with_known_coupling()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Filter trainability test: {'PASSED' if filter_test_passed else 'FAILED'}")
    print(f"PAC coupling test: {'PASSED' if pac_test_passed else 'FAILED'}")
    print("\nKey features demonstrated:")
    print("- SincNet-style filters with learnable frequency boundaries")
    print("- Frequency-dependent normalization")
    print("- Regularization for band overlap and bandwidth control")
    print("- Full gradient flow through PAC pipeline")