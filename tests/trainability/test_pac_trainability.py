#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 23:30:00 (ywatanabe)"
# File: ./tests/test_pac_trainability.py

"""
Test trainability of PAC with SincNet-style learnable frequency bands.

This test verifies that:
1. Frequency bands can be optimized through backpropagation
2. Loss decreases with training
3. Learned bands improve PAC detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from gpac import PAC, generate_pac_signal


class TestPACTrainability:
    """Test PAC trainability with learnable frequency bands."""
    
    def test_band_optimization(self):
        """Test that frequency bands can be optimized to detect known PAC."""
        print("\n" + "="*60)
        print("PAC TRAINABILITY TEST")
        print("="*60)
        
        # Parameters
        fs = 256
        duration = 2.0
        n_samples = int(fs * duration)
        n_epochs = 100
        batch_size = 16
        
        # True PAC parameters (what we want to learn)
        true_phase_band = [8, 12]    # Alpha
        true_amp_band = [50, 100]     # Gamma
        
        print(f"\nTrue frequency bands:")
        print(f"  Phase: {true_phase_band} Hz")
        print(f"  Amplitude: {true_amp_band} Hz")
        
        # Initialize PAC with wrong bands (to be learned)
        pac_model = PAC(
            seq_len=n_samples,
            fs=fs,
            pha_start_hz=4,
            pha_end_hz=16,
            pha_n_bands=4,
            amp_start_hz=30,
            amp_end_hz=110,
            amp_n_bands=4,
            trainable=True,  # Use trainable filters
            n_perm=0
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pac_model = pac_model.to(device)
        
        # Print initial bands
        if hasattr(pac_model.bandpass, 'get_filter_banks'):
            initial_bands = pac_model.bandpass.get_filter_banks()
            print("\nInitial frequency bands:")
            print(f"  Phase bands: {initial_bands['pha_bands'].cpu().numpy()}")
            print(f"  Amplitude bands: {initial_bands['amp_bands'].cpu().numpy()}")
        
        # Optimizer
        optimizer = optim.Adam(pac_model.parameters(), lr=0.01)
        
        # Training loop
        losses = []
        print("\nTraining to find optimal frequency bands...")
        
        for epoch in tqdm(range(n_epochs)):
            # Generate batch of signals
            batch_signals = []
            batch_labels = []
            
            for _ in range(batch_size):
                # Half with strong PAC, half with no PAC
                if np.random.rand() > 0.5:
                    # Generate PAC signal with true frequencies
                    signal = generate_pac_signal(
                        duration=duration,
                        fs=fs,
                        phase_freq=np.mean(true_phase_band),
                        amp_freq=np.mean(true_amp_band),
                        coupling_strength=0.8,
                        noise_level=0.1
                    )
                    label = 1.0  # Strong PAC
                else:
                    # Generate random noise
                    signal = np.random.randn(n_samples) * 0.1
                    label = 0.0  # No PAC
                
                batch_signals.append(signal)
                batch_labels.append(label)
            
            # Convert to tensors - PAC expects (batch, n_segments, seq_len)
            x = torch.tensor(np.array(batch_signals), dtype=torch.float32).unsqueeze(1).to(device)
            y = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            # Forward pass
            pac_output = pac_model(x)  # Returns dictionary
            pac_values = pac_output['pac']  # Shape: (batch, 1, n_pha, n_amp)
            
            # Compute loss: maximize PAC in the correct bands
            # We'll use the maximum PAC value across all band combinations
            # Squeeze channel dimension and convert to float32
            pac_values = pac_values.squeeze(1).float()  # Shape: (batch, n_pha, n_amp)
            max_pac = pac_values.view(batch_size, -1).max(dim=1)[0]
            
            # Loss: MSE between max PAC and label
            loss = nn.MSELoss()(max_pac, y)
            
            # Add regularization from filter
            if hasattr(pac_model.bandpass, 'get_regularization_loss'):
                reg_losses = pac_model.bandpass.get_regularization_loss(
                    lambda_overlap=0.001,
                    lambda_bandwidth=0.001
                )
                loss = loss + reg_losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply constraints to ensure valid parameters
            if hasattr(pac_model.bandpass, 'constrain_parameters'):
                pac_model.bandpass.constrain_parameters()
            
            losses.append(loss.item())
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"\nEpoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
                if hasattr(pac_model.bandpass, 'get_filter_banks'):
                    current_bands = pac_model.bandpass.get_filter_banks()
                    print(f"Current phase bands: {current_bands['pha_bands'][:2].cpu().numpy()}")
                    print(f"Current amp bands: {current_bands['amp_bands'][:2].cpu().numpy()}")
        
        # Final evaluation
        print("\n" + "-"*60)
        print("TRAINING COMPLETE")
        print("-"*60)
        
        # Get final bands
        if hasattr(pac_model.bandpass, 'get_filter_banks'):
            final_bands = pac_model.bandpass.get_filter_banks()
            print("\nFinal learned frequency bands:")
            print(f"  Phase bands: {final_bands['pha_bands'].cpu().numpy()}")
            print(f"  Amplitude bands: {final_bands['amp_bands'].cpu().numpy()}")
            
            # Find the band closest to true band
            pha_centers = final_bands['pha_bands'].mean(dim=1)
            amp_centers = final_bands['amp_bands'].mean(dim=1)
            
            true_pha_center = np.mean(true_phase_band)
            true_amp_center = np.mean(true_amp_band)
            
            best_pha_idx = torch.argmin(torch.abs(pha_centers - true_pha_center))
            best_amp_idx = torch.argmin(torch.abs(amp_centers - true_amp_center))
            
            print(f"\nBest learned bands (closest to true):")
            print(f"  Phase: {final_bands['pha_bands'][best_pha_idx].cpu().numpy()} Hz (true: {true_phase_band})")
            print(f"  Amplitude: {final_bands['amp_bands'][best_amp_idx].cpu().numpy()} Hz (true: {true_amp_band})")
        
        # Test on new data
        print("\nTesting on new signals...")
        test_pac_signal = generate_pac_signal(
            duration=duration,
            fs=fs,
            phase_freq=np.mean(true_phase_band),
            amp_freq=np.mean(true_amp_band),
            coupling_strength=0.8,
            noise_level=0.1
        )
        test_noise = np.random.randn(n_samples) * 0.1
        
        with torch.no_grad():
            pac_signal_result = pac_model(torch.tensor(test_pac_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            noise_result = pac_model(torch.tensor(test_noise, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            
            max_pac_signal = pac_signal_result['pac'].max().item()
            max_pac_noise = noise_result['pac'].max().item()
            
            print(f"\nMax PAC for signal with coupling: {max_pac_signal:.4f}")
            print(f"Max PAC for noise: {max_pac_noise:.4f}")
            print(f"Ratio: {max_pac_signal / (max_pac_noise + 1e-10):.2f}")
        
        # Verify training worked
        assert losses[-1] < losses[0] * 0.5, "Loss did not decrease sufficiently"
        assert max_pac_signal > max_pac_noise * 2, "Trained model doesn't distinguish PAC from noise"
        
        print("\n✓ PAC trainability test passed!")
        
        return losses


def run_trainability_test():
    """Run PAC trainability test."""
    test = TestPACTrainability()
    losses = test.test_band_optimization()
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PAC Training Loss')
    plt.grid(True)
    plt.savefig('pac_training_loss.png')
    print("\nTraining curve saved to pac_training_loss.png")


if __name__ == "__main__":
    run_trainability_test()