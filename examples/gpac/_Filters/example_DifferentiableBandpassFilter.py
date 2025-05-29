#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 22:58:46 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/gpac/_Filters/example_DifferentiableBandpassFilter.py
# ----------------------------------------
import os

__FILE__ = "./examples/gpac/_Filters/example_DifferentiableBandpassFilter.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Example demonstrating DifferentiableBandpassFilter with learnable bands."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from gpac._Filters._DifferentiableBandpassFilter import DifferentiableBandPassFilter

# Create output directory
os.makedirs("example_outputs", exist_ok=True)

# Parameters
fs = 256
sig_len = 512
n_epochs = 50

# Create a target signal with specific frequency
target_freq = 15.0  # Hz
t = torch.arange(sig_len) / fs
target_signal = torch.sin(2 * np.pi * target_freq * t) + 0.5 * torch.sin(
    2 * np.pi * target_freq * 2 * t
)

# Initialize filter with initial bands away from target
filter_model = DifferentiableBandPassFilter(
    sig_len=sig_len,
    fs=fs,
    pha_low_hz=25,  # Initial band: 25-35 Hz (away from 15 Hz target)
    pha_high_hz=35,
    pha_n_bands=1,
    amp_low_hz=50,
    amp_high_hz=100,
    amp_n_bands=1,
    filter_length=101,
    normalization="std",
)

# Optimizer
optimizer = optim.Adam(filter_model.parameters(), lr=0.05)

# Track band evolution
band_history = []

# Training loop
losses = []
print("Training filter to find optimal frequency band...")
for epoch in range(n_epochs):
    # Get current bands
    bands = filter_model.get_filter_banks()
    current_band = bands["pha_bands"][0].detach().numpy()
    band_history.append(current_band.copy())

    # Forward pass
    filtered = filter_model(target_signal.unsqueeze(0).unsqueeze(0))

    # Loss: maximize energy in filtered signal
    energy = torch.mean(filtered[0, 0, 0] ** 2)
    loss = -energy  # Negative because we want to maximize

    # Add regularization
    reg_losses = filter_model.get_regularization_loss(0.001, 0.001)
    loss = loss + reg_losses["total"]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Constrain parameters
    filter_model.constrain_parameters()

    losses.append(-loss.item())  # Store positive energy

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: Band = [{current_band[0]:.1f}, {current_band[1]:.1f}] Hz, Energy = {energy.item():.4f}"
        )

# Final band
final_bands = filter_model.get_filter_banks()
final_band = final_bands["pha_bands"][0].numpy()

# Plotting
fig = plt.figure(figsize=(15, 10))

# 1. Band evolution
ax1 = plt.subplot(2, 2, 1)
band_history = np.array(band_history)
ax1.fill_between(
    range(n_epochs),
    band_history[:, 0],
    band_history[:, 1],
    alpha=0.3,
    label="Learned band",
)
ax1.axhline(
    y=target_freq,
    color="r",
    linestyle="--",
    label=f"Target freq: {target_freq} Hz",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Band Evolution During Training")
ax1.legend()
ax1.grid(True)

# 2. Loss curve
ax2 = plt.subplot(2, 2, 2)
ax2.plot(losses)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Energy")
ax2.set_title("Training Progress (Energy Maximization)")
ax2.grid(True)

# 3. Filter frequency response
ax3 = plt.subplot(2, 2, 3)
# Initial filter
filter_model_initial = DifferentiableBandPassFilter(
    sig_len=sig_len,
    fs=fs,
    pha_low_hz=25,
    pha_high_hz=35,
    pha_n_bands=1,
    amp_low_hz=50,
    amp_high_hz=100,
    amp_n_bands=1,
    filter_length=101,
)
with torch.no_grad():
    initial_kernel = filter_model_initial._compute_filters()[0].numpy()
    final_kernel = filter_model._compute_filters()[0].numpy()

# Compute frequency responses
freq_response_initial = np.abs(np.fft.fft(initial_kernel, n=1024))[:512]
freq_response_final = np.abs(np.fft.fft(final_kernel, n=1024))[:512]
freqs = np.linspace(0, fs / 2, 512)

ax3.plot(freqs, freq_response_initial, "b--", label="Initial filter", alpha=0.5)
ax3.plot(freqs, freq_response_final, "g-", label="Learned filter", linewidth=2)
ax3.axvline(x=target_freq, color="r", linestyle="--", label=f"Target: {target_freq} Hz")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Magnitude")
ax3.set_title("Filter Frequency Response")
ax3.set_xlim(0, 50)
ax3.legend()
ax3.grid(True)

# 4. Filtered signals comparison
ax4 = plt.subplot(2, 2, 4)
with torch.no_grad():
    # Filter with initial bands
    filtered_initial = filter_model_initial(target_signal.unsqueeze(0).unsqueeze(0))
    # Filter with learned bands
    filtered_final = filter_model(target_signal.unsqueeze(0).unsqueeze(0))

t_plot = t[:256].numpy()
ax4.plot(t_plot, target_signal[:256].numpy(), "k-", label="Original", alpha=0.5)
ax4.plot(
    t_plot,
    filtered_initial[0, 0, 0, :256].numpy(),
    "b--",
    label="Initial filter output",
    alpha=0.7,
)
ax4.plot(
    t_plot,
    filtered_final[0, 0, 0, :256].numpy(),
    "g-",
    label="Learned filter output",
    linewidth=2,
)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude")
ax4.set_title("Filtered Signal Comparison")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig("example_outputs/differentiable_filter_learning.png", dpi=150)
print("\nSaved figure to example_outputs/differentiable_filter_learning.png")

# Save a second figure showing regularization effects
plt.figure(figsize=(10, 6))

# Show multiple filters with overlap penalty
multi_filter = DifferentiableBandPassFilter(
    sig_len=sig_len,
    fs=fs,
    pha_low_hz=5,
    pha_high_hz=30,
    pha_n_bands=4,
    amp_low_hz=40,
    amp_high_hz=120,
    amp_n_bands=4,
    filter_length=101,
)

with torch.no_grad():
    bands_info = multi_filter.get_filter_banks()
    all_bands = bands_info["all_bands"].numpy()

    # Plot bands
    for i, (low, high) in enumerate(all_bands):
        color = "blue" if i < 4 else "red"
        label = "Phase bands" if i == 0 else ("Amplitude bands" if i == 4 else "")
        plt.barh(
            i,
            high - low,
            left=low,
            height=0.8,
            color=color,
            alpha=0.6,
            label=label,
        )
        plt.text(
            (low + high) / 2,
            i,
            f"{low:.0f}-{high:.0f}",
            ha="center",
            va="center",
        )

plt.xlabel("Frequency (Hz)")
plt.ylabel("Band Index")
plt.title("Multi-band Filter Configuration")
plt.xlim(0, 130)
plt.grid(True, axis="x")
plt.legend()
plt.tight_layout()
plt.savefig("example_outputs/multi_band_filter_config.png", dpi=150)
print("Saved figure to example_outputs/multi_band_filter_config.png")

print(f"\n--- Results ---")
print(f"Target frequency: {target_freq} Hz")
print(f"Initial band: [25.0, 35.0] Hz")
print(f"Learned band: [{final_band[0]:.1f}, {final_band[1]:.1f}] Hz")
print(f"Band center moved from 30.0 Hz to {(final_band[0] + final_band[1])/2:.1f} Hz")

# EOF
