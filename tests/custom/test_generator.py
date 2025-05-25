#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gpac import SyntheticDataGenerator

# Create a simple generator
generator = SyntheticDataGenerator(
    fs=256.0,
    duration_sec=2.0,
    n_channels=4,
    n_segments=1,
)

# Generate a simple dataset
data = generator.generate_and_split()
train_dataset = data['train']

# Get a sample
signal, label = train_dataset[0]
print(f"Generated signal shape: {signal.shape}")
print(f"Label: {label}")
print(f"Data type: {type(signal)}")

# Test if we can use it directly with calculate_pac
import gpac
pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
    signal.unsqueeze(0),  # Add batch dimension
    fs=256.0,
    pha_n_bands=10,
    amp_n_bands=10,
)
print(f"PAC values shape: {pac_values.shape}")
print("✅ SyntheticDataGenerator works with calculate_pac!")