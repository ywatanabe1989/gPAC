#!/usr/bin/env python3
"""
Example: Accessing frequency band definitions in gPAC

This example demonstrates how to access the full frequency band definitions
(not just center frequencies) from the PAC calculator.
"""

import torch
import numpy as np
from gpac import PAC


def main():
    # Initialize PAC calculator with explicit bands
    pha_bands = [[4, 8], [8, 12], [12, 30]]
    amp_bands = [[30, 50], [50, 80], [80, 120], [120, 180]]
    
    pac = PAC(
        seq_len=2000,
        fs=1000,
        pha_bands_hz=pha_bands,
        amp_bands_hz=amp_bands,
        n_perm=None  # No permutation testing for this example
    )
    
    # Access band definitions directly from PAC object
    print("Accessing frequency bands from PAC object:")
    print(f"Phase bands (Hz): \n{pac.pha_bands_hz}")
    print(f"Amplitude bands (Hz): \n{pac.amp_bands_hz}")
    
    # The bands are tensors, so you can perform operations on them
    print("\nBand widths:")
    pha_widths = pac.pha_bands_hz[:, 1] - pac.pha_bands_hz[:, 0]
    amp_widths = pac.amp_bands_hz[:, 1] - pac.amp_bands_hz[:, 0]
    print(f"Phase band widths (Hz): {pha_widths}")
    print(f"Amplitude band widths (Hz): {amp_widths}")
    
    # Generate some test data
    batch_size = 2
    n_channels = 4
    data = torch.randn(batch_size, n_channels, 2000)
    
    # Compute PAC
    result = pac(data)
    
    # Band definitions are also included in the output
    print("\nBands from forward pass output:")
    print(f"Phase bands shape: {result['phase_bands_hz'].shape}")
    print(f"Amplitude bands shape: {result['amplitude_bands_hz'].shape}")
    
    # The center frequencies are still available for backward compatibility
    print("\nCenter frequencies (for backward compatibility):")
    print(f"Phase center frequencies: {result['phase_frequencies']}")
    print(f"Amplitude center frequencies: {result['amplitude_frequencies']}")
    
    # You can also still access them via properties
    print(f"Phase mids (via property): {pac.pha_mids}")
    print(f"Amplitude mids (via property): {pac.amp_mids}")
    
    # Example: Find which band contains a specific frequency
    target_freq = 10.0  # Hz
    print(f"\nFinding phase band containing {target_freq} Hz:")
    for i, (low, high) in enumerate(pac.pha_bands_hz):
        if low <= target_freq <= high:
            print(f"  Band {i}: [{low:.1f}, {high:.1f}] Hz")
    
    # Example: Using range-based band generation
    print("\n" + "="*50)
    print("Using automatic band generation with ranges:")
    
    pac_auto = PAC(
        seq_len=2000,
        fs=1000,
        pha_range_hz=(4, 30),
        amp_range_hz=(30, 180),
        pha_n_bands=6,
        amp_n_bands=8,
    )
    
    print(f"Auto-generated phase bands:\n{pac_auto.pha_bands_hz}")
    print(f"Auto-generated amplitude bands:\n{pac_auto.amp_bands_hz}")


if __name__ == "__main__":
    main()