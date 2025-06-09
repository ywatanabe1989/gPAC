#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 15:49:52 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/demo_data_generation.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/demo_data_generation.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionality:
    * Demonstrates PAC dataset generation with different configurations
Input:
    * PAC configuration dictionaries for various class/component combinations
Output:
    * Console output showing dataset properties, shapes, and sample data
Prerequisites:
    * gpac package with dataset generation functionality
"""

from pprint import pprint

import gpac
import mngs


def demo_batch(pac_config: dict) -> None:
    """Demonstrate PAC batch generation."""
    mngs.str.printc("=== Demo Batch ===")
    batch = gpac.dataset.generate_pac_batch(pac_config=pac_config)
    signal, label, metadata = batch
    print(f"Signal shape: {signal.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Metadata keys: {list(metadata.keys())}")
    pprint(label)
    pprint(metadata)
    print()

    # Plotting
    fig, ax = mngs.plt.subplots()
    i_batch, i_ch, i_segment = 0, 0, 0
    ax.plot(signal[i_batch, i_ch, i_segment], label="")
    ax.hide_spines()
    ax.set_n_ticks()
    ax.set_xyt(x="Time", y="Amplitude", t="Signal with PAC")
    return fig


def demo_dataset(pac_config: dict) -> None:
    """Demonstrate PAC dataset generation."""
    mngs.str.printc("=== Demo Dataset ===", c="yellow")
    dataset = gpac.dataset.generate_pac_dataset(pac_config=pac_config)
    print(f"Dataset length: {len(dataset)}")
    signal, label, metadata = dataset[0]
    print(f"Sample shape: {signal.shape}")
    print(f"Label: {label}")
    print()


def demo_dataloader(pac_config: dict) -> None:
    """Demonstrate PAC dataloader generation."""
    mngs.str.printc("=== Demo DataLoader ===", c="yellow")
    dataloader = gpac.dataset.generate_pac_dataloader(pac_config=pac_config)
    for batch_idx, (signal, label, metadata) in enumerate(dataloader):
        print(f"Batch {batch_idx}: signal {signal.shape}, label {label.shape}")
        if batch_idx >= 2:
            break
    print()


def main() -> int:
    """Main function to demonstrate all PAC data generation methods."""
    pac_config = [
        gpac.dataset.single_class_single_pac_config,
        gpac.dataset.single_class_multi_pac_config,
        gpac.dataset.multi_class_single_pac_config,
        gpac.dataset.multi_class_multi_pac_config,
    ][0]

    fig = demo_batch(pac_config)
    mngs.io.save(fig, "demo_batch.gif")

    demo_dataset(pac_config)
    demo_dataloader(pac_config)

    return 0


if __name__ == "__main__":
    main()

# EOF
