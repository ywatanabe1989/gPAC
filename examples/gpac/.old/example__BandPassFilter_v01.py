#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 16:26:57 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/example_bandpass_filter.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/example_bandpass_filter.py"
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


def demo_static_bandpass_filter() -> None:
    """Demonstrate PAC batch generation."""
    mngs.str.printc("=== Demo BandPassFilter ===", c="yellow")

    # Arange
    # --------------------
    ## Data
    batch = gpac.dataset.generate_pac_batch()
    signal, label, metadata = batch

    ## Instantiation
    filter = gpac._BandPassFilter(
        fs=metadata["fs"][0],
        pha_range_hz=(1.0, 30.0),
        amp_range_hz=(30.0, 150.0),
        pha_n_bands=10,
        amp_n_bands=20,
        trainable=False,
    ).cuda()
    pprint(filter.info)

    # Act
    # --------------------
    filted = filter(signal.cuda())

    # Assert
    # --------------------
    n_bands_total = filter.info["pha_n_bands"] + filter.info["amp_n_bands"]
    expected_shape = signal.shape[:-1] + (n_bands_total, signal.shape[-1])
    assert filted.shape == expected_shape

    __import__("ipdb").set_trace()


def demo_trainable_bandpass_filter() -> None:
    """Demonstrate PAC batch generation."""
    mngs.str.printc("=== Demo BandPassFilter ===", c="yellow")
    batch = gpac.dataset.generate_pac_batch()
    signal, label, metadata = batch
    __import__("ipdb").set_trace()
    gpac._BandPassFilter(
        fs=metadata["fs"][0],
    )


def main() -> int:
    """Main function to demonstrate all PAC data generation methods."""
    fig = demo_static_bandpass_filter()
    fig = demo_trainable_bandpass_filter()

    return 0


if __name__ == "__main__":
    main()

# EOF
