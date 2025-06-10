#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 23:25:09 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_init_pac_calculator.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_init_pac_calculator.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

sys.path.append("../../../src")
import gpac
import tensorpac


def _create_frequency_bands(
    pha_range_hz, pha_n_bands, amp_range_hz, amp_n_bands
):
    pha_edges = np.linspace(*pha_range_hz, pha_n_bands + 1)
    amp_edges = np.linspace(*amp_range_hz, amp_n_bands + 1)
    pha_bands_hz = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands_hz = np.c_[amp_edges[:-1], amp_edges[1:]]
    return pha_bands_hz, amp_bands_hz


def _init_pac_calculator_gpac(
    duration_sec,
    fs,
    pha_bands_hz,
    amp_bands_hz,
    n_perm,
    fp16,
    device,
    device_ids,
    trainable,
):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pac_calculator = gpac.PAC(
        seq_len=int(duration_sec * fs),
        fs=fs,
        pha_bands_hz=pha_bands_hz,
        amp_bands_hz=amp_bands_hz,
        n_perm=n_perm,
        fp16=fp16,
        device_ids=device_ids,
        trainable=trainable,
    ).to(device)

    return pac_calculator


def _init_pac_calculator_tensorpac(pha_bands_hz, amp_bands_hz, n_perm):
    pac_calculator = tensorpac.Pac(
        idpac=(2, 0, 0),
        f_pha=pha_bands_hz,
        f_amp=amp_bands_hz,
        dcomplex="hilbert",
        verbose=False,
    )

    return pac_calculator


def init_pac_calculator(
    package="gpac",
    n_batches=3,
    n_samples=16,
    batch_size=4,
    n_channels=4,
    n_segments=4,
    duration_sec=4,
    fs=512,
    pha_range_hz=(2.0, 30.0),
    pha_n_bands=30,
    amp_range_hz=(30.0, 230.0),
    amp_n_bands=30,
    n_perm=4,
    fp16=False,
    device="cuda",
    device_ids="all",
    trainable=False,
):
    pha_bands_hz, amp_bands_hz = _create_frequency_bands(
        pha_range_hz, pha_n_bands, amp_range_hz, amp_n_bands
    )

    if package == "gpac":
        return _init_pac_calculator_gpac(
            duration_sec,
            fs,
            pha_bands_hz,
            amp_bands_hz,
            n_perm,
            fp16,
            device,
            device_ids,
            trainable,
        )
    elif package == "tensorpac":
        return _init_pac_calculator_tensorpac(
            pha_bands_hz, amp_bands_hz, n_perm
        )
    else:
        raise ValueError("package must be either gpac or tensorpac")


if __name__ == "__main__":
    pac_calculator_gpac = init_pac_calculator(package="gpac")
    pac_calculator_tensorpac = init_pac_calculator(package="tensorpac")

# EOF
