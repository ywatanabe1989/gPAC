#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 01:00:43 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_init_pac_calculator.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_init_pac_calculator.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np

"""Imports"""
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.append("../../../src")

import torch

matplotlib.use("Agg")

sys.path.append("../../../src")
import gpac
import tensorpac


def init_pac_calculator(
    # Parameters for Benchmarking
    package: str = "gpac",
    n_runs: int = 3,
    # Parameters for Data Genaration
    n_samples: int = 16,
    batch_size: int = 4,
    n_channels: int = 4,
    n_segments: int = 4,
    duration_sec: int = 4,
    fs: int = 512,
    # Parameters for PAC Calculation
    pha_range_hz: list = (2.0, 30.0),
    pha_n_bands: int = 30,
    amp_range_hz: list = (30.0, 230.0),
    amp_n_bands: int = 30,
    device: str = "cuda",
    device_ids: list = None,
    n_perm: int = 4,
    fp16: bool = False,
    multi_gpu: bool = False,
):
    def _init_pac_calculator_gpac(
        # Parameters for Benchmarking
        package: str = "gpac",
        n_runs: int = 3,
        # Parameters for Data Genaration
        n_samples: int = 16,
        batch_size: int = 4,
        n_channels: int = 4,
        n_segments: int = 4,
        duration_sec: int = 4,
        fs: int = 512,
        # Parameters for PAC Calculation
        pha_range_hz: list = (2.0, 30.0),
        pha_n_bands: int = 30,
        amp_range_hz: list = (30.0, 230.0),
        amp_n_bands: int = 30,
        n_perm: int = 4,
        fp16: bool = False,
        device: str = "cuda",
        multi_gpu: bool = False,
        device_ids: list = None,
    ):

        # Define explicit frequency bands to match gPAC (sequential non-overlapping)
        pha_edges = np.linspace(*pha_range_hz, pha_n_bands + 1)
        amp_edges = np.linspace(*amp_range_hz, amp_n_bands + 1)

        # Convert to band pairs for TensorPAC
        pha_bands_hz = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands_hz = np.c_[amp_edges[:-1], amp_edges[1:]]

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize PAC Calculator
        if device_ids is None:
            device_ids = "all" if multi_gpu else [0]
        pac_calculator_gpac = gpac.PAC(
            seq_len=int(duration_sec * fs),
            fs=fs,
            # pha_range_hz=pha_range_hz,
            # pha_n_bands=pha_n_bands,
            # amp_range_hz=amp_range_hz,
            # amp_n_bands=amp_n_bands,
            pha_bands_hz=pha_bands_hz,
            amp_bands_hz=amp_bands_hz,
            device_ids=device_ids,
            n_perm=n_perm,
            fp16=fp16,
        ).to(device)
        return pac_calculator_gpac

    def _init_pac_calculator_tensorpac(
        # Parameters for Data Genaration
        n_samples: int = 16,
        batch_size: int = 4,
        n_channels: int = 4,
        n_segments: int = 4,
        duration_sec: int = 4,
        fs: int = 512,
        # Parameters for PAC Calculation
        pha_range_hz: list = (2.0, 30.0),
        pha_n_bands: int = 30,
        amp_range_hz: list = (30.0, 230.0),
        amp_n_bands: int = 30,
        n_perm: int = 4,
        fp16: bool = False,
        multi_gpu: bool = False,
    ):

        # Define explicit frequency bands to match gPAC (sequential non-overlapping)
        pha_edges = np.linspace(*pha_range_hz, pha_n_bands + 1)
        amp_edges = np.linspace(*amp_range_hz, amp_n_bands + 1)

        # Convert to band pairs for TensorPAC
        pha_bands_hz = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands_hz = np.c_[amp_edges[:-1], amp_edges[1:]]

        # Initialize PAC with explicit bands
        pac_calculator_tensorpac = tensorpac.Pac(
            idpac=(2, 0, 0),
            f_pha=pha_bands_hz,
            f_amp=amp_bands_hz,
            dcomplex="hilbert",
            verbose=False,
        )
        return pac_calculator_tensorpac

    if package == "gpac":
        return _init_pac_calculator_gpac(
            # Parameters for Benchmarking
            package=package,
            n_runs=n_runs,
            # Parameters for Data Genaration
            n_samples=n_samples,
            batch_size=batch_size,
            n_channels=n_channels,
            n_segments=n_segments,
            duration_sec=duration_sec,
            fs=fs,
            # Parameters for PAC Calculation
            pha_range_hz=pha_range_hz,
            pha_n_bands=pha_n_bands,
            amp_range_hz=amp_range_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            fp16=fp16,
            device=device,
            multi_gpu=multi_gpu,
            device_ids=device_ids,
        )
    elif package == "tensorpac":
        return _init_pac_calculator_tensorpac(
            # Parameters for Data Genaration
            n_samples=n_samples,
            batch_size=batch_size,
            n_channels=n_channels,
            n_segments=n_segments,
            duration_sec=duration_sec,
            fs=fs,
            # Parameters for PAC Calculation
            pha_range_hz=pha_range_hz,
            pha_n_bands=pha_n_bands,
            amp_range_hz=amp_range_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            # fp16=fp16,
            # device=device,
            # multi_gpu=multi_gpu,
            # device_ids=device_ids,
        )
    else:
        raise ValueError("package must be either gpac or tensorpac")


if __name__ == "__main__":
    pac_calculator_gpac = init_pac_calculator(package="gpac")
    pac_calculator_tensorpac = init_pac_calculator(package="tensorpac")

# EOF
