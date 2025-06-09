#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 21:50:38 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helpers.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helpers.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import torch

"""Imports"""
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.append("../../../src")
import gpac

sys.path.append(__DIR__)

from _parameter_sweep_benchmark_init_pac_calculator import init_pac_calculator
from _parameter_sweep_benchmark_measure_computation_time import \
    measure_computation_time


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
    pha_start_hz: float = 2.0,
    pha_end_hz: float = 30.0,
    pha_n_bands: int = 30,
    amp_start_hz: int = 30,
    amp_end_hz: int = 230,
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
        pha_start_hz: float = 2.0,
        pha_end_hz: float = 30.0,
        pha_n_bands: int = 30,
        amp_start_hz: int = 30,
        amp_end_hz: int = 230,
        amp_n_bands: int = 30,
        n_perm: int = 4,
        fp16: bool = False,
        device: str = "cuda",
        multi_gpu: bool = False,
        device_ids: list = None,
    ):

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize PAC Calculator
        if device_ids is None:
            device_ids = "all" if multi_gpu else [0]
        pac_calculator_gpac = gpac.PAC(
            seq_len=int(duration_sec * fs),
            fs=fs,
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
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
        pha_start_hz: float = 2.0,
        pha_end_hz: float = 30.0,
        pha_n_bands: int = 30,
        amp_start_hz: int = 30,
        amp_end_hz: int = 230,
        amp_n_bands: int = 30,
        n_perm: int = 4,
        fp16: bool = False,
        multi_gpu: bool = False,
    ):
        # Initialize PAC Calculator
        # fixme
        pac_calculator_tensorpac = None
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
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            fp16=fp16,
            device=device,
            multi_gpu=multi_gpu,
            device_ids=device_ids,
        )
    elif package == "tensorpac":
        return _init_pac_calculator_tensorpac(
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
            pha_start_hz=pha_start_hz,
            pha_end_hz=pha_end_hz,
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_start_hz,
            amp_end_hz=amp_end_hz,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            fp16=fp16,
            device=device,
            multi_gpu=multi_gpu,
            device_ids=device_ids,
        )
    else:
        raise ValueError("package must be either gpac or tensorpac")

# EOF
