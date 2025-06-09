#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-08 21:42:34 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/parameter_sweep_benchmark_measure_compuation_time.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/parameter_sweep_benchmark_measure_compuation_time.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import time

import torch

"""Imports"""
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.append("../../../src")


def measure_computation_time(
    package: str = None,
    pac_calculator=None,
    dataloader=None,
    device: str = None,
) -> tuple:
    def _measure_computation_time_gpac(
        dataloader, device, pac_calculator_gpac
    ):
        # Warm-up run
        with torch.no_grad():
            signal, labels, metadata = next(iter(dataloader))
            _ = pac_calculator_gpac(signal.to(device))

        # Main computation
        comp_start_time = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                signal, labels, metadata = batch
                output_gp = pac_calculator_gpac(signal.to(device))
        comp_time = time.time() - comp_start_time

        return comp_time, output_gp

    def _measure_computation_time_tensorpac(
        dataloader, pac_calculator_tensorpac
    ):
        print("Implement me")
        return None, None

    if package == "gpac":
        return _measure_computation_time_gpac(
            dataloader, device, pac_calculator
        )
    elif package == "tensorpac":
        return _measure_computation_time_tensorpac(dataloader, pac_calculator)
    else:
        raise ValueError("package must be either gpac or tensorpac")

# EOF
