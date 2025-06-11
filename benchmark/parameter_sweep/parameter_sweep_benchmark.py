#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 14:48:42 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/parameter_sweep/parameter_sweep_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/parameter_sweep/parameter_sweep_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import time
import warnings

import torch

"""
Systematic parameter sweep benchmark comparing gPAC and TensorPAC performance.

This module conducts comprehensive benchmarking by varying parameters one at a time
to analyze performance scaling characteristics between gPAC and TensorPAC implementations.

Functionalities:
    - Parameter sweep across computation dimensions (batch size, channels, etc.)
    - Performance timing for both GPU (gPAC) and CPU (TensorPAC) implementations
    - Systematic comparison with configurable parameter grids
    - Results aggregation and periodic saving

Dependencies:
    - packages: gpac, tensorpac, torch, numpy, matplotlib, mngs, yaml
    - helper modules: _parameter_sweep_helper_init_pac_calculator, _parameter_sweep_helper_setup_parameters

Input/Output:
    - input: Generated synthetic PAC data via gpac.dataset
    - output: ./benchmark_results.pkl (timing results and sample comodulogramss)
"""

"""Imports"""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.append("../../../src")
sys.path.append(__DIR__)

import gpac
import mngs
from _parameter_sweep_helper_init_pac_calculator import init_pac_calculator
from _parameter_sweep_helper_resource_monitor import \
    monitor_resources_during_computation
from _parameter_sweep_helper_setup_parameters import setup_parameters_grid
from tqdm import tqdm

os.environ["TORCH_LOGS"] = "recompiles"


def benchmark_one_condition(
    # Parameters for Benchmarking
    package: str = "gpac",
    n_batches: int = 3,
    # Parameters for Data Genaration
    n_samples: int = 16,
    batch_size: int = 4,
    n_channels: int = 4,
    n_segments: int = 4,
    duration_sec: int = 4,
    fs: int = 512,
    # Parameters for PAC Calculation
    pha_range_hz: float = (2.0, 30.0),
    pha_n_bands: int = 30,
    amp_range_hz: int = (30.0, 230.0),
    amp_n_bands: int = 30,
    n_perm: int = 4,
    fp16: bool = False,
    # Parameters for Tensorpac Calculation
    n_cpus: int = 32,
    # Parameters for gPAC Calculation
    device: str = "cuda",
    device_ids: list = None,
    trainable: bool = False,
) -> tuple:
    """Benchmark gPAC with given parameters using correct API."""

    # DataLoader Initialization
    dataloader = gpac.dataset.generate_pac_dataloader(
        n_samples=n_samples,
        batch_size=batch_size,
        n_channels=n_channels,
        n_segments=n_segments,
        duration_sec=duration_sec,
        fs=fs,
        pac_config=gpac.dataset.single_class_single_pac_config,
    )

    # PAC Calculator Initialization
    pac_calculator = init_pac_calculator(
        package=package,
        n_batches=n_batches,
        n_samples=n_samples,
        batch_size=batch_size,
        n_channels=n_channels,
        n_segments=n_segments,
        duration_sec=duration_sec,
        fs=fs,
        pha_range_hz=pha_range_hz,
        pha_n_bands=pha_n_bands,
        amp_range_hz=amp_range_hz,
        amp_n_bands=amp_n_bands,
        n_perm=n_perm,
        fp16=fp16,
        device=device,
        device_ids=device_ids,
        trainable=trainable,
    )

    # Main
    calculation_time, sample_comodulogram, resource_usage = compute(
        package=package,
        pac_calculator=pac_calculator,
        dataloader=dataloader,
        n_batches=n_batches,
        n_perm=n_perm,
        device=device,
        n_cpus=n_cpus,
    )

    assert sample_comodulogram.shape == (
        pha_n_bands,
        amp_n_bands,
    )

    return calculation_time, sample_comodulogram, resource_usage


def compute(
    package: str = None,
    pac_calculator=None,
    dataloader=None,
    n_batches=4,
    n_perm=0,
    n_cpus=32,
    device: str = None,
) -> tuple:

    def _compute_gpac(dataloader, device, pac_calculator_gpac):
        with torch.no_grad():
            batch = next(iter(dataloader))
            signal, labels, metadata = batch
            _ = pac_calculator_gpac(signal.to(device))

        comp_start_time = time.time()
        with torch.no_grad():
            for i_batch, batch in enumerate(dataloader):
                signal, labels, metadata = batch
                last_output_gp = pac_calculator_gpac(signal.to(device))
                if i_batch == (n_batches - 1):
                    break
        comp_time = time.time() - comp_start_time

        last_output_np = last_output_gp["pac"].detach().cpu().numpy()
        i_batch, i_channel, i_segment = 0, 0, 0
        first_comodulogram = last_output_np[i_batch, i_channel, i_segment]

        return comp_time, first_comodulogram

    def _pre_reshape_signal_for_tensorpac(signal):
        batch_size, n_channels, n_segments, seq_len = signal.shape
        signal_reshaped = signal.reshape(-1, seq_len)
        return signal_reshaped, (batch_size, n_channels, n_segments, seq_len)

    def _post_reshape_pac_result_for_tensorpac(
        output_tp, orig_shape, n_pha_bands, n_amp_bands
    ):
        batch_size, n_channels, n_segments, _ = orig_shape
        output_transposed = output_tp.transpose(2, 1, 0)
        return output_transposed.reshape(
            batch_size, n_channels, n_segments, n_pha_bands, n_amp_bands
        )

    def _compute_tensorpac(
        dataloader, pac_calculator_tensorpac, n_perm, n_cpus
    ):
        batch = next(iter(dataloader))
        signal, labels, metadata = batch
        signal_reshaped, orig_shape = _pre_reshape_signal_for_tensorpac(signal)

        output_tp = pac_calculator_tensorpac.filterfit(
            sf=metadata["fs"][0].item(),
            x_pha=signal_reshaped,
            n_perm=n_perm,
            n_jobs=n_cpus,
            random_state=42,
            verbose=False,
        )

        n_pha_bands = len(pac_calculator_tensorpac.xvec)
        n_amp_bands = len(pac_calculator_tensorpac.yvec)

        output_tp = _post_reshape_pac_result_for_tensorpac(
            output_tp, orig_shape, n_pha_bands, n_amp_bands
        )

        start_time = time.time()
        for i_batch, batch in enumerate(dataloader):
            signal, labels, metadata = batch
            signal_reshaped, orig_shape = _pre_reshape_signal_for_tensorpac(
                signal
            )

            last_output_tp = pac_calculator_tensorpac.filterfit(
                sf=metadata["fs"][0].item(),
                x_pha=signal_reshaped,
                x_amp=signal_reshaped,
                n_perm=n_perm,
                n_jobs=n_cpus,
                random_state=42,
                verbose=False,
            )
            last_output_tp = _post_reshape_pac_result_for_tensorpac(
                last_output_tp, orig_shape, n_pha_bands, n_amp_bands
            )
            if i_batch == (n_batches - 1):
                break
        comp_time = time.time() - start_time

        i_batch, i_channel, i_segment = 0, 0, 0
        first_comodulogram = last_output_tp[i_batch, i_channel, i_segment]

        return comp_time, first_comodulogram

    def _compute_with_monitoring(compute_func, *args, **kwargs):
        return monitor_resources_during_computation(
            compute_func, *args, **kwargs
        )

    if package == "gpac":
        result, resources = _compute_with_monitoring(
            _compute_gpac, dataloader, device, pac_calculator
        )
    elif package == "tensorpac":
        result, resources = _compute_with_monitoring(
            _compute_tensorpac, dataloader, pac_calculator, n_perm, n_cpus
        )
    else:
        raise ValueError("package must be either gpac or tensorpac")

    comp_time, first_comodulogram = result
    return comp_time, first_comodulogram, resources


def main(args):
    params_spaces = setup_parameters_grid(
        quick=args.quick, shuffle=args.shuffle
    )
    agg = mngs.dict.listed_dict()
    save_counter = 0
    for params_space in tqdm(params_spaces, total=len(params_spaces)):
        mngs.str.printc(
            (f"Parameters:\n{params_space}"),
            c="yellow",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            comp_time, sample_comodulogram, resource_usage = (
                benchmark_one_condition(**params_space)
            )
            comp_time_per_batch = comp_time / params_space["n_batches"]

        mngs.str.printc(
            f"Comodulogram Shape: {sample_comodulogram.shape}\n"
            f"Computation Time [sec]: {comp_time:.3f} = ({comp_time_per_batch:.3f} [sec/batch])",
            c="green",
        )

        agg["computation_time_for_all_runs"].append(comp_time)
        agg["sample_comodulogram"].append(sample_comodulogram)

        for resource_key, resource_value in resource_usage.items():
            agg[resource_key].append(resource_value)

        for k, v in params_space.items():
            agg[k].append(v)

        if (save_counter + 1) % 10 == 0:
            mngs.io.save(agg, "benchmark_results.pkl")
        save_counter += 1

    mngs.io.save(agg, "benchmark_results.pkl")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    parser = argparse.ArgumentParser(
        description="Parameter sweep benchmark for gPAC"
    )
    parser.add_argument(
        "--no-quick",
        action="store_true",
        help="Quick test with fewer parameter values",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Quick test with fewer parameter values",
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    args.quick = not args.no_quick
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Start mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    # Main
    exit_status = main(args)

    # Close the mngs framework
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
