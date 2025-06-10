#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 19:18:16 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_setup_parameters.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_setup_parameters.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import itertools
import random

PARAMS_BASE = {
    # Benchmarking Parameters
    "package": "gpac",
    "n_batches": 4,
    # Signal Size Parameters
    "batch_size": 4,
    "n_channels": 8,
    "duration_sec": 4.0,
    "fs": 512,
    # PAC Resolution Parameters
    "pha_n_bands": 16,
    "amp_n_bands": 16,
    # Computation Parameters
    "n_perm": 0,
    "fp16": False,
    # Computation Parameters for gPAC
    "device": "cuda",
    "device_ids": [0],
    "trainable": False,
    # Computation Parameters for Tensorpac
    "n_cpus": 32,
}

PARAMS_GRID = {
    # Benchmarking Parameters
    "package": ["gpac", "tensorpac"],
    # Computation Parameters
    "fp16": [False, True],
    # Computation Parameters for gPAC
    # "device": ["cuda", "cpu"],
    "device": ["cuda"],
    "trainable": [False, True],
    "device_ids": [[0], [0, 1, 2, 3]],
    # Computation Parameters for Tensorpac
    "n_cpus": [16, 32],
}

PARAMS_VARIATION = {
    # Benchmarking Parameters
    "n_batches": [4, 16],
    # Signal Size Parameters
    "batch_size": [1, 4, 16],
    "n_channels": [1, 4, 16],
    "duration_sec": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    "fs": [512, 1024],
    # PAC Resolution Parameters - reduced ranges
    "pha_n_bands": [16, 32, 64],
    "amp_n_bands": [16, 32, 64],
    # Computation Parameters
    "n_perm": [0, 8, 16, 32, 64],
}


def setup_parameters_grid(
    grid_params=PARAMS_GRID, shuffle=False, random_seed=42, quick=False
):
    # Quick mode reduces parameter space
    params_variation = PARAMS_VARIATION.copy()
    if quick:
        for param_name, param_values in params_variation.items():
            params_variation[param_name] = param_values[::2]

    param_sets = []

    # Generate grid combinations for specified parameters
    values_grid = [grid_params[param] for param in grid_params]
    combinations_grid = list(itertools.product(*values_grid))

    for grid_combo in combinations_grid:
        # Create base config with grid parameters
        params_base = PARAMS_BASE.copy()
        for param_idx, param_name in enumerate(grid_params):
            params_base[param_name] = grid_combo[param_idx]

        # # Filter invalid combinations and clean parameters
        # if params_base["package"] == "tensorpac":
        #     if (
        #         params_base.get("device") == "cuda"
        #         or params_base.get("trainable") == True
        #     ):
        #         continue
        #     params_base.pop("device", None)
        #     params_base.pop("device_ids", None)
        #     params_base.pop("trainable", None)
        # else:
        #     # For gpac, skip multi-GPU when device is cpu
        #     if (
        #         params_base.get("device") == "cpu"
        #         and len(params_base.get("device_ids", [0])) > 1
        #     ):
        #         continue
        #     params_base.pop("n_cpus", None)

        param_sets.append(params_base.copy())

        # Vary remaining parameters one at a time
        for param_name, param_values in params_variation.items():
            # Skip package-specific parameters
            if params_base["package"] == "tensorpac" and param_name in [
                "device",
                "device_ids",
                "trainable",
            ]:
                continue
            if params_base["package"] == "gpac" and param_name == "n_cpus":
                continue

            for param_value in param_values:
                if params_base.get(param_name) == param_value:
                    continue

                param_set = params_base.copy()
                param_set[param_name] = param_value
                param_sets.append(param_set)

    if shuffle:
        random.seed(random_seed)
        random.shuffle(param_sets)

    return param_sets


if __name__ == "__main__":
    from pprint import pprint

    import pandas as pd

    for quick in [False, True]:
        param_space = setup_parameters_grid(quick=quick, shuffle=True)
        print(f"Quick Mode: {quick}")
        print(f"Number of parameter combinations: {len(param_space)}")
        print(f"The first two combinations:")
        pprint(param_space[:2])

    df = pd.DataFrame(param_space)

# EOF
