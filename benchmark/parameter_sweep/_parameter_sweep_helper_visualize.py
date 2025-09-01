#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-15 10:44:53 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/parameter_sweep/_parameter_sweep_helper_visualize.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/parameter_sweep/_parameter_sweep_helper_visualize.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

import numpy as np
import pandas as pd
import scitex

sys.path.append(__DIR__)
from pprint import pprint

from _parameter_sweep_helper_init_pac_calculator import _create_frequency_bands
from _parameter_sweep_helper_setup_parameters import (PARAMS_BASE, PARAMS_GRID,
                                                      PARAMS_VARIATION)

# Optional: Import legend helper if you want to use the alternative approach
# from legend_helper import save_legend_separately

# As slicing does not support list
PARAMS_BASE["device_ids"] = str(PARAMS_BASE["device_ids"])
PARAMS_GRID["device_ids"] = [str(k) for k in PARAMS_GRID["device_ids"]]


def print_params():
    scitex.str.printc("PARAMS_BASE:", c="yellow")
    pprint(PARAMS_BASE)
    scitex.str.printc("PARAMS_GRID:", c="yellow")
    pprint(PARAMS_GRID)
    scitex.str.printc("PARAMS_VARIATION:", c="yellow")
    pprint(PARAMS_VARIATION)


def plot_parameter_scaling_on_grids(df, sharey=False):
    import matplotlib.pyplot as plt

    grids = list(scitex.utils.yield_grids(PARAMS_GRID))
    n_grids = len(grids)
    n_params = len(PARAMS_VARIATION)

    df["grid_point"] = None

    n_keys = len(PARAMS_VARIATION.keys())
    n_cols = 4
    n_rows = (n_keys + n_cols - 1) // n_cols

    fig, axes = scitex.plt.subplots(
        ncols=n_cols, nrows=n_rows, sharey=sharey, sharex=False
    )

    # Create color palette based on package type
    unique_grids = []
    for i_tgt_key, tgt_key in enumerate(PARAMS_VARIATION.keys()):
        df_key = df.copy()
        base_for_tgt = PARAMS_BASE.copy()
        base_for_tgt.pop(tgt_key)

        for grid in grids:
            base_for_grid = base_for_tgt.copy()
            base_for_grid.update(grid)

            indi = scitex.pd.find_indi(df_key, base_for_grid)
            grid_str = scitex.dict.to_str(grid)
            df_key.loc[indi, "grid_point"] = grid_str
            if grid_str not in unique_grids:
                unique_grids.append(grid_str)

        df_key = df_key.sort_values(["grid_point", tgt_key])

        # Create color palette with gradation
        tensorpac_grids = [g for g in unique_grids if "tensorpac" in g]
        gpac_grids = [g for g in unique_grids if "tensorpac" not in g]

        red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(tensorpac_grids)))
        blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(gpac_grids)))

        palette = {}
        for idx, grid_str in enumerate(tensorpac_grids):
            palette[grid_str] = red_colors[idx]
        for idx, grid_str in enumerate(gpac_grids):
            palette[grid_str] = blue_colors[idx]

        ax = axes.flat[i_tgt_key]
        ax.sns_scatterplot(
            data=df_key,
            x=tgt_key,
            y="computation_time_for_all_runs",
            hue="grid_point",
            palette=palette,
        )

        ax.set_xyt(scitex.str.format_plot_text(tgt_key), None, None)
        ax.legend("separate")

    fig.supxyt(
        "Variable",
        "Computation time [sec]",
        "gPAC vs Tensorpac Speed Comparison",
    )
    return fig


# def plot_parameter_scaling_on_grids(df):
#     grids = list(scitex.utils.yield_grids(PARAMS_GRID))
#     n_grids = len(grids)
#     n_params = len(PARAMS_VARIATION)

#     df["grid_point"] = None

#     n_keys = len(PARAMS_VARIATION.keys())
#     n_cols = 4
#     n_rows = (n_keys + n_cols - 1) // n_cols

#     fig, axes = scitex.plt.subplots(
#         ncols=n_cols, nrows=n_rows, sharey=True, sharex=False
#     )

#     # Create color palette based on package type
#     unique_grids = []
#     for i_tgt_key, tgt_key in enumerate(PARAMS_VARIATION.keys()):
#         df_key = df.copy()
#         base_for_tgt = PARAMS_BASE.copy()
#         base_for_tgt.pop(tgt_key)

#         for grid in grids:
#             base_for_grid = base_for_tgt.copy()
#             base_for_grid.update(grid)

#             indi = scitex.pd.find_indi(df_key, base_for_grid)
#             grid_str = scitex.dict.to_str(grid)
#             df_key.loc[indi, "grid_point"] = grid_str
#             if grid_str not in unique_grids:
#                 unique_grids.append(grid_str)

#         df_key = df_key.sort_values(["grid_point", tgt_key])

#         # Create color palette
#         palette = {}
#         for grid_str in unique_grids:
#             if "tensorpac" in grid_str:
#                 palette[grid_str] = "red"
#             else:
#                 palette[grid_str] = "blue"

#         ax = axes.flat[i_tgt_key]
#         ax.sns_lineplot(
#             data=df_key,
#             x=tgt_key,
#             y="computation_time_for_all_runs",
#             hue="grid_point",
#             palette=palette,
#         )
#         ax.set_xyt(tgt_key, None, None)
#         ax.legend("separate")

#     fig.supxyt(
#         "Variable",
#         "Computation time [sec]",
#         "gPAC vs Tensorpac Speed Comparison",
#     )
#     return fig

# def plot_parameter_scaling_on_grids(df):
#     grids = list(scitex.utils.yield_grids(PARAMS_GRID))
#     n_grids = len(grids)
#     n_params = len(PARAMS_VARIATION)

#     df["grid_point"] = None

#     n_keys = len(PARAMS_VARIATION.keys())
#     n_cols = 4
#     n_rows = (n_keys + n_cols - 1) // n_cols

#     fig, axes = scitex.plt.subplots(
#         ncols=n_cols, nrows=n_rows, sharey=True, sharex=False
#     )

#     for i_tgt_key, tgt_key in enumerate(PARAMS_VARIATION.keys()):
#         df_key = df.copy()
#         base_for_tgt = PARAMS_BASE.copy()
#         base_for_tgt.pop(tgt_key)

#         for grid in grids:
#             base_for_grid = base_for_tgt.copy()
#             base_for_grid.update(grid)

#             indi = scitex.pd.find_indi(df_key, base_for_grid)
#             df_key.loc[indi, "grid_point"] = scitex.dict.to_str(grid)

#         # Sort by Variable
#         df_key.sort_values(["grid_point", tgt_key])

#         # Plot
#         ax = axes.flat[i_tgt_key]
#         ax.sns_lineplot(
#             data=df_key,
#             x=tgt_key,
#             y="computation_time_for_all_runs",
#             hue="grid_point",
#             # legend=True,  # Ensure legend is created
#         )
#         ax.set_xyt(tgt_key, None, None)
#         ax.legend("separate")

#     fig.supxyt(
#         "Variable",
#         "Computation time [sec]",
#         "gPAC vs Tensorpac Speed Comparison",
#     )
#     return fig


def plot_comodulograms(df):
    params_gpac = PARAMS_BASE.copy()
    params_tensorpac = params_gpac.copy()
    params_tensorpac["package"] = "tensorpac"

    comodulogram_gpac = scitex.pd.slice(df, params_gpac)[
        "sample_comodulogram"
    ].iloc[0]
    comodulogram_tensorpac = scitex.pd.slice(df, params_tensorpac)[
        "sample_comodulogram"
    ].iloc[0]
    assert comodulogram_gpac.shape == comodulogram_tensorpac.shape

    vmin = min(comodulogram_gpac.min(), comodulogram_tensorpac.min())
    vmax = max(comodulogram_gpac.max(), comodulogram_tensorpac.max())

    pha_range_hz = (2.0, 30.0)
    pha_n_bands = comodulogram_gpac.shape[0]
    amp_range_hz = (30.0, 230.0)
    amp_n_bands = comodulogram_gpac.shape[1]
    pha_bands_hz, amp_bands_hz = _create_frequency_bands(
        pha_range_hz, pha_n_bands, amp_range_hz, amp_n_bands
    )
    pha_mids_hz = pha_bands_hz.mean(axis=-1)
    amp_mids_hz = amp_bands_hz.mean(axis=-1)

    fig, axes = scitex.plt.subplots(ncols=2)
    comodulograms = [comodulogram_gpac, comodulogram_tensorpac]
    titles = ["gPAC", "Tensorpac"]

    for idx, (comodulogram, title) in enumerate(zip(comodulograms, titles)):
        axes[idx].plot_image(comodulogram, label=title, vmin=vmin, vmax=vmax)
        axes[idx].set_xyt(t=title)
        axes[idx].set_xticks(
            range(len(pha_mids_hz)), [f"{freq:.1f}" for freq in pha_mids_hz]
        )
        axes[idx].set_yticks(
            range(len(amp_mids_hz)), [f"{freq:.1f}" for freq in amp_mids_hz]
        )
        axes[idx].set_n_ticks()

    fig.supxyt(
        "Frequency for Phase [Hz]",
        "Frequency for Amplitude [Hz]",
        "Phase-Amplitude Coupling Comodulogram",
    )

    return fig


def main(args):
    print_params()
    df = pd.DataFrame(
        scitex.io.load(
            "./benchmark/parameter_sweep/parameter_sweep_benchmark_out/benchmark_results.pkl"
        )
    )
    df["device_ids"] = df["device_ids"].astype(str)

    fig1 = plot_parameter_scaling_on_grids(df, sharey=True)
    scitex.io.save(fig1, "01_parameter_scaling_on_grids_y-shared.gif")

    fig1 = plot_parameter_scaling_on_grids(df, sharey=False)
    scitex.io.save(fig1, "01_parameter_scaling_on_grids_y-not-shared.gif")

    fig2 = plot_comodulograms(df)
    scitex.io.save(fig2, "02_comodulograms.gif")

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def run_main():
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import scitex

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.gen.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    scitex.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
