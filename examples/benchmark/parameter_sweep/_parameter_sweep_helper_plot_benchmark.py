#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 18:43:56 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_helper_plot_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_helper_plot_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import mngs
import pandas as pd


def load_results_as_df():
    PATH = "./examples/benchmark/parameter_sweep/parameter_sweep_benchmark_out/benchmark_results_quick.pkl"
    data = mngs.io.load(PATH)

    dict_for_df = {
        k: v
        for k, v in data.items()
        if k
        not in [
            "sample_comodulogram",
            "device_ids",
            "device",
            "trainable",
            "n_cpus",
        ]
    }
    return pd.DataFrame(dict_for_df)


# In [25]: dict_for_df.keys()
# Out[25]: dict_keys(['computation_time_for_all_runs', 'package', 'n_batches', 'batch_size', 'n_channels', 'duration_sec', 'fs', 'pha_n_bands', 'amp_n_bands', 'n_perm', 'fp16', 'device', 'trainable', 'n_cpus'])

# In [26]: dict_for_df = {k: v for k, v in data.items() if k not in ["sample_comodulogram", "device_ids", "device", "tr    ...: ainable", "n_cpus", ]}

# In [27]: df = pd.DataFrame(
#     ...:     dict_for_df
#     ...: )
#     ...:

# In [28]: len(df)
# Out[28]: 441

# In [29]: df
# Out[29]:
#      computation_time_for_all_runs    package  n_batches  batch_size  ...  pha_n_bands  amp_n_bands  n_perm   fp16
# 0                         0.204946       gpac          4           4  ...          128           15       0  False
# 1                         0.005661       gpac          4           4  ...           10           15      64  False
# 2                         8.508178       gpac          4           4  ...           10           15       0   True
# 3                         0.222199       gpac          4           4  ...           10          128       0   True
# 4                         0.005229       gpac          4           4  ...           10           15       0   True
# ..                             ...        ...        ...         ...  ...          ...          ...     ...    ...
# 436                       8.634795  tensorpac          4           4  ...           10           32       0   True
# 437                       8.630671  tensorpac          4           4  ...           10           32       0  False
# 438                       0.004242       gpac          4           4  ...           10           32       0  False
# 439                       4.926409  tensorpac          4           4  ...           10           15      64  False
# 440                       0.004294       gpac          4           1  ...           10           15       0  False

# [441 rows x 11 columns]


# def plot_parameter_dependency_plots(df):
#     """Create comprehensive visualization of parameter dependencies."""
#     import mngs

#     # Get unique parameters
#     params = df["param_name"].unique()
#     n_params = len(params)

#     # Create figure with subplots
#     fig = mngs.plt.figure(figsize=(20, 12))
#     gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

#     # Color map for parameters
#     colors = plt.cm.tab10(np.linspace(0, 1, n_params))

#     for idx, param in enumerate(params):
#         row = idx // 3
#         col = idx % 3
#         ax = fig.add_subplot(gs[row, col])

#         # Filter data for this parameter
#         param_data = df[df["param_name"] == param].copy()
#         param_data = param_data.sort_values("param_value")

#         # Plot gPAC performance
#         ax.errorbar(
#             param_data["param_value"],
#             param_data["gpac_time_mean"],
#             yerr=param_data["gpac_time_std"],
#             label="gPAC",
#             marker="o",
#             color=colors[idx],
#             linewidth=2,
#             markersize=8,
#             capsize=5,
#         )

#         # Plot TensorPAC performance where available
#         valid_tp = ~param_data["tensorpac_time_mean"].isna()
#         if valid_tp.any():
#             ax.errorbar(
#                 param_data.loc[valid_tp, "param_value"],
#                 param_data.loc[valid_tp, "tensorpac_time_mean"],
#                 yerr=param_data.loc[valid_tp, "tensorpac_time_std"],
#                 label="TensorPAC",
#                 marker="s",
#                 color=colors[idx],
#                 alpha=0.5,
#                 linewidth=2,
#                 markersize=8,
#                 capsize=5,
#                 linestyle="--",
#             )

#         ax.set_xlabel(param.replace("_", " ").title())
#         ax.set_ylabel("Time (seconds)")
#         ax.set_title(f'Performance vs {param.replace("_", " ").title()}')
#         ax.grid(True, alpha=0.3)
#         ax.legend()

#         # Log scale for some parameters
#         if param in ["batch_size", "n_channels", "fs"]:
#             ax.set_xscale("log", base=2)

#         # Add speedup on secondary y-axis where applicable
#         if valid_tp.any():
#             ax2 = ax.twinx()
#             speedups = param_data.loc[valid_tp, "speedup"]
#             ax2.plot(
#                 param_data.loc[valid_tp, "param_value"],
#                 speedups,
#                 "g-",
#                 alpha=0.7,
#                 linewidth=2,
#                 label="Speedup",
#             )
#             ax2.set_ylabel("Speedup (x)", color="g")
#             ax2.tick_params(axis="y", labelcolor="g")

#     plt.suptitle("gPAC Parameter Dependencies", fontsize=16)

#     # Create throughput scaling plot
#     fig2, axes = mngs.plt.subplots(2, 2, figsize=(14, 10))
#     axes = axes.flatten()

#     # Select key parameters for throughput analysis
#     throughput_params = ["batch_size", "n_channels", "seq_sec", "pha_n_bands"]

#     for idx, param in enumerate(throughput_params):
#         ax = axes[idx]
#         param_data = df[df["param_name"] == param].copy()
#         param_data = param_data.sort_values("param_value")

#         ax.plot(
#             param_data["param_value"],
#             param_data["gpac_throughput"],
#             "o-",
#             color=colors[idx],
#             linewidth=2,
#             markersize=8,
#             label="gPAC",
#         )

#         ax.set_xlabel(param.replace("_", " ").title())
#         ax.set_ylabel("Throughput (Million samples/s)")
#         ax.set_title(f'Throughput vs {param.replace("_", " ").title()}')
#         ax.grid(True, alpha=0.3)

#         if param in ["batch_size", "n_channels"]:
#             ax.set_xscale("log", base=2)

#     plt.suptitle("gPAC Throughput Scaling", fontsize=16)
#     # plt.tight_layout()

#     return fig, fig2

# EOF
