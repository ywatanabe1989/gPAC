#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 11:33:54 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/_parameter_sweep_run_parameter_sweep.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/benchmark/parameter_sweep/_parameter_sweep_run_parameter_sweep.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# def main(args):
#     """Main function."""

#     print("\n" + "=" * 60)
#     print("gPAC Parameter Sweep Benchmark")
#     print("=" * 60)

#     # Run parameter sweep
#     df_sweep = run_parameter_sweep(args)
#     mngs.io.save(
#         df_sweep.to_dict(),
#         "./parameter_sweep_benchmark_out/parameter_sweep_results.yaml",
#     )
#     mngs.io.save(
#         df_sweep, "./parameter_sweep_benchmark_out/parameter_sweep_results.csv"
#     )

#     # Create visualizations
#     print("\nCreating parameter dependency plots...")
#     fig1, fig2 = plot_parameter_dependency_plots(df_sweep)
#     mngs.io.save(
#         fig1, "./parameter_sweep_benchmark_out/parameter_dependencies.gif"
#     )
#     mngs.io.save(
#         fig2, "./parameter_sweep_benchmark_out/throughput_scaling.gif"
#     )

#     # Print summary statistics
#     print("\n" + "=" * 60)
#     print("Summary Statistics")
#     print("=" * 60)

#     for param in df_sweep["param_name"].unique():
#         param_data = df_sweep[df_sweep["param_name"] == param]
#         print(f"\n{param}:")
#         print(
#             f"  gPAC time range: {param_data['gpac_time_mean'].min():.4f} - "
#             f"{param_data['gpac_time_mean'].max():.4f} seconds"
#         )
#         valid_speedups = param_data["speedup"][~param_data["speedup"].isna()]
#         if len(valid_speedups) > 0:
#             print(
#                 f"  Speedup range: {valid_speedups.min():.1f}x - {valid_speedups.max():.1f}x"
#             )
#         print(
#             f"  Max throughput: {param_data['gpac_throughput'].max():.2f} M samples/s"
#         )

#     return 0

# def run_parameter_sweep(args):
#     """Run systematic parameter sweep."""
#     results = defaultdict(list)

#     # Setup device
#     if args.device == "auto":
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     else:
#         device = args.device

#     print(f"Running parameter sweep on device: {device}")
#     print("=" * 60)

#     # Quick mode adjustments
#     if args.quick:
#         for param in VARIATIONS:
#             if param not in ["fp16", "device"]:
#                 VARIATIONS[param] = VARIATIONS[param][::2]

#     # Iterate through each parameter
#     for param_name, param_values in VARIATIONS.items():
#         print(f"\nVarying parameter: {param_name}")
#         print("-" * 40)

#         for param_value in param_values:
#             # Create test configuration
#             config = BASELINE.copy()
#             config[param_name] = param_value
#             config["device"] = device

#             # Skip invalid combinations
#             if param_name == "fp16" and param_value and device == "cpu":
#                 continue

#             print(f"  {param_name} = {param_value}...", end=" ", flush=True)

#             # Skip invalid combinations
#             if config["multi_gpu"] and not torch.cuda.is_available():
#                 continue
#             if config["multi_gpu"] and torch.cuda.device_count() < 2:
#                 continue

#             # Run benchmarks
#             times_gpac = []
#             times_tp = []

#             for _ in range(args.n_repeats):
#                 # gPAC
#                 try:
#                     time_gpac, _ = benchmark_gpac(
#                         batch_size=config["batch_size"],
#                         n_channels=config["n_channels"],
#                         seq_sec=config["seq_sec"],
#                         fs=config["fs"],
#                         pha_n_bands=config["pha_n_bands"],
#                         amp_n_bands=config["amp_n_bands"],
#                         n_perm=config["n_perm"],
#                         fp16=config["fp16"],
#                         multi_gpu=config["multi_gpu"],
#                         device=config["device"],
#                     )
#                     times_gpac.append(time_gpac)
#                 except Exception as e:
#                     print(f"gPAC error: {e}")
#                     times_gpac.append(np.nan)

#                 # TensorPAC (skip if using GPU-specific features, but allow batch_size > 1)
#                 if not config["fp16"] and not config["multi_gpu"]:
#                     try:
#                         # Generate data for TensorPAC
#                         seq_len = int(config["fs"] * config["seq_sec"])
#                         data_tp = np.random.randn(
#                             config["batch_size"], config["n_channels"], seq_len
#                         ).astype(np.float32)

#                         time_tp, _ = benchmark_tensorpac(
#                             data_tp,
#                             config["fs"],
#                             config["pha_n_bands"],
#                             config["amp_n_bands"],
#                             config["n_perm"],
#                         )
#                         times_tp.append(time_tp)
#                     except Exception as e:
#                         print(f"TensorPAC error: {e}")
#                         times_tp.append(np.nan)
#                 else:
#                     times_tp.append(np.nan)

#             # Store results
#             results["param_name"].append(param_name)
#             results["param_value"].append(param_value)
#             results["gpac_time_mean"].append(np.nanmean(times_gpac))
#             results["gpac_time_std"].append(np.nanstd(times_gpac))
#             results["tensorpac_time_mean"].append(np.nanmean(times_tp))
#             results["tensorpac_time_std"].append(np.nanstd(times_tp))
#             results["speedup"].append(
#                 np.nanmean(times_tp) / np.nanmean(times_gpac)
#             )
#             results["config"].append(config.copy())

#             # Calculate throughput (samples per second)
#             seq_len = int(config["fs"] * config["seq_sec"])
#             total_samples = (
#                 config["batch_size"] * config["n_channels"] * seq_len
#             )
#             results["gpac_throughput"].append(
#                 total_samples / np.nanmean(times_gpac) / 1e6
#             )

#             print(
#                 f"gPAC: {np.nanmean(times_gpac):.4f}s, "
#                 + f"TensorPAC: {np.nanmean(times_tp):.4f}s, "
#                 + f"Speedup: {results['speedup'][-1]:.1f}x"
#             )

#     # Convert to DataFrame for easier analysis
#     df = pd.DataFrame(results)

#     return df

# EOF
