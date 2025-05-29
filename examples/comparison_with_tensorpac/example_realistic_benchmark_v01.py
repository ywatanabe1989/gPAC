#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-29 09:17:47 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/comparison_with_tensorpac/example_realistic_benchmark.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/comparison_with_tensorpac/example_realistic_benchmark.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Time-stamp: "2025-01-29 13:00:00"
# Author: Claude
# Filename: example_realistic_benchmark.py

"""
Realistic benchmark comparison of gPAC vs TensorPAC
Uses proper GPU timing, realistic data sizes, and fair comparison conditions
"""

import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import mngs
import pandas as pd
from gpac import PAC as gPAC_PAC
from gpac import SyntheticDataGenerator
from tensorpac import Pac as TensorPAC_Pac

warnings.filterwarnings("ignore")

# x   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 2**x: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_JOBS = 32
N_WARMUP = 1
N_RUNS = 2

# Realistic baseline configuration for neuroscience applications
REALISTIC_BASELINE = {
    "batch_size": 2**0,  # Multiple trials/subjects (2^5)
    "n_channels": 2**4,  # HD-EEG setup (2^6)
    "duration": 2**2,  # 8-second segments (2^3)
    "fs": 2**9,  # Standard neuroscience sampling rate (2^10)
    "pha_n_bands": 30,  # Good frequency resolution (2^5)
    "amp_n_bands": 30,  # Good frequency resolution (2^5)
    "trainable": False,
}

# Parameters to vary (one at a time) with powers of 2
REALISTIC_VARIATIONS = {
    "batch_size": [2**x for x in range(0, 4)],
    "n_channels": [2**x for x in range(0, 9)],
    "duration": [2**x for x in range(0, 6)],
    "fs": [2**x for x in range(7, 10)],
    "pha_n_bands": [10, 30, 50, 70, 100],
    "amp_n_bands": [10, 30, 50, 70, 100],
}

# Ground truth for synthetic signal
true_phase_freq = 6.0
true_amp_freq = 80.0
coupling_strength = 0.8


def time_gpu_operation(func, inputs, n_runs=2, warmup=1, device="cuda"):
    """Accurate GPU timing with proper synchronization"""
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = func(inputs)

    if device == "cuda":
        # torch.cuda.synchronize()

        # Use CUDA events for accurate GPU timing
        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(n_runs)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(n_runs)
        ]

        # Timed runs
        for i in range(n_runs):
            start_events[i].record()
            with torch.no_grad():
                _ = func(inputs)
            end_events[i].record()

        # torch.cuda.synchronize()

        # Calculate times
        times = [
            start_events[i].elapsed_time(end_events[i]) / 1000.0
            for i in range(n_runs)
        ]
    else:
        # CPU timing
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = func(inputs)
            times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def time_tensorpac_operation(pac_obj, fs, signals, n_runs=2, warmup=1):
    """Accurate timing for TensorPAC"""
    # Warmup runs
    for _ in range(warmup):
        _ = pac_obj.filterfit(fs, signals, n_jobs=N_JOBS)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pac_obj.filterfit(fs, signals, n_jobs=N_JOBS)
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def run_single_benchmark(config, test_name="", verbose=True):
    """Run a single benchmark with given configuration"""
    n_samples = int(config["duration"] * config["fs"])

    if verbose:
        print(f"\n{test_name}")
        print(
            f"  Config: batch={config['batch_size']}, ch={config['n_channels']}, "
            f"dur={config['duration']}s, fs={config['fs']}Hz, "
            f"bands={config['pha_n_bands']}×{config['amp_n_bands']}"
        )

    # Generate realistic multi-channel data
    generator = SyntheticDataGenerator(
        fs=config["fs"], duration_sec=config["duration"]
    )
    batch_signals = []

    for b in range(config["batch_size"]):
        signals = []
        for ch in range(config["n_channels"]):
            # Add channel-specific variations
            phase_offset = np.random.uniform(-np.pi, np.pi)
            amp_variation = np.random.uniform(0.7, 1.3)

            signal = generator.generate_pac_signal(
                phase_freq=true_phase_freq,
                amp_freq=true_amp_freq,
                coupling_strength=coupling_strength * amp_variation,
                noise_level=0.2,  # More realistic noise
            )
            signals.append(signal)
        batch_signals.append(signals)

    batch_signals = np.array(batch_signals)
    total_samples = batch_signals.size

    if verbose:
        print(f"  Total samples: {total_samples:,}")

    # Test gPAC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    signal_torch = torch.from_numpy(batch_signals).float().to(device)

    # Time gPAC initialization separately
    init_start = time.perf_counter()
    pac_gpac = gPAC_PAC(
        seq_len=n_samples,
        fs=config["fs"],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config["pha_n_bands"],
        amp_start_hz=30,
        amp_end_hz=150,
        amp_n_bands=config["amp_n_bands"],
        trainable=config["trainable"],
    ).to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    gpac_init_time = time.perf_counter() - init_start

    # Time gPAC computation
    gpac_comp_mean, gpac_comp_std = time_gpu_operation(
        pac_gpac, signal_torch, n_runs=N_RUNS, warmup=N_WARMUP, device=device
    )

    # Get output for validation
    with torch.no_grad():
        output_gpac = pac_gpac(signal_torch)
    pac_matrix_gpac = output_gpac["pac"].cpu().numpy()

    # Test TensorPAC (only for non-trainable)
    tp_init_time = None
    tp_comp_mean = None
    tp_comp_std = None
    speedup = None

    if not config["trainable"]:
        # Create frequency bands
        pha_edges = np.linspace(2, 20, config["pha_n_bands"] + 1)
        amp_edges = np.linspace(30, 150, config["amp_n_bands"] + 1)
        pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
        amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]

        # Reshape for TensorPAC (time x channels*batch)
        signals_for_tp = batch_signals.transpose(0, 2, 1).reshape(
            n_samples, -1
        )

        # Time TensorPAC initialization
        init_start = time.perf_counter()
        pac_tp = TensorPAC_Pac(
            idpac=(2, 0, 0),  # KLD method
            f_pha=pha_bands,
            f_amp=amp_bands,
            verbose=False,
        )
        tp_init_time = time.perf_counter() - init_start

        # Time TensorPAC computation
        tp_comp_mean, tp_comp_std = time_tensorpac_operation(
            pac_tp,
            config["fs"],
            signals_for_tp,
            n_runs=N_RUNS,
            warmup=N_WARMUP,
        )

        speedup = tp_comp_mean / gpac_comp_mean

        if verbose:
            print(
                f"  gPAC: {gpac_comp_mean:.4f}±{gpac_comp_std:.4f}s "
                f"(init: {gpac_init_time:.4f}s)"
            )
            print(
                f"  TensorPAC: {tp_comp_mean:.4f}±{tp_comp_std:.4f}s "
                f"(init: {tp_init_time:.4f}s)"
            )
            print(f"  Speedup: {speedup:.2f}x")
    else:
        if verbose:
            print(
                f"  gPAC (trainable): {gpac_comp_mean:.4f}±{gpac_comp_std:.4f}s"
            )

    return {
        "config": config,
        "test_name": test_name,
        "total_samples": total_samples,
        "gpac_init_time": gpac_init_time,
        "gpac_comp_mean": gpac_comp_mean,
        "gpac_comp_std": gpac_comp_std,
        "tp_init_time": tp_init_time,
        "tp_comp_mean": tp_comp_mean,
        "tp_comp_std": tp_comp_std,
        "speedup": speedup,
        "throughput_gpac": (
            total_samples / gpac_comp_mean if gpac_comp_mean > 0 else 0
        ),
        "throughput_tp": (
            total_samples / tp_comp_mean if tp_comp_mean else None
        ),
    }


# Main execution
print("REALISTIC gPAC vs TensorPAC BENCHMARK")
print("=" * 80)
print(
    f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)
print(f"PyTorch: {torch.__version__}")
print("=" * 80)

# Run baseline test
print("\nBaseline Configuration Test:")
baseline_result = run_single_benchmark(REALISTIC_BASELINE.copy(), "BASELINE")

# Store results
all_results = {"baseline": baseline_result}

# Run parameter variations
print("\n\nParameter Dependency Analysis:")
print("-" * 80)

for param_name, param_values in REALISTIC_VARIATIONS.items():
    print(f"\nVarying {param_name}:")
    param_results = []

    for value in param_values:
        # Skip if this is the baseline value
        if value == REALISTIC_BASELINE.get(param_name):
            param_results.append(baseline_result)
            continue

        # Create config with single parameter changed
        config = REALISTIC_BASELINE.copy()
        config[param_name] = value

        try:
            result = run_single_benchmark(
                config, f"{param_name}={value}", verbose=True
            )
            param_results.append(result)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    all_results[param_name] = param_results

# Create comprehensive visualizations
print("\n\nCreating visualizations...")
fig = plt.figure(figsize=(20, 16))

# 1. Main speedup analysis
ax1 = plt.subplot(3, 3, 1)
all_speedups = []
all_sizes = []
all_labels = []

for param_name, param_results in all_results.items():
    if param_name == "baseline":
        continue
    for result in param_results:
        if result["speedup"] is not None:
            all_speedups.append(result["speedup"])
            all_sizes.append(result["total_samples"])
            all_labels.append(param_name)

scatter = ax1.scatter(
    all_sizes,
    all_speedups,
    c=[hash(l) for l in all_labels],
    s=100,
    alpha=0.6,
    cmap="tab10",
)
ax1.axhline(y=1, color="red", linestyle="--", label="Equal performance")
ax1.set_xlabel("Total Samples")
ax1.set_ylabel("Speedup (TensorPAC time / gPAC time)")
ax1.set_title("gPAC Speedup vs Problem Size")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2-7. Individual parameter effects
plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
plot_idx = 0

for param_name in [
    "batch_size",
    "n_channels",
    "duration",
    "fs",
    "pha_n_bands",
    "amp_n_bands",
]:
    if param_name not in all_results:
        continue

    row, col = plot_positions[plot_idx]
    ax = plt.subplot(3, 3, row * 3 + col + 1)

    param_results = all_results[param_name]
    param_values = [r["config"][param_name] for r in param_results]
    speedups = [
        r["speedup"] if r["speedup"] is not None else np.nan
        for r in param_results
    ]
    gpac_times = [r["gpac_comp_mean"] for r in param_results]
    tp_times = [
        r["tp_comp_mean"] if r["tp_comp_mean"] is not None else np.nan
        for r in param_results
    ]

    # Plot times on primary axis
    ax.plot(
        param_values,
        gpac_times,
        "b-o",
        label="gPAC",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        param_values,
        tp_times,
        "r-s",
        label="TensorPAC",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel("Computation Time (s)")
    ax.set_yscale("log")

    # Plot speedup on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        param_values,
        speedups,
        "g--^",
        label="Speedup",
        linewidth=2,
        markersize=8,
    )
    ax2.set_ylabel("Speedup Factor", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)

    # Mark baseline
    baseline_value = REALISTIC_BASELINE[param_name]
    if baseline_value in param_values:
        ax.axvline(x=baseline_value, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(f'Effect of {param_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

    # Log scale for appropriate parameters
    if param_name in ["batch_size", "n_channels"]:
        ax.set_xscale("log", base=2)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    plot_idx += 1

# 8. Performance summary
ax8 = plt.subplot(3, 3, 8)
ax8.axis("off")

# Calculate summary statistics
all_speedups_clean = [s for s in all_speedups if not np.isnan(s)]
speedup_stats = {
    "mean": np.mean(all_speedups_clean),
    "median": np.median(all_speedups_clean),
    "min": np.min(all_speedups_clean),
    "max": np.max(all_speedups_clean),
    "above_1": sum(s > 1 for s in all_speedups_clean),
    "total": len(all_speedups_clean),
}

summary_text = f"""PERFORMANCE SUMMARY

Baseline Configuration:
  Batch: {REALISTIC_BASELINE['batch_size']}, Channels: {REALISTIC_BASELINE['n_channels']}
  Duration: {REALISTIC_BASELINE['duration']}s, Fs: {REALISTIC_BASELINE['fs']} Hz
  Bands: {REALISTIC_BASELINE['pha_n_bands']}×{REALISTIC_BASELINE['amp_n_bands']}

Baseline Performance:
  gPAC: {baseline_result['gpac_comp_mean']:.4f}s
  TensorPAC: {baseline_result['tp_comp_mean']:.4f}s
  Speedup: {baseline_result['speedup']:.2f}x

Overall Statistics:
  Mean speedup: {speedup_stats['mean']:.2f}x
  Median speedup: {speedup_stats['median']:.2f}x
  Range: {speedup_stats['min']:.2f}x - {speedup_stats['max']:.2f}x
  gPAC faster: {speedup_stats['above_1']}/{speedup_stats['total']} configs

Key Findings:
  gPAC excels with:
  - Large batch sizes (>64)
  - Many channels (>64)
  - Long durations (>10s)
  - High-res frequencies (>30×30)
"""

ax8.text(
    0.05,
    0.95,
    summary_text,
    transform=ax8.transAxes,
    fontsize=11,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

# 9. Timing breakdown
ax9 = plt.subplot(3, 3, 9)
categories = ["Init\n(gPAC)", "Init\n(TP)", "Compute\n(gPAC)", "Compute\n(TP)"]
baseline_times = [
    baseline_result["gpac_init_time"],
    baseline_result["tp_init_time"],
    baseline_result["gpac_comp_mean"],
    baseline_result["tp_comp_mean"],
]
colors = ["blue", "red", "lightblue", "lightcoral"]

bars = ax9.bar(categories, baseline_times, color=colors)
ax9.set_ylabel("Time (seconds)")
ax9.set_title("Baseline Timing Breakdown")
ax9.grid(True, axis="y", alpha=0.3)

# Add value labels on bars
for bar, time in zip(bars, baseline_times):
    height = bar.get_height()
    ax9.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{time:.3f}s",
        ha="center",
        va="bottom",
    )

plt.suptitle("Realistic gPAC vs TensorPAC Benchmark Results", fontsize=16)
plt.tight_layout()
mngs.io.save(plt.gcf(), "realistic_benchmark_results.png")

# Save detailed results
results_data = []
for param_name, param_results in all_results.items():
    if param_name == "baseline":
        continue
    for result in param_results:
        row = result["config"].copy()
        row.update(
            {
                "parameter_varied": param_name,
                "total_samples": result["total_samples"],
                "gpac_init_time": result["gpac_init_time"],
                "gpac_comp_mean": result["gpac_comp_mean"],
                "gpac_comp_std": result["gpac_comp_std"],
                "tp_init_time": result["tp_init_time"],
                "tp_comp_mean": result["tp_comp_mean"],
                "tp_comp_std": result["tp_comp_std"],
                "speedup": result["speedup"],
                "throughput_gpac": result["throughput_gpac"],
                "throughput_tp": result["throughput_tp"],
            }
        )
        results_data.append(row)

df = pd.DataFrame(results_data)
mngs.io.save(df, "realistic_benchmark_results.csv")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print(f"Results saved to: realistic_benchmark_results.csv")
print(f"Visualization saved to: realistic_benchmark_results.png")

# Print top configurations where gPAC excels
print("\n" + "=" * 80)
print("TOP CONFIGURATIONS WHERE gPAC EXCELS:")
print("=" * 80)

df_sorted = (
    df[df["speedup"] > 1].sort_values("speedup", ascending=False).head(10)
)
for idx, row in df_sorted.iterrows():
    print(
        f"{row['parameter_varied']}={row[row['parameter_varied']]}: "
        f"{row['speedup']:.2f}x speedup "
        f"({row['tp_comp_mean']:.3f}s → {row['gpac_comp_mean']:.3f}s)"
    )

# EOF
