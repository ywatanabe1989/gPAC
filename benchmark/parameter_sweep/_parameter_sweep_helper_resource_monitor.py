#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 14:38:17 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/parameter_sweep/_parameter_sweep_helper_resource_monitor.py
# ----------------------------------------
import os
__FILE__ = (
    "./benchmark/parameter_sweep/_parameter_sweep_helper_resource_monitor.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import psutil
import pynvml


def get_system_resources():
    """Get current system resource usage."""
    resources = {}

    # CPU usage
    resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    resources["cpu_count"] = psutil.cpu_count()

    # RAM usage
    memory = psutil.virtual_memory()
    resources["ram_total_gb"] = memory.total / (1024**3)
    resources["ram_used_gb"] = memory.used / (1024**3)
    resources["ram_percent"] = memory.percent

    return resources


def get_gpu_resources():
    """Get GPU resource usage."""
    gpu_resources = {}

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for gpu_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # GPU memory
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_resources[f"gpu_{gpu_id}_memory_total_mb"] = (
                memory_info.total / (1024**2)
            )
            gpu_resources[f"gpu_{gpu_id}_memory_used_mb"] = (
                memory_info.used / (1024**2)
            )
            gpu_resources[f"gpu_{gpu_id}_memory_percent"] = (
                memory_info.used / memory_info.total
            ) * 100

            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_resources[f"gpu_{gpu_id}_util_percent"] = utilization.gpu

    except Exception as e:
        print(f"GPU monitoring error: {e}")

    return gpu_resources


def monitor_resources_during_computation(func, *args, **kwargs):
    """Monitor resources during computation."""
    # Pre-computation resources
    pre_system = get_system_resources()
    pre_gpu = get_gpu_resources()

    # Execute computation
    result = func(*args, **kwargs)

    # Post-computation resources
    post_system = get_system_resources()
    post_gpu = get_gpu_resources()

    # Calculate resource usage
    resource_usage = {
        "cpu_percent_avg": (
            pre_system["cpu_percent"] + post_system["cpu_percent"]
        )
        / 2,
        "ram_used_gb_max": max(
            pre_system["ram_used_gb"], post_system["ram_used_gb"]
        ),
        "ram_percent_max": max(
            pre_system["ram_percent"], post_system["ram_percent"]
        ),
    }

    # Add GPU metrics
    for key in pre_gpu:
        if "memory_used" in key or "util_percent" in key:
            resource_usage[f"{key}_max"] = max(
                pre_gpu.get(key, 0), post_gpu.get(key, 0)
            )

    return result, resource_usage

# EOF
