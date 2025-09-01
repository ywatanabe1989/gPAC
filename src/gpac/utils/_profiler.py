#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:08:29 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/utils/_profiler.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/utils/_profiler.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psutil
import torch

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class ProfileContainer:
    """Container for profiling results."""

    name: str
    duration: float
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_used_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temp: Optional[float] = None
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None

    def __str__(self):
        """Format profile result as a readable string."""
        lines = [
            f"📊 {self.name}",
            f"⏱️  Time: {self.duration:.3f}s",
            f"💻 CPU: {self.cpu_percent:.1f}%",
            f"🧠 RAM: {self.ram_used_gb:.2f}GB ({self.ram_percent:.1f}%)",
        ]

        if self.gpu_memory_used_gb is not None:
            vram_lines = [
                f"📦 VRAM (Allocated): {self.gpu_memory_allocated_gb:.2f}GB ({self.gpu_memory_percent:.1f}%)"
            ]
            if self.gpu_memory_reserved_gb is not None:
                vram_lines.append(
                    f"📦 VRAM (Reserved): {self.gpu_memory_reserved_gb:.2f}GB"
                )
            if self.gpu_used_percent is not None:
                vram_lines.insert(0, f"🎮 GPU: {self.gpu_used_percent:.1f}%")
            if self.gpu_temp:
                vram_lines.append(f"🌡️  Temp: {self.gpu_temp:.0f}°C")
            lines.extend(vram_lines)

        return "\n".join(filter(None, lines))


class Profiler:
    """Comprehensive profiler for tracking performance metrics."""

    def __init__(self, enable_gpu: bool = True):
        """Initialize profiler.

        Args:
            enable_gpu: Whether to track GPU metrics
        """
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.results: List[ProfileContainer] = []
        self._active_profiles: Dict[str, Dict[str, Any]] = {}

        # Check GPU availability
        self.gpus = []
        if self.enable_gpu and GPUTIL_AVAILABLE:
            try:
                self.gpus = GPUtil.getGPUs()
            except:
                self.enable_gpu = False

    def _get_cpu_memory_stats(self) -> Dict[str, float]:
        """Get current CPU and memory statistics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_percent = memory.percent

        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": ram_used_gb,
            "ram_percent": ram_percent,
        }

    def _get_gpu_stats(self) -> Dict[str, Optional[float]]:
        """Get current GPU statistics using PyTorch native APIs."""
        if not self.enable_gpu:
            return {
                "gpu_used_percent": None,
                "gpu_memory_used_gb": None,
                "gpu_memory_percent": None,
                "gpu_temp": None,
                "gpu_memory_allocated_gb": None,
                "gpu_memory_reserved_gb": None,
            }

        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)

            total_memory = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )
            allocated_percent = (allocated / total_memory) * 100

            # Try to get GPU utilization and temperature from GPUtil
            gpu_load = None
            gpu_temp = None
            if self.gpus:
                try:
                    gpu = GPUtil.getGPUs()[0]
                    gpu_load = gpu.load * 100
                    gpu_temp = gpu.temperature
                except:
                    pass

            return {
                "gpu_used_percent": gpu_load,
                "gpu_memory_used_gb": allocated,
                "gpu_memory_percent": allocated_percent,
                "gpu_temp": gpu_temp,
                "gpu_memory_allocated_gb": allocated,
                "gpu_memory_reserved_gb": reserved,
            }
        except Exception as e:
            return {
                "gpu_used_percent": None,
                "gpu_memory_used_gb": None,
                "gpu_memory_percent": None,
                "gpu_temp": None,
                "gpu_memory_allocated_gb": None,
                "gpu_memory_reserved_gb": None,
            }

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        start_time = time.time()

        self._active_profiles[name] = {
            "start_time": start_time,
            "initial_stats": {
                **self._get_cpu_memory_stats(),
                **self._get_gpu_stats(),
            },
        }

        if self.enable_gpu:
            torch.cuda.synchronize()

        try:
            yield
        finally:
            if self.enable_gpu:
                torch.cuda.synchronize()

            end_time = time.time()
            duration = end_time - start_time

            final_stats = {
                **self._get_cpu_memory_stats(),
                **self._get_gpu_stats(),
            }

            result = ProfileContainer(
                name=name,
                duration=duration,
                cpu_percent=final_stats["cpu_percent"],
                ram_used_gb=final_stats["ram_used_gb"],
                ram_percent=final_stats["ram_percent"],
                gpu_used_percent=final_stats["gpu_used_percent"],
                gpu_memory_used_gb=final_stats["gpu_memory_used_gb"],
                gpu_memory_percent=final_stats["gpu_memory_percent"],
                gpu_temp=final_stats["gpu_temp"],
                gpu_memory_allocated_gb=final_stats.get("gpu_memory_allocated_gb"),
                gpu_memory_reserved_gb=final_stats.get("gpu_memory_reserved_gb"),
            )

            self.results.append(result)
            del self._active_profiles[name]

    def print_summary(self):
        """Print a summary of all profiling results."""
        if not self.results:
            print("No profiling results available.")
            return

        print("\n" + "=" * 60)
        print("📊 PROFILING SUMMARY")
        print("=" * 60)

        for result in self.results:
            print(f"\n{result}")
            print("-" * 40)

        total_time = sum(r.duration for r in self.results)
        print(f"\n⏱️  Total Time: {total_time:.3f}s")

        if self.results:
            peak_ram = max(r.ram_used_gb for r in self.results)
            print(f"🧠 Peak RAM: {peak_ram:.2f}GB")

            if any(r.gpu_memory_used_gb is not None for r in self.results):
                peak_vram = max(
                    r.gpu_memory_used_gb
                    for r in self.results
                    if r.gpu_memory_used_gb is not None
                )
                print(f"📦 Peak VRAM (Allocated): {peak_vram:.2f}GB")

        print("=" * 60)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as a dictionary."""
        if not self.results:
            return {}

        summary = {
            "total_time": sum(r.duration for r in self.results),
            "sections": {r.name: r.duration for r in self.results},
            "peak_ram_gb": max(r.ram_used_gb for r in self.results),
            "peak_cpu_percent": max(r.cpu_percent for r in self.results),
        }

        if any(r.gpu_memory_used_gb is not None for r in self.results):
            gpu_results = [r for r in self.results if r.gpu_memory_used_gb is not None]
            summary.update(
                {
                    "peak_vram_gb": max(r.gpu_memory_used_gb for r in gpu_results),
                    "peak_vram_allocated_gb": max(
                        r.gpu_memory_allocated_gb
                        for r in gpu_results
                        if r.gpu_memory_allocated_gb is not None
                    ),
                }
            )

        return summary

    def reset(self):
        """Reset all profiling results."""
        self.results.clear()
        self._active_profiles.clear()


def create_profiler(enable_gpu: bool = True) -> Profiler:
    """Create a new profiler instance."""
    return Profiler(enable_gpu=enable_gpu)


# EOF
