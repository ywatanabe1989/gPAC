#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:25:39 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/utils/test__profiler.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/utils/test__profiler.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest

import time
from unittest.mock import patch

from gpac.utils._profiler import ProfileContainer, Profiler, create_profiler


class TestProfileContainer:
    def test_profile_container_creation(self):
        container = ProfileContainer(
            name="test",
            duration=1.5,
            cpu_percent=50.0,
            ram_used_gb=2.0,
            ram_percent=25.0,
        )
        assert container.name == "test"
        assert container.duration == 1.5
        assert container.cpu_percent == 50.0
        assert container.ram_used_gb == 2.0
        assert container.ram_percent == 25.0

    def test_profile_container_str_cpu_only(self):
        container = ProfileContainer(
            name="cpu_test",
            duration=2.0,
            cpu_percent=75.5,
            ram_used_gb=4.2,
            ram_percent=52.1,
        )
        result = str(container)
        assert "📊 cpu_test" in result
        assert "⏱️  Time: 2.000s" in result
        assert "💻 CPU: 75.5%" in result
        assert "🧠 RAM: 4.20GB (52.1%)" in result

    def test_profile_container_str_with_gpu(self):
        container = ProfileContainer(
            name="gpu_test",
            duration=1.0,
            cpu_percent=50.0,
            ram_used_gb=2.0,
            ram_percent=25.0,
            gpu_used_percent=80.0,
            gpu_memory_used_gb=3.5,
            gpu_memory_percent=70.0,
            gpu_temp=65.0,
            gpu_memory_allocated_gb=3.5,
            gpu_memory_reserved_gb=4.0,
        )
        result = str(container)
        assert "🎮 GPU: 80.0%" in result
        assert "📦 VRAM (Allocated): 3.50GB (70.0%)" in result
        assert "📦 VRAM (Reserved): 4.00GB" in result
        assert "🌡️  Temp: 65°C" in result


class TestProfiler:
    def test_profiler_init_no_gpu(self):
        profiler = Profiler(enable_gpu=False)
        assert not profiler.enable_gpu
        assert profiler.results == []
        assert profiler._active_profiles == {}

    @patch("torch.cuda.is_available", return_value=True)
    def test_profiler_init_with_gpu(self, mock_cuda):
        profiler = Profiler(enable_gpu=True)
        assert profiler.enable_gpu == True

    @patch("psutil.cpu_percent", return_value=50.0)
    @patch("psutil.virtual_memory")
    def test_get_cpu_memory_stats(self, mock_memory, mock_cpu):
        mock_memory.return_value.used = 2 * (1024**3)  # 2GB
        mock_memory.return_value.percent = 25.0

        profiler = Profiler(enable_gpu=False)
        stats = profiler._get_cpu_memory_stats()

        assert stats["cpu_percent"] == 50.0
        assert stats["ram_used_gb"] == 2.0
        assert stats["ram_percent"] == 25.0

    def test_get_gpu_stats_disabled(self):
        profiler = Profiler(enable_gpu=False)
        stats = profiler._get_gpu_stats()

        expected_keys = [
            "gpu_used_percent",
            "gpu_memory_used_gb",
            "gpu_memory_percent",
            "gpu_temp",
            "gpu_memory_allocated_gb",
            "gpu_memory_reserved_gb",
        ]
        for key in expected_keys:
            assert stats[key] is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.memory_allocated", return_value=2 * (1024**3))  # 2GB
    @patch("torch.cuda.memory_reserved", return_value=3 * (1024**3))  # 3GB
    @patch("torch.cuda.get_device_properties")
    def test_get_gpu_stats_enabled(
        self, mock_props, mock_reserved, mock_allocated, mock_device, mock_cuda
    ):
        mock_props.return_value.total_memory = 8 * (1024**3)  # 8GB

        profiler = Profiler(enable_gpu=True)
        stats = profiler._get_gpu_stats()

        assert stats["gpu_memory_allocated_gb"] == 2.0
        assert stats["gpu_memory_reserved_gb"] == 3.0
        assert stats["gpu_memory_percent"] == 25.0  # 2/8 * 100

    @patch("psutil.cpu_percent", return_value=60.0)
    @patch("psutil.virtual_memory")
    def test_profile_context_manager(self, mock_memory, mock_cpu):
        mock_memory.return_value.used = 1 * (1024**3)  # 1GB
        mock_memory.return_value.percent = 12.5

        profiler = Profiler(enable_gpu=False)

        with profiler.profile("test_operation"):
            time.sleep(0.1)

        assert len(profiler.results) == 1
        result = profiler.results[0]
        assert result.name == "test_operation"
        assert result.duration >= 0.1
        assert result.cpu_percent == 60.0
        assert result.ram_used_gb == 1.0

    def test_profile_multiple_operations(self):
        profiler = Profiler(enable_gpu=False)

        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.used = 2 * (1024**3)
                mock_memory.return_value.percent = 25.0

                with profiler.profile("op1"):
                    time.sleep(0.05)

                with profiler.profile("op2"):
                    time.sleep(0.03)

        assert len(profiler.results) == 2
        assert profiler.results[0].name == "op1"
        assert profiler.results[1].name == "op2"
        assert profiler.results[0].duration >= 0.05
        assert profiler.results[1].duration >= 0.03

    def test_print_summary_no_results(self, capsys):
        profiler = Profiler(enable_gpu=False)
        profiler.print_summary()

        captured = capsys.readouterr()
        assert "No profiling results available." in captured.out

    def test_print_summary_with_results(self, capsys):
        profiler = Profiler(enable_gpu=False)
        profiler.results.append(
            ProfileContainer(
                name="test_op",
                duration=1.5,
                cpu_percent=50.0,
                ram_used_gb=2.0,
                ram_percent=25.0,
            )
        )

        profiler.print_summary()
        captured = capsys.readouterr()

        assert "📊 PROFILING SUMMARY" in captured.out
        assert "📊 test_op" in captured.out
        assert "⏱️  Total Time: 1.500s" in captured.out
        assert "🧠 Peak RAM: 2.00GB" in captured.out

    def test_get_summary_dict_empty(self):
        profiler = Profiler(enable_gpu=False)
        summary = profiler.get_summary_dict()
        assert summary == {}

    def test_get_summary_dict_with_results(self):
        profiler = Profiler(enable_gpu=False)
        profiler.results.extend(
            [
                ProfileContainer(
                    name="op1",
                    duration=1.0,
                    cpu_percent=50.0,
                    ram_used_gb=2.0,
                    ram_percent=25.0,
                ),
                ProfileContainer(
                    name="op2",
                    duration=0.5,
                    cpu_percent=75.0,
                    ram_used_gb=3.0,
                    ram_percent=35.0,
                ),
            ]
        )

        summary = profiler.get_summary_dict()

        assert summary["total_time"] == 1.5
        assert summary["sections"] == {"op1": 1.0, "op2": 0.5}
        assert summary["peak_ram_gb"] == 3.0
        assert summary["peak_cpu_percent"] == 75.0

    def test_get_summary_dict_with_gpu_results(self):
        profiler = Profiler(enable_gpu=False)
        profiler.results.append(
            ProfileContainer(
                name="gpu_op",
                duration=1.0,
                cpu_percent=50.0,
                ram_used_gb=2.0,
                ram_percent=25.0,
                gpu_memory_used_gb=4.0,
                gpu_memory_allocated_gb=4.0,
            )
        )

        summary = profiler.get_summary_dict()

        assert "peak_vram_gb" in summary
        assert "peak_vram_allocated_gb" in summary
        assert summary["peak_vram_gb"] == 4.0
        assert summary["peak_vram_allocated_gb"] == 4.0

    def test_reset(self):
        profiler = Profiler(enable_gpu=False)
        profiler.results.append(
            ProfileContainer(
                name="test",
                duration=1.0,
                cpu_percent=50.0,
                ram_used_gb=1.0,
                ram_percent=10.0,
            )
        )
        profiler._active_profiles["active"] = {"start_time": time.time()}

        profiler.reset()

        assert profiler.results == []
        assert profiler._active_profiles == {}

    def test_create_profiler(self):
        profiler = create_profiler(enable_gpu=False)
        assert isinstance(profiler, Profiler)
        assert not profiler.enable_gpu

    def test_nested_profiling_not_supported(self):
        profiler = Profiler(enable_gpu=False)

        with patch("psutil.cpu_percent", return_value=50.0):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.used = 1 * (1024**3)
                mock_memory.return_value.percent = 12.5

                with profiler.profile("outer"):
                    with profiler.profile("inner"):
                        time.sleep(0.05)

        # Should have two separate results
        assert len(profiler.results) == 2
        names = [r.name for r in profiler.results]
        assert "outer" in names
        assert "inner" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
