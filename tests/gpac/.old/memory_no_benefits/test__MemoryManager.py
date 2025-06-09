#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:32:26 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/memory/test__MemoryManager.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/memory/test__MemoryManager.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from unittest.mock import patch

import pytest
import torch
from gpac.memory._MemoryManagementStrategy import MemoryManagementStrategy
from gpac.memory._MemoryManager import MemoryManager, MemoryStrategy


class TestMemoryStrategy:
    def test_memory_strategy_creation(self):
        strategy = MemoryStrategy(
            name="test",
            max_batch_size=32,
            max_permutations=100,
            chunk_size=10,
            description="test strategy",
        )
        assert strategy.name == "test"
        assert strategy.max_batch_size == 32
        assert strategy.max_permutations == 100
        assert strategy.chunk_size == 10
        assert strategy.description == "test strategy"


class TestMemoryManager:
    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_init_cpu_default(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)  # 16GB
        mock_memory.return_value.available = 8 * (1024**3)  # 8GB

        manager = MemoryManager()

        assert manager.strategy == "auto"
        assert manager.max_usage == 0.8
        assert manager.vram_gb == "auto"
        assert manager.device_info["device"] == "cpu"
        assert manager.total_memory == 16.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_init_gpu_default(self, mock_props, mock_cuda):
        mock_props.return_value.name = "RTX 4090"
        mock_props.return_value.total_memory = 24 * (1024**3)  # 24GB
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 9

        manager = MemoryManager()

        assert manager.device_info["device"] == "cuda"
        assert manager.device_info["name"] == "RTX 4090"
        assert manager.total_memory == 24.0

    def test_init_invalid_max_usage(self):
        with pytest.raises(ValueError, match="max_usage must be between"):
            MemoryManager(max_usage=1.5)

        with pytest.raises(ValueError, match="max_usage must be between"):
            MemoryManager(max_usage=0.05)

    def test_init_invalid_vram_gb(self):
        with pytest.raises(ValueError, match="vram_gb must be positive"):
            MemoryManager(vram_gb=-5)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_memory_limit_auto(self, mock_props, mock_cuda):
        mock_props.return_value.total_memory = 24 * (1024**3)
        mock_props.return_value.name = "Test GPU"
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0

        manager = MemoryManager(max_usage=0.8)
        assert manager.memory_limit_gb == 24.0 * 0.8

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_memory_limit_explicit(self, mock_props, mock_cuda):
        mock_props.return_value.total_memory = 24 * (1024**3)
        mock_props.return_value.name = "Test GPU"
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0

        manager = MemoryManager(vram_gb=16)
        assert manager.memory_limit_gb == 16.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_memory_limit_exceeds_available(
        self, mock_props, mock_cuda, capsys
    ):
        mock_props.return_value.total_memory = 16 * (1024**3)
        mock_props.return_value.name = "Test GPU"
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0

        manager = MemoryManager(vram_gb=32)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert manager.memory_limit_gb == 16.0

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_get_available_memory_cpu(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 32 * (1024**3)
        mock_memory.return_value.available = 16 * (1024**3)

        manager = MemoryManager()
        available = manager._get_available_memory()
        assert available == 16.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=4 * (1024**3))
    def test_get_available_memory_gpu(
        self, mock_allocated, mock_props, mock_cuda
    ):
        mock_props.return_value.total_memory = 24 * (1024**3)
        mock_props.return_value.name = "Test GPU"
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0

        manager = MemoryManager(max_usage=0.8)
        available = manager._get_available_memory()
        # memory_limit_gb = 24 * 0.8 = 19.2
        # allocated = 4GB
        # available = 19.2 - 4 = 15.2
        assert abs(available - 15.2) < 1e-10

    def test_define_strategies(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.total = 16 * (1024**3)
                mock_memory.return_value.available = 8 * (1024**3)

                manager = MemoryManager()
                strategies = manager._define_strategies()

                assert "vectorized" in strategies
                assert "chunked" in strategies
                assert "sequential" in strategies
                assert strategies["vectorized"].max_batch_size == 64
                assert strategies["chunked"].max_batch_size == 32
                assert strategies["sequential"].max_batch_size == 16

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_select_strategy_high_memory(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 128 * (1024**3)
        mock_memory.return_value.available = 64 * (1024**3)

        manager = MemoryManager()
        manager.memory_limit_gb = 64.0

        x = torch.randn(4, 64, 1000)
        strategy = manager.select_strategy(x, n_perm=100)

        assert isinstance(strategy, MemoryManagementStrategy)
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "vectorized"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_select_strategy_3d_input(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        x = torch.randn(2, 32, 500)
        strategy = manager.select_strategy(x, n_perm=10)

        assert isinstance(strategy, MemoryManagementStrategy)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_select_strategy_4d_input(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        x = torch.randn(2, 32, 10, 500)
        strategy = manager.select_strategy(x, n_perm=10)

        assert isinstance(strategy, MemoryManagementStrategy)

    def test_select_strategy_invalid_input(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.total = 16 * (1024**3)
                mock_memory.return_value.available = 8 * (1024**3)

                manager = MemoryManager()
                x = torch.randn(32, 500)  # 2D input

                with pytest.raises(ValueError, match="Input must be 3D or 4D"):
                    manager.select_strategy(x, n_perm=10)

    def test_get_optimal_chunk_size_small(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.total = 16 * (1024**3)
                mock_memory.return_value.available = 8 * (1024**3)

                manager = MemoryManager()
                chunk_size = manager.get_optimal_chunk_size(30)
                assert chunk_size == 30

    def test_get_optimal_chunk_size_large(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value.total = 16 * (1024**3)
                mock_memory.return_value.available = 8 * (1024**3)

                manager = MemoryManager()
                chunk_size = manager.get_optimal_chunk_size(1000)
                assert chunk_size <= 250  # 1000 // 4

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_update_available_memory(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        initial_memory = manager.available_memory

        mock_memory.return_value.available = 4 * (1024**3)
        manager.update_available_memory()

        assert manager.available_memory != initial_memory

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_estimate_strategy_memory(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        x = torch.randn(2, 64, 1000)
        strategy = MemoryManagementStrategy(
            "vectorized", "vectorized", "vectorized"
        )

        memory_gb = manager._estimate_strategy_memory(
            x, n_perm=100, strategy=strategy, fp16=True
        )

        assert memory_gb > 0
        assert isinstance(memory_gb, float)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_show_downgrade_warning(self, mock_memory, mock_cuda, capsys):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        ideal = MemoryManagementStrategy(
            "vectorized", "vectorized", "vectorized"
        )
        actual = MemoryManagementStrategy("chunked", "chunked", "sequential")

        manager._show_downgrade_warning(
            ideal, actual, ["Permutation chunking"], 50.0, (2, 64, 1000), 100
        )

        captured = capsys.readouterr()
        assert "MEMORY STRATEGY DOWNGRADE WARNING" in captured.out
        assert "RECOMMENDATIONS" in captured.out

    @patch("torch.cuda.is_available", return_value=False)
    @patch("psutil.virtual_memory")
    def test_get_memory_info(self, mock_memory, mock_cuda):
        mock_memory.return_value.total = 16 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)

        manager = MemoryManager()
        info = manager.get_memory_info()

        assert "device_info" in info
        assert "total_memory_gb" in info
        assert "memory_limit_gb" in info
        assert "available_memory_gb" in info
        assert "strategies" in info
        assert info["total_memory_gb"] == 16.0
        assert info["max_usage_fraction"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
