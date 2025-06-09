#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:29:57 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/memory/test__MemoryManagementStrategy.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/memory/test__MemoryManagementStrategy.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
from gpac.memory._MemoryManagementStrategy import (PRESETS,
                                                   MemoryManagementStrategy,
                                                   create_strategy_for_memory)


class TestMemoryManagementStrategy:
    def test_default_initialization(self):
        strategy = MemoryManagementStrategy()
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "auto"
        assert strategy.batch_chunk_size is None
        assert strategy.channel_chunk_size is None
        assert strategy.permutation_chunk_size is None

    def test_custom_initialization(self):
        strategy = MemoryManagementStrategy(
            batch="chunked",
            channel="sequential",
            permutation="vectorized",
            batch_chunk_size=16,
            channel_chunk_size=32,
        )
        assert strategy.batch == "chunked"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "vectorized"
        assert strategy.batch_chunk_size == 16
        assert strategy.channel_chunk_size == 32

    def test_invalid_batch_mode(self):
        with pytest.raises(ValueError, match="Invalid batch mode"):
            MemoryManagementStrategy(batch="invalid")

    def test_invalid_channel_mode(self):
        with pytest.raises(ValueError, match="Invalid channel mode"):
            MemoryManagementStrategy(channel="auto")

    def test_invalid_permutation_mode(self):
        with pytest.raises(ValueError, match="Invalid permutation mode"):
            MemoryManagementStrategy(permutation="invalid")

    def test_invalid_chunk_sizes(self):
        with pytest.raises(
            ValueError, match="batch_chunk_size must be positive"
        ):
            MemoryManagementStrategy(batch="chunked", batch_chunk_size=-1)

        with pytest.raises(
            ValueError, match="channel_chunk_size must be positive"
        ):
            MemoryManagementStrategy(channel="chunked", channel_chunk_size=0)

        with pytest.raises(
            ValueError, match="permutation_chunk_size must be positive"
        ):
            MemoryManagementStrategy(
                permutation="chunked", permutation_chunk_size=-5
            )

    def test_get_effective_sizes_vectorized(self):
        strategy = MemoryManagementStrategy(
            batch="vectorized", channel="vectorized", permutation="vectorized"
        )
        sizes = strategy.get_effective_sizes(
            batch_size=32, n_channels=64, n_perm=100
        )

        assert sizes["batch"] == 32
        assert sizes["channel"] == 64
        assert sizes["permutation"] == 100
        assert sizes["batch_total"] == 32
        assert sizes["channel_total"] == 64
        assert sizes["permutation_total"] == 100

    def test_get_effective_sizes_sequential(self):
        strategy = MemoryManagementStrategy(
            batch="sequential", channel="sequential", permutation="sequential"
        )
        sizes = strategy.get_effective_sizes(
            batch_size=32, n_channels=64, n_perm=100
        )

        assert sizes["batch"] == 1
        assert sizes["channel"] == 1
        assert sizes["permutation"] == 1

    def test_get_effective_sizes_chunked(self):
        strategy = MemoryManagementStrategy(
            batch="chunked",
            channel="chunked",
            permutation="chunked",
            batch_chunk_size=8,
            channel_chunk_size=16,
            permutation_chunk_size=25,
        )
        sizes = strategy.get_effective_sizes(
            batch_size=32, n_channels=64, n_perm=100
        )

        assert sizes["batch"] == 8
        assert sizes["channel"] == 16
        assert sizes["permutation"] == 25

    def test_get_effective_sizes_chunked_auto(self):
        strategy = MemoryManagementStrategy(
            batch="chunked", channel="chunked", permutation="chunked"
        )
        sizes = strategy.get_effective_sizes(
            batch_size=16, n_channels=8, n_perm=30
        )

        # Should use min of chunk_size and actual size
        assert sizes["batch"] == 16  # min(32, 16)
        assert sizes["channel"] == 8  # min(32, 8)
        assert sizes["permutation"] == 30  # min(50, 30)

    def test_get_effective_sizes_no_permutations(self):
        strategy = MemoryManagementStrategy()
        sizes = strategy.get_effective_sizes(
            batch_size=32, n_channels=64, n_perm=None
        )

        assert sizes["permutation"] == 0
        assert sizes["permutation_total"] == 0

    def test_str_representation_vectorized(self):
        strategy = MemoryManagementStrategy()
        result = str(strategy)
        assert "Batch: vectorized" in result
        assert "Channel: vectorized" in result
        assert "Permutation: auto" in result

    def test_str_representation_chunked(self):
        strategy = MemoryManagementStrategy(
            batch="chunked",
            channel="chunked",
            permutation="chunked",
            batch_chunk_size=16,
            channel_chunk_size=32,
            permutation_chunk_size=50,
        )
        result = str(strategy)
        assert "Batch: chunked(16)" in result
        assert "Channel: chunked(32)" in result
        assert "Permutation: chunked(50)" in result

    def test_str_representation_chunked_auto(self):
        strategy = MemoryManagementStrategy(
            batch="chunked", channel="sequential", permutation="chunked"
        )
        result = str(strategy)
        assert "Batch: chunked(auto)" in result
        assert "Channel: sequential" in result
        assert "Permutation: chunked(auto)" in result


class TestPresets:
    def test_conservative_preset(self):
        strategy = PRESETS["conservative"]
        assert strategy.batch == "sequential"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "sequential"

    def test_balanced_preset(self):
        strategy = PRESETS["balanced"]
        assert strategy.batch == "chunked"
        assert strategy.channel == "chunked"
        assert strategy.permutation == "chunked"
        assert strategy.batch_chunk_size == 16
        assert strategy.channel_chunk_size == 32
        assert strategy.permutation_chunk_size == 50

    def test_aggressive_preset(self):
        strategy = PRESETS["aggressive"]
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "vectorized"

    def test_ultra_aggressive_preset(self):
        strategy = PRESETS["ultra_aggressive"]
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "vectorized"

    def test_batch_optimized_preset(self):
        strategy = PRESETS["batch_optimized"]
        assert strategy.batch == "vectorized"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "sequential"

    def test_channel_optimized_preset(self):
        strategy = PRESETS["channel_optimized"]
        assert strategy.batch == "sequential"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "sequential"

    def test_permutation_optimized_preset(self):
        strategy = PRESETS["permutation_optimized"]
        assert strategy.batch == "sequential"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "vectorized"


class TestCreateStrategyForMemory:
    def test_ultra_high_memory(self):
        strategy = create_strategy_for_memory(80.0)
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "vectorized"

    def test_high_memory(self):
        strategy = create_strategy_for_memory(50.0)
        assert strategy.batch == "vectorized"
        assert strategy.channel == "vectorized"
        assert strategy.permutation == "vectorized"

    def test_mid_high_memory_no_perm(self):
        strategy = create_strategy_for_memory(30.0, n_perm=None)
        assert strategy.batch == "vectorized"
        assert strategy.channel == "chunked"
        assert strategy.channel_chunk_size == 64
        assert strategy.permutation == "vectorized"

    def test_mid_high_memory_many_perm(self):
        strategy = create_strategy_for_memory(30.0, n_perm=500)
        assert strategy.batch == "vectorized"
        assert strategy.channel == "chunked"
        assert strategy.permutation == "chunked"
        assert strategy.permutation_chunk_size == 100

    def test_mid_memory(self):
        strategy = create_strategy_for_memory(20.0)
        assert strategy.batch == "chunked"
        assert strategy.channel == "chunked"
        assert strategy.permutation == "chunked"
        assert strategy.batch_chunk_size == 32
        assert strategy.channel_chunk_size == 32
        assert strategy.permutation_chunk_size == 50

    def test_consumer_memory(self):
        strategy = create_strategy_for_memory(12.0)
        assert strategy.batch == "chunked"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "sequential"
        assert strategy.batch_chunk_size == 8

    def test_low_memory(self):
        strategy = create_strategy_for_memory(4.0)
        assert strategy.batch == "sequential"
        assert strategy.channel == "sequential"
        assert strategy.permutation == "sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
