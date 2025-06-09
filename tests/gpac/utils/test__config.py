#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:25:28 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/utils/test__config.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/utils/test__config.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
from gpac.utils._config import GPACConfig, config


class TestGPACConfig:
    def test_default_values(self):
        cfg = GPACConfig()
        assert cfg.default_fp16 is True
        assert cfg.default_compile_mode is True
        assert cfg.max_memory_usage_fraction == 0.95
        assert cfg.cache_filter_kernels is True
        assert cfg.enable_profiling is False

    def test_custom_values(self):
        cfg = GPACConfig(
            default_fp16=False,
            default_compile_mode=False,
            max_memory_usage_fraction=0.8,
            cache_filter_kernels=False,
            enable_profiling=True,
        )
        assert cfg.default_fp16 is False
        assert cfg.default_compile_mode is False
        assert cfg.max_memory_usage_fraction == 0.8
        assert cfg.cache_filter_kernels is False
        assert cfg.enable_profiling is True

    def test_validate_valid_memory_fraction(self):
        cfg = GPACConfig(max_memory_usage_fraction=0.5)
        cfg.validate()  # Should not raise

        cfg = GPACConfig(max_memory_usage_fraction=0.1)
        cfg.validate()  # Should not raise

        cfg = GPACConfig(max_memory_usage_fraction=1.0)
        cfg.validate()  # Should not raise

    def test_validate_invalid_memory_fraction_too_low(self):
        cfg = GPACConfig(max_memory_usage_fraction=0.05)
        with pytest.raises(
            ValueError, match="max_memory_usage_fraction must be in"
        ):
            cfg.validate()

    def test_validate_invalid_memory_fraction_too_high(self):
        cfg = GPACConfig(max_memory_usage_fraction=1.5)
        with pytest.raises(
            ValueError, match="max_memory_usage_fraction must be in"
        ):
            cfg.validate()

    def test_validate_invalid_memory_fraction_zero(self):
        cfg = GPACConfig(max_memory_usage_fraction=0.0)
        with pytest.raises(
            ValueError, match="max_memory_usage_fraction must be in"
        ):
            cfg.validate()

    def test_global_config_instance(self):
        assert isinstance(config, GPACConfig)
        assert config.default_fp16 is True
        assert config.max_memory_usage_fraction == 0.95

    def test_config_modification(self):
        original_value = config.default_fp16
        config.default_fp16 = not original_value
        assert config.default_fp16 == (not original_value)
        # Restore original
        config.default_fp16 = original_value

    def test_config_validation_through_global(self):
        original_fraction = config.max_memory_usage_fraction
        config.max_memory_usage_fraction = 0.7
        config.validate()  # Should not raise

        config.max_memory_usage_fraction = 2.0
        with pytest.raises(ValueError):
            config.validate()

        # Restore original
        config.max_memory_usage_fraction = original_fraction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
