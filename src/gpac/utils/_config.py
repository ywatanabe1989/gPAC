#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:08:26 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/utils/_config.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/utils/_config.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from dataclasses import dataclass


@dataclass
class GPACConfig:
    """Global configuration for gPAC."""

    default_fp16: bool = True
    default_compile_mode: bool = True
    max_memory_usage_fraction: float = 0.95
    cache_filter_kernels: bool = True
    enable_profiling: bool = False

    def validate(self):
        if not 0.1 <= self.max_memory_usage_fraction <= 1.0:
            raise ValueError("max_memory_usage_fraction must be in [0.1, 1.0]")


# Global config instance
config = GPACConfig()

# EOF
