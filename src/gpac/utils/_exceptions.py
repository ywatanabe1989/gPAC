#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:08:28 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/utils/_exceptions.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/utils/_exceptions.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


class GPACError(Exception):
    """Base Exception for gPAC."""

    pass


class MemoryError(GPACError):
    """Memory-related errors."""

    pass


class ConfigurationError(GPACError):
    """Configuration-related errors."""

    pass


class DeviceError(GPACError):
    """Device/CUDA-related errors."""

    pass


# EOF
