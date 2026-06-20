#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 21:55:30 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/__init__.py

from ._PAC import PAC, set_deterministic
from . import core, utils, dataset

__version__ = "0.3.4"
__all__ = ["PAC", "set_deterministic", "core", "utils", "dataset"]

# EOF
