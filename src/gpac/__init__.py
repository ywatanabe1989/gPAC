#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 17:35:28 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._PAC import PAC
from ._pac import calculate_pac
from ._SyntheticDataGenerator import SyntheticDataGenerator

__version__ = "0.1.0"

# EOF
