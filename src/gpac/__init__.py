#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-28 17:04:29 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/___init__.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/___init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._BandPassFilter import BandPassFilter
from ._Hilbert import Hilbert
from ._ModulationIndex import ModulationIndex
from ._PAC import PAC
from ._SyntheticDataGenerator import SyntheticDataGenerator, generate_pac_signal

# Hidden imports - not exposed in __all__
from ._Profiler import Profiler as _Profiler
from ._Profiler import ProfileResult as _ProfileResult
from ._Profiler import create_profiler as _create_profiler

__all__ = [
    "PAC",
    "BandPassFilter",
    "Hilbert",
    "ModulationIndex",
    "SyntheticDataGenerator",
    "generate_pac_signal",
]

# EOF
