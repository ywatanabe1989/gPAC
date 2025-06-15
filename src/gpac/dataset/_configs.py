#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-15 17:59:44 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/dataset/_configs.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/dataset/_configs.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

single_class_single_pac_config = {
    "single_pac": {
        "components": [
            {
                "pha_hz": 8.0,
                "amp_hz": 80.0,
                "strength": 0.5,
            }
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    }
}

single_class_multi_pac_config = {
    "multi_pac": {
        "components": [
            {
                "pha_hz": 8.0,
                "amp_hz": 80.0,
                "strength": 0.4,
            },
            {
                "pha_hz": 12.0,
                "amp_hz": 120.0,
                "strength": 0.3,
            },
            {
                "pha_hz": 6.0,
                "amp_hz": 60.0,
                "strength": 0.5,
            },
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    }
}

multi_class_single_pac_config = {
    "no_pac": {
        "components": [],
        "noise_levels": [0.1, 0.2, 0.3],
    },
    "theta_gamma": {
        "components": [
            {
                "pha_hz": 8.0,
                "amp_hz": 80.0,
                "strength": 0.5,
            }
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    },
    "alpha_beta": {
        "components": [
            {
                "pha_hz": 10.0,
                "amp_hz": 20.0,
                "strength": 0.4,
            }
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    },
}

multi_class_multi_pac_config = {
    "no_pac": {
        "components": [],
        "noise_levels": [0.1, 0.2, 0.3],
    },
    "single_pac": {
        "components": [
            {
                "pha_hz": 8.0,
                "amp_hz": 80.0,
                "strength": 0.5,
            }
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    },
    "dual_pac": {
        "components": [
            {
                "pha_hz": 8.0,
                "amp_hz": 80.0,
                "strength": 0.4,
            },
            {
                "pha_hz": 12.0,
                "amp_hz": 120.0,
                "strength": 0.3,
            },
        ],
        "noise_levels": [0.1, 0.2, 0.3],
    },
}

# EOF
