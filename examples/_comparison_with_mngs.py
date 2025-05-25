#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 09:29:15 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/examples/comparison_with_mngs.py
# ----------------------------------------
import os
__FILE__ = (
    "./examples/comparison_with_mngs.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import mngs

"""
Functionalities:
  - Does XYZ
  - Does XYZ
  - Does XYZ
  - Saves XYZ

Dependencies:
  - scripts:
    - /path/to/script1
    - /path/to/script2
  - packages:
    - package1
    - package2
IO:
  - input-files:
    - /path/to/input/file.xxx
    - /path/to/input/file.xxx

  - output-files:
    - /path/to/input/file.xxx
    - /path/to/input/file.xxx

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

"""Imports"""
import argparse

"""Warnings"""
# mngs.pd.ignore_SettingWithCopyWarning()
# warnings.simplefilter("ignore", UserWarning)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def main(args):
    import gpac

    xx, tt, fs = mngs.dsp.demo_sig(sig_type="pac")
    pha_n_bands = 50
    amp_n_bands = 30

    pac, pha_mids_hz, amp_mids_hz = mngs.dsp.pac(
        xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
    )

    pac_g, pha_mids_hz_g, amp_mids_hz_g = gpac.calculate_pac(
        xx[:2, :2], fs, pha_n_bands=pha_n_bands, amp_n_bands=amp_n_bands
    )

    assert pac.shape == pac_g.shape

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    import mngs

    script_mode = mngs.gen.is_script()
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(
    #     "--var",
    #     "-v",
    #     type=int,
    #     choices=None,
    #     default=1,
    #     help="(default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--flag",
    #     "-f",
    #     action="store_true",
    #     default=False,
    #     help="(default: %%(default)s)",
    # )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
