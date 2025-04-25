#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-10 09:34:15 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/PackageHandlers/_BaseHandler.py
# ----------------------------------------
import os
__FILE__ = (
    "/ssh:sp:/home/ywatanabe/proj/gPAC/scripts/PackageHandlers/_BaseHandler.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------



"""
This script defines BaseHandler for phase-amplitude coupling calculations.
"""

# Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Type, Union

import mngs
import numpy as np
import pandas as pd


# Functions & Classes
@dataclass
class BaseHandler(ABC):
    # Signal properties
    seq_len: int
    fs: float

    # Model
    device: [str]
    model: [Type]

    # Phase
    pha_min_hz: Union[int, float]
    pha_max_hz: Union[int, float]
    pha_n_bands: Union[int, float]

    # Amplitude
    amp_min_hz: Union[int, float]
    amp_max_hz: Union[int, float]
    amp_n_bands: Union[int, float]

    # Surrogate
    n_perm: int

    # Calculation options
    chunk_size: int
    fp16: bool
    in_place: bool
    trainable: bool
    use_threads: bool

    # Delete me
    dim_handler = mngs.gen.DimHandler()
    ts = None

    # Optional
    init_start_str = "Model Initialization Starts"
    init_end_str = "Model Initialization Ends"
    calc_start_str = "PAC Calculation Starts"
    calc_end_str = "PAC Calculation Ends"

    @abstractmethod
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_n_bands: Union[int, float],
        pha_min_hz: Union[int, float],
        pha_max_hz: Union[int, float],
        amp_n_bands: Union[int, float],
        amp_min_hz: Union[int, float],
        amp_max_hz: Union[int, float],
        n_perm: int,
        chunk_size: int,
        fp16: bool,
        in_place: bool,
        trainable: bool,
        device: str,
        use_threads: bool,
        ts: mngs.gen.TimeStamper,
    ):

        # Signal properties
        self.seq_len = seq_len
        self.fs = fs

        # Phase
        self.pha_n_bands = pha_n_bands
        self.pha_min_hz = pha_min_hz
        self.pha_max_hz = pha_max_hz

        # Amplitude
        self.amp_n_bands = amp_n_bands
        self.amp_min_hz = amp_min_hz
        self.amp_max_hz = amp_max_hz

        # Surrogate
        self.n_perm = n_perm

        # Calculation options
        self.chunk_size = chunk_size
        self.fp16 = fp16
        self.in_place = in_place
        self.trainable = trainable
        self.device = device
        self.use_threads = use_threads

        # Time Stamper
        self.ts = ts

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    @abstractmethod
    def __str__(
        self,
    ):
        pass

    @abstractmethod
    def calc_pac(self, xx):
        pass

    @property
    @abstractmethod
    def freqs_amp(self):
        pass

    @property
    @abstractmethod
    def freqs_pha(self):
        pass

    @property
    def stats(self):
        indi_init_start = np.where(self.ts.record["comment"] == self.init_start_str)[0]
        indi_init_end = np.where(self.ts.record["comment"] == self.init_end_str)[0]
        init_start_times = np.array(self.ts.record.loc[indi_init_start].timestamp)
        init_end_times = np.array(self.ts.record.loc[indi_init_end].timestamp)
        init_delta_times = init_end_times - init_start_times
        init_delta_times_mm = init_delta_times.mean()
        init_delta_times_ss = init_delta_times.std()
        init_delta_times_nn = len(init_delta_times)

        indi_calc_start = np.where(self.ts.record["comment"] == self.calc_start_str)[0]
        indi_calc_end = np.where(self.ts.record["comment"] == self.calc_end_str)[0]
        calc_start_times = np.array(self.ts.record.loc[indi_calc_start].timestamp)
        calc_end_times = np.array(self.ts.record.loc[indi_calc_end].timestamp)
        calc_delta_times = calc_end_times - calc_start_times
        calc_delta_times_mm = calc_delta_times.mean()
        calc_delta_times_ss = calc_delta_times.std()
        calc_delta_times_nn = len(calc_delta_times)

        dic = {
            "time": _get_middle_time(calc_start_times[0], calc_end_times[-1]),
            "init_time_mean_sec": init_delta_times_mm,
            "init_time_std_sec": init_delta_times_ss,
            "init_time_nn": init_delta_times_nn,
            "calc_time_mean_sec": calc_delta_times_mm,
            "calc_time_std_sec": calc_delta_times_ss,
            "calc_time_nn": calc_delta_times_nn,
        }

        df = pd.DataFrame(data=dic, index=[str(self)]).round(3)
        return df

def _get_middle_time(dt1, dt2):
    """
    Returns the middle time between two datetime objects.

    Parameters:
    - dt1, dt2: datetime.datetime objects.

    Returns:
    - datetime.datetime object representing the middle time between dt1 and dt2.
    """
    # Ensure dt1 is the earlier and dt2 is the later datetime
    if dt1 > dt2:
        dt1, dt2 = dt2, dt1

    # Calculate the difference and divide by 2 to find the middle timedelta
    half_diff = (dt2 - dt1) / 2

    # Add the half difference to the first datetime to get the middle time
    middle_time = dt1 + half_diff

    middle_time = datetime.fromtimestamp(middle_time)

    return middle_time

# EOF