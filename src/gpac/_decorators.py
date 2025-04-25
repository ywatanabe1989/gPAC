#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-25 17:31:42 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_decorators.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_decorators.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

import numpy as np
import torch


def torch_fn(func):
    def wrapper(*args, **kwargs):
        target_device_kwarg = kwargs.get("device", None)
        device = None
        tensor_devices = [
            arg.device for arg in args if isinstance(arg, torch.Tensor)
        ]
        tensor_devices.extend(
            [v.device for v in kwargs.values() if isinstance(v, torch.Tensor)]
        )
        if target_device_kwarg:
            device = target_device_kwarg
        elif tensor_devices:
            if len(set(tensor_devices)) > 1:
                warnings.warn(
                    f"Function '{func.__name__}' received tensors on multiple devices ({set(tensor_devices)}). "
                    f"Using the first detected device ({tensor_devices[0]}). "
                    "Explicitly provide the 'device' kwarg for clarity."
                )
            device = tensor_devices[0]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        first_arg_was_numpy = (
            isinstance(args[0], np.ndarray) if args else False
        )
        processed_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                try:
                    dtype = (
                        np.float32 if arg.dtype == np.float64 else arg.dtype
                    )
                    tensor_arg = torch.from_numpy(arg.astype(dtype))
                    processed_args.append(tensor_arg.to(device))
                except TypeError as error:
                    warnings.warn(
                        f"Warning: Could not convert numpy array arg in '{func.__name__}' to tensor: {error}. Passing as is."
                    )
                    processed_args.append(arg)
            elif isinstance(arg, torch.Tensor):
                processed_args.append(arg.to(device))
            else:
                processed_args.append(arg)
        processed_kwargs = {}
        for key, value in kwargs.items():
            if key == "device":
                continue
            if isinstance(value, np.ndarray):
                try:
                    dtype = (
                        np.float32
                        if value.dtype == np.float64
                        else value.dtype
                    )
                    tensor_value = torch.from_numpy(value.astype(dtype))
                    processed_kwargs[key] = tensor_value.to(device)
                except TypeError as error:
                    warnings.warn(
                        f"Warning: Could not convert numpy array kwarg '{key}' in '{func.__name__}' to tensor: {error}. Passing as is."
                    )
                    processed_kwargs[key] = value
            elif isinstance(value, torch.Tensor):
                processed_kwargs[key] = value.to(device)
            else:
                processed_kwargs[key] = value
        result = func(*processed_args, **processed_kwargs)

        def convert_to_numpy_if_needed(item):
            if isinstance(item, torch.Tensor) and first_arg_was_numpy:
                return item.cpu().numpy()
            return item

        if isinstance(result, tuple):
            return tuple(convert_to_numpy_if_needed(item) for item in result)
        elif isinstance(result, list):
            return [convert_to_numpy_if_needed(item) for item in result]
        elif isinstance(result, dict):
            return {
                k: convert_to_numpy_if_needed(v) for k, v in result.items()
            }
        else:
            return convert_to_numpy_if_needed(result)

    return wrapper

# EOF