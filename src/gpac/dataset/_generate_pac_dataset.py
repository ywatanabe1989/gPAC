#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-15 17:55:04 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/dataset/_generate_pac_dataset.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/dataset/_generate_pac_dataset.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import torch

"""
Synthetic PAC dataset generation with predefined class configurations.

This module provides production-ready configurations for generating synthetic
Phase-Amplitude Coupling datasets across different experimental scenarios.

Available default configurations:
- single_class_single_pac_config: One class with single PAC component
- single_class_multi_pac_config: One class with multiple PAC components
- multi_class_single_pac_config: Multiple classes with single PAC each
- multi_class_multi_pac_config: Multiple classes with varying PAC complexity
"""

from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from ._configs import single_class_single_pac_config
from ._SyntheticDataGenerator import SyntheticDataGenerator
from ._SyntheticPACDataset import SyntheticPACDataset


def generate_pac_dataset(
    n_samples: int = 16,
    n_channels: int = 19,
    n_segments: int = 8,
    duration_sec: int = 2,
    fs: float = 512.0,
    balanced: bool = True,
    pac_config: dict = single_class_single_pac_config,
    random_seed: Optional[int] = 42,
) -> SyntheticPACDataset:
    """
    Generate synthetic PAC dataset with configurable multi-class scenarios.

    Parameters
    ----------
    n_samples : int, default=10
        Number of samples per class
    n_channels : int, default=8
        Number of EEG channels per sample
    n_segments : int, default=5
        Number of time segments per channel
    duration_sec : float, default=2.0
        Duration per segment in seconds
    fs : float, default=512.0
        Sampling frequency in Hz
    balanced : bool, default=True
        Whether to balance samples across classes
    pac_config : dict, default=single_class_single_pac_config
        Class configuration dictionary with structure:
        {
            'class_name': {
                'components': [
                    {'pha_hz': float, 'amp_hz': float, 'strength': float}
                ],
                'noise_levels': [float, ...]
            }
        }
        Use predefined configs:
        - single_class_single_pac_config
        - single_class_multi_pac_config
        - multi_class_single_pac_config
        - multi_class_multi_pac_config
    random_seed : int, optional, default=42
        Random seed for reproducibility

    Returns
    -------
    SyntheticPACDataset
        Dataset with:
        - signals: shape (n_samples, n_channels, n_segments, seq_len)
        - labels: shape (n_samples, n_channels, n_segments)
        - metadata: fs, noise levels, PAC parameters, class info

    Examples
    --------
    >>> import gpac
    >>> pac_config = gpac.dataset.multi_class_multi_pac_config
    >>> dataset = gpac.dataset.generate_pac_dataset(pac_config=pac_config)
    """

    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    gen = SyntheticDataGenerator(fs, duration_sec, random_seed)

    signals = []
    labels_int = []
    metadata_list = []

    class_names = list(pac_config.keys())
    n_classes = len(class_names)

    if balanced:
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        class_counts = [samples_per_class] * n_classes
        for ii in range(remainder):
            class_counts[ii] += 1
    else:
        class_counts = np.random.multinomial(n_samples, [1 / n_classes] * n_classes)

    for class_idx, class_name in enumerate(class_names):
        config = pac_config[class_name]
        n_class_samples = class_counts[class_idx]

        for sample_idx in range(n_class_samples):
            sample_signals = []
            for ch_idx in range(n_channels):
                channel_signals = []
                for seg_idx in range(n_segments):
                    noise_level = np.random.choice(config["noise_levels"])

                    if not config["components"]:
                        signal = gen.no_pac(noise=noise_level)
                        pac_info = []
                    else:
                        signal = np.zeros(gen.n_samples)
                        pac_info = []

                        for component in config["components"]:
                            pha_hz_options = (
                                component["pha_hz"]
                                if isinstance(component["pha_hz"], list)
                                else [component["pha_hz"]]
                            )
                            amp_hz_options = (
                                component["amp_hz"]
                                if isinstance(component["amp_hz"], list)
                                else [component["amp_hz"]]
                            )
                            strength_options = (
                                component["strength"]
                                if isinstance(component["strength"], list)
                                else [component["strength"]]
                            )

                            pha_hz = np.random.choice(pha_hz_options)
                            amp_hz = np.random.choice(amp_hz_options)
                            strength = np.random.choice(strength_options)

                            pac_component = gen.pac(pha_hz, amp_hz, strength, 0)
                            signal += pac_component

                            pac_info.append(
                                {
                                    "pha_hz": pha_hz,
                                    "amp_hz": amp_hz,
                                    "strength": strength,
                                }
                            )

                        if noise_level > 0:
                            noise = np.random.normal(0, noise_level, gen.n_samples)
                            signal += noise

                    channel_signals.append(signal)
                sample_signals.append(channel_signals)

            signals.append(sample_signals)
            labels_int.append(class_idx)
            metadata_list.append(
                {
                    "fs": fs,
                    "class_name": class_name,
                    "noise_level": noise_level,
                    "n_pac_components": len(pac_info),
                    "pac_components": pac_info,
                }
            )

    signals = np.array(signals)
    labels_int = np.array(labels_int)

    metadata_dict = {}
    for key in metadata_list[0].keys():
        if key == "pac_components":
            metadata_dict[key] = [meta[key] for meta in metadata_list]
        else:
            values = [meta[key] for meta in metadata_list]
            metadata_dict[key] = np.array(values)

    indices = np.random.permutation(len(signals))
    signals = signals[indices]
    labels_int = labels_int[indices]
    for key in metadata_dict:
        if key == "pac_components":
            metadata_dict[key] = [metadata_dict[key][idx] for idx in indices]
        else:
            metadata_dict[key] = metadata_dict[key][indices]

    return gen.create_torch_dataset(signals, labels_int, metadata_dict)


def generate_pac_dataloader(
    n_samples: int = 16,
    n_channels: int = 19,
    n_segments: int = 8,
    duration_sec: int = 2,
    fs: float = 512.0,
    balanced: bool = True,
    pac_config: dict = single_class_single_pac_config,
    random_seed: Optional[int] = 42,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Generate PyTorch DataLoader for PAC dataset.

    Parameters
    ----------
    n_samples : int, default=16
        Number of samples per class
    n_channels : int, default=19
        Number of EEG channels per sample
    n_segments : int, default=8
        Number of time segments per channel
    duration_sec : int, default=2
        Duration per segment in seconds
    fs : float, default=512.0
        Sampling frequency in Hz
    balanced : bool, default=True
        Whether to balance samples across classes
    pac_config : dict, default=single_class_single_pac_config
        Class configuration dictionary
    random_seed : int, optional, default=42
        Random seed for reproducibility
    batch_size : int, default=16
        Batch size for DataLoader
    shuffle : bool, default=True
        Whether to shuffle data
    num_workers : int, default=0
        Number of worker processes

    Returns
    -------
    DataLoader
        PyTorch DataLoader for PAC dataset
    """
    dataset = generate_pac_dataset(
        n_samples=n_samples,
        n_channels=n_channels,
        n_segments=n_segments,
        duration_sec=duration_sec,
        fs=fs,
        balanced=balanced,
        pac_config=pac_config,
        random_seed=random_seed,
    )
    generator = None
    if shuffle and random_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(random_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def generate_pac_batch(
    batch_size: int = 16,
    n_channels: int = 19,
    n_segments: int = 8,
    duration_sec: int = 2,
    fs: float = 512.0,
    balanced: bool = True,
    pac_config: dict = single_class_single_pac_config,
    random_seed: Optional[int] = 42,
) -> tuple:
    """
    Generate single batch of PAC data.

    Parameters
    ----------
    batch_size : int, default=16
        Size of the batch
    n_channels : int, default=19
        Number of EEG channels per sample
    n_segments : int, default=8
        Number of time segments per channel
    duration_sec : int, default=2
        Duration per segment in seconds
    fs : float, default=512.0
        Sampling frequency in Hz
    balanced : bool, default=True
        Whether to balance samples across classes
    pac_config : dict, default=single_class_single_pac_config
        Class configuration dictionary
    random_seed : int, optional, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple
        (signals, labels, metadata) batch tensors
    """
    dataset = generate_pac_dataset(
        n_samples=batch_size,
        n_channels=n_channels,
        n_segments=n_segments,
        duration_sec=duration_sec,
        fs=fs,
        balanced=balanced,
        pac_config=pac_config,
        random_seed=random_seed,
    )

    signals = torch.stack([dataset[ii][0] for ii in range(len(dataset))])
    labels = torch.stack([dataset[ii][1] for ii in range(len(dataset))])

    metadata_batch = {}
    for key in dataset.metadata.keys():
        if key == "class_name":
            metadata_batch[key] = [
                str(dataset.metadata[key][ii]) for ii in range(len(dataset))
            ]
        elif key == "pac_components":
            components_list = []
            for ii in range(len(dataset)):
                sample_components = []
                for comp in dataset.metadata[key][ii]:
                    sample_components.append(
                        {
                            "pha_hz": float(comp["pha_hz"]),
                            "amp_hz": float(comp["amp_hz"]),
                            "strength": float(comp["strength"]),
                        }
                    )
                components_list.append(sample_components)
            metadata_batch[key] = components_list
        else:
            metadata_batch[key] = torch.stack(
                [dataset.metadata[key][ii] for ii in range(len(dataset))]
            )

    return signals, labels, metadata_batch


# EOF
