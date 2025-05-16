#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 04:35:32 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/scripts/exp_01_synthetic_data_preparation/generate_synthetic_data.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_01_synthetic_data_preparation/generate_synthetic_data.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generates synthetic Phase-Amplitude Coupling (PAC) signals for multiclass classification
  - Creates five distinct signal classes with different PAC characteristics
  - Class 1: Low pha freq (4-8 Hz) coupled with low amp freq (50-70 Hz)
  - Class 2: Low pha freq (4-8 Hz) coupled with high amp freq (150-170 Hz)
  - Class 3: Medium pha freq (9-14 Hz) coupled with medium amp freq (100-120 Hz)
  - Class 4: High pha freq (15-20 Hz) coupled with low amp freq (50-70 Hz)
  - Class 5: High pha freq (15-20 Hz) coupled with high amp freq (150-170 Hz)
  - Saves generated signals to disk for later use in other experiments
  - Supports various signal parameters and noise levels

Dependencies:
  - packages:
    - mngs
    - numpy
    - torch
    - matplotlib

IO:
  - output-files:
    - ./scripts/exp_01_synthetic_data_preparation/data/synthetic_pac_signals.npz
    - ./scripts/exp_01_synthetic_data_preparation/data/synthetic_pac_signals.pt
    - ./scripts/exp_01_synthetic_data_preparation/data/synthetic_pac_examples.png
"""

"""Imports"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
from torch.utils.data import Dataset

# Get the project root directory
project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))


# DataGenerator class for creating synthetic PAC signals
class DataGenerator:
    """
    Class for generating synthetic PAC signals with different coupling properties.
    """

    def __init__(self, fs=1000.0, random_seed=None):
        self.fs = fs
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_pac_with_signal(
        self,
        n_seconds,
        pha_mod_freq,
        amp_car_freq,
        pha_bandwidth,
        amp_bandwidth,
        coupling_strength=0.8,
        noise_level=0.1,
    ):
        """
        Generate a PAC signal with specified coupling relationship.

        Parameters
        ----------
        n_seconds : float
            Duration of the signal in seconds
        pha_mod_freq : float
            Frequency of the phase modulator in Hz
        amp_car_freq : float
            Frequency of the amplitude carrier in Hz
        pha_bandwidth : float
            Bandwidth of the phase component
        amp_bandwidth : float
            Bandwidth of the amplitude component
        coupling_strength : float
            Strength of the coupling between phase and amplitude (0-1)
        noise_level : float
            Level of random noise to add to the signal

        Returns
        -------
        numpy.ndarray
            The generated PAC signal
        """
        # Create time vector
        t = np.arange(0, n_seconds, 1 / self.fs)

        # Create phase signal (slow oscillation)
        phase_signal = np.sin(2 * np.pi * pha_mod_freq * t)

        # Create amplitude modulation based on phase
        modulation = (
            1 + coupling_strength * np.cos(2 * np.pi * pha_mod_freq * t)
        ) / 2

        # Create carrier signal (fast oscillation)
        carrier = np.sin(2 * np.pi * amp_car_freq * t)

        # Apply amplitude modulation to carrier
        modulated_carrier = modulation * carrier

        # Create final signal with both components
        pac_signal = phase_signal + modulated_carrier

        # Add noise
        noise = np.random.normal(0, noise_level, len(t))
        signal = pac_signal + noise

        return signal


"""Parameters"""
# Load configuration from YAML files
PARAMS = mngs.io.load_configs()

# Default parameters for synthetic PAC signal generation
DEFAULT_PARAMS = {
    "fs": PARAMS.BASELINE.fs,  # Sampling frequency in Hz
    "duration": PARAMS.BASELINE.t_sec,  # Signal duration in seconds
    "n_samples": 200,  # Number of samples per class
    "n_channels": PARAMS.BASELINE.n_chs,  # Number of channels
    "n_segments": PARAMS.BASELINE.n_segments,  # Number of segments
    "n_classes": 5,  # Number of PAC classes
    "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5],  # Noise levels to use
    # Class 1: Low phase freq + Low amp freq
    "class1_pha_range": [4.0, 8.0],  # Theta band
    "class1_amp_range": [50.0, 70.0],  # Low gamma
    # Class 2: Low phase freq + High amp freq
    "class2_pha_range": [4.0, 8.0],  # Theta band
    "class2_amp_range": [150.0, 170.0],  # High gamma
    # Class 3: Medium phase freq + Medium amp freq
    "class3_pha_range": [9.0, 14.0],  # Alpha band
    "class3_amp_range": [100.0, 120.0],  # Mid gamma
    # Class 4: High phase freq + Low amp freq
    "class4_pha_range": [15.0, 20.0],  # Beta band
    "class4_amp_range": [50.0, 70.0],  # Low gamma
    # Class 5: High phase freq + High amp freq
    "class5_pha_range": [15.0, 20.0],  # Beta band
    "class5_amp_range": [150.0, 170.0],  # High gamma
    # Coupling parameters
    "coupling_strengths": [0.6, 0.7, 0.8, 0.9],  # Varying coupling strengths
}

"""Functions & Classes"""
class SyntheticPACDataset(Dataset):
    """
    PyTorch Dataset for synthetic PAC signals.

    This dataset can be used across different experiments consistently.
    """

    def __init__(self, signals, labels, metadata=None):
        """
        Initialize the dataset with signals and labels.

        Parameters
        ----------
        signals : torch.Tensor
            Tensor of shape (n_samples, n_channels, n_segments, seq_len)
        labels : torch.Tensor
            Tensor of shape (n_samples,) containing class labels
        metadata : dict, optional
            Additional metadata for each sample
        """
        self.signals = signals
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.metadata is not None:
            meta = {k: v[idx] for k, v in self.metadata.items()}
            return signal, label, meta

        return signal, label


def generate_class_signals(
    data_gen: DataGenerator,
    class_id: int,
    pha_range: List[float],
    amp_range: List[float],
    params: Dict,
    start_idx: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate signals for a specific class with given phase and amplitude ranges.

    Parameters
    ----------
    data_gen : DataGenerator
        Instance of DataGenerator to create signals
    class_id : int
        Class identifier (0-4 for 5 classes)
    pha_range : List[float]
        [min, max] range for phase frequency in Hz
    amp_range : List[float]
        [min, max] range for amplitude frequency in Hz
    params : Dict
        Parameters for signal generation
    start_idx : int
        Starting index for this class in the overall array

    Returns
    -------
    Tuple[np.ndarray, Dict]
        Generated signals and corresponding metadata
    """
    # Calculate sizes
    n_noise_levels = len(params["noise_levels"])
    n_coupling = len(params["coupling_strengths"])
    n_samples_per_class = params["n_samples"]

    # Handle small sample sizes
    if n_samples_per_class <= 0:
        # If n_samples is 0 or negative, return empty arrays
        return np.zeros(
            (
                0,
                params["n_channels"],
                params["n_segments"],
                int(params["duration"] * params["fs"]),
            )
        ), {
            "class_labels": np.array([], dtype=int),
            "pha_freqs": np.array([]),
            "amp_freqs": np.array([]),
            "noise_levels": np.array([]),
            "coupling_strengths": np.array([]),
            "sample_ids": np.array([], dtype=int),
        }

    # Special case for very small sample sizes
    if n_samples_per_class < n_noise_levels * n_coupling:
        # Create at most n_samples_per_class samples (one per condition up to the limit)
        samples_per_condition = 1
        # We'll use as many conditions as we have samples
        total_samples = min(n_samples_per_class, n_noise_levels * n_coupling)
    else:
        # Normal case - distribute samples across all conditions
        samples_per_condition = n_samples_per_class // (
            n_noise_levels * n_coupling
        )
        total_samples = samples_per_condition * n_noise_levels * n_coupling

    seq_len = int(params["duration"] * params["fs"])
    signals = np.zeros(
        (total_samples, params["n_channels"], params["n_segments"], seq_len)
    )

    # Metadata storage
    metadata = {
        "class_labels": np.full(total_samples, class_id, dtype=int),
        "pha_freqs": np.zeros(total_samples),
        "amp_freqs": np.zeros(total_samples),
        "noise_levels": np.zeros(total_samples),
        "coupling_strengths": np.zeros(total_samples),
        "sample_ids": np.zeros(total_samples, dtype=int),
    }

    # Generate signals for this class
    sample_idx = 0
    condition_count = 0

    # For small sample sizes, limit the number of conditions we use
    for noise_idx, noise_level in enumerate(params["noise_levels"]):
        if sample_idx >= total_samples:
            break

        for coupling_idx, coupling_strength in enumerate(
            params["coupling_strengths"]
        ):
            if sample_idx >= total_samples:
                break

            # For very small sample sizes, we track how many conditions we've used
            if n_samples_per_class < n_noise_levels * n_coupling:
                if condition_count >= total_samples:
                    break
                condition_count += 1

            for i in range(samples_per_condition):
                # Random frequencies from ranges
                pha_freq = np.random.uniform(pha_range[0], pha_range[1])
                amp_freq = np.random.uniform(amp_range[0], amp_range[1])

                # Generate PAC signal
                pac_data = data_gen.generate_pac_with_signal(
                    n_seconds=params["duration"],
                    pha_mod_freq=pha_freq,
                    amp_car_freq=amp_freq,
                    pha_bandwidth=pha_freq / 4,
                    amp_bandwidth=amp_freq / 4,
                    coupling_strength=coupling_strength,
                    noise_level=noise_level,
                )

                # Reshape to match expected dimensions and create multiple channels/segments
                signal_1d = pac_data.reshape(1, 1, -1)

                # Repeat across channels - make slightly different versions
                for channel_idx in range(params["n_channels"]):
                    # Add small channel-specific variations
                    channel_variation = np.random.normal(
                        0, 0.02, len(pac_data)
                    )
                    channel_signal = pac_data + channel_variation

                    for segment_idx in range(params["n_segments"]):
                        # Add small segment-specific variations if multiple segments
                        if params["n_segments"] > 1:
                            segment_variation = np.random.normal(
                                0, 0.01, len(pac_data)
                            )
                            segment_signal = channel_signal + segment_variation
                        else:
                            segment_signal = channel_signal

                        signals[sample_idx, channel_idx, segment_idx, :] = (
                            segment_signal
                        )

                # Store metadata
                metadata["pha_freqs"][sample_idx] = pha_freq
                metadata["amp_freqs"][sample_idx] = amp_freq
                metadata["noise_levels"][sample_idx] = noise_level
                metadata["coupling_strengths"][sample_idx] = coupling_strength
                metadata["sample_ids"][sample_idx] = i

                sample_idx += 1

                # Progress update
                global_idx = start_idx + sample_idx
                if global_idx % 50 == 0:
                    print(f"Generated {global_idx} signals")

    return signals, metadata


def generate_synthetic_pac_signals(params: Dict) -> Dict:
    """
    Generate synthetic PAC signals for multiple classes.

    Parameters
    ----------
    params : Dict
        Dictionary with signal generation parameters

    Returns
    -------
    Dict
        Dictionary containing generated signals and metadata
    """
    # Set up data generator
    data_gen = DataGenerator(fs=params["fs"], random_seed=42)

    # Calculate total signals with handling for small sample sizes
    n_noise_levels = len(params["noise_levels"])
    n_coupling = len(params["coupling_strengths"])
    n_classes = params["n_classes"]  # Number of different PAC classes
    n_samples_per_class = params["n_samples"]

    # Handle small sample sizes
    if n_samples_per_class <= 0:
        # Empty dataset case
        seq_len = int(params["duration"] * params["fs"])
        empty_signals = np.zeros(
            (0, params["n_channels"], params["n_segments"], seq_len)
        )
        empty_metadata = {
            "class_labels": np.array([], dtype=int),
            "pha_freqs": np.array([]),
            "amp_freqs": np.array([]),
            "noise_levels": np.array([]),
            "coupling_strengths": np.array([]),
            "sample_ids": np.array([], dtype=int),
        }
        return {
            "signals_np": empty_signals,
            "signals_pt": torch.from_numpy(empty_signals.astype(np.float32)),
            "metadata": empty_metadata,
            "params": params,
        }

    if n_samples_per_class < n_noise_levels * n_coupling:
        # Each class gets n_samples_per_class samples with one sample per condition used
        samples_per_condition = 1
        # We'll use as many conditions as we have samples
        class_total_samples = min(
            n_samples_per_class, n_noise_levels * n_coupling
        )
        total_samples = class_total_samples * n_classes
    else:
        # Normal case - distribute samples across all conditions
        samples_per_condition = max(
            1, n_samples_per_class // (n_noise_levels * n_coupling)
        )
        total_samples = (
            samples_per_condition * n_noise_levels * n_coupling * n_classes
        )

    print(
        f"Generating {total_samples} total signals "
        + f"({n_samples_per_class} per class, {n_classes} classes)"
    )

    # Pre-allocate final arrays
    seq_len = int(params["duration"] * params["fs"])
    signals_np = np.zeros(
        (total_samples, params["n_channels"], params["n_segments"], seq_len)
    )

    # Metadata dictionary for the entire dataset
    metadata = {
        "class_labels": np.zeros(total_samples, dtype=int),
        "pha_freqs": np.zeros(total_samples),
        "amp_freqs": np.zeros(total_samples),
        "noise_levels": np.zeros(total_samples),
        "coupling_strengths": np.zeros(total_samples),
        "sample_ids": np.zeros(total_samples, dtype=int),
    }

    # Generate signals for each class
    class_size = n_samples_per_class

    # Class 1: Low phase freq + Low amp freq
    print("\nGenerating Class 1: Low phase freq + Low amp freq")
    signals_class1, metadata_class1 = generate_class_signals(
        data_gen,
        0,
        params["class1_pha_range"],
        params["class1_amp_range"],
        params,
        0,
    )
    idx_start = 0
    idx_end = len(signals_class1)
    signals_np[idx_start:idx_end] = signals_class1
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class1[key]

    # Class 2: Low phase freq + High amp freq
    print("\nGenerating Class 2: Low phase freq + High amp freq")
    signals_class2, metadata_class2 = generate_class_signals(
        data_gen,
        1,
        params["class2_pha_range"],
        params["class2_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class2)
    signals_np[idx_start:idx_end] = signals_class2
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class2[key]

    # Class 3: Medium phase freq + Medium amp freq
    print("\nGenerating Class 3: Medium phase freq + Medium amp freq")
    signals_class3, metadata_class3 = generate_class_signals(
        data_gen,
        2,
        params["class3_pha_range"],
        params["class3_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class3)
    signals_np[idx_start:idx_end] = signals_class3
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class3[key]

    # Class 4: High phase freq + Low amp freq
    print("\nGenerating Class 4: High phase freq + Low amp freq")
    signals_class4, metadata_class4 = generate_class_signals(
        data_gen,
        3,
        params["class4_pha_range"],
        params["class4_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class4)
    signals_np[idx_start:idx_end] = signals_class4
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class4[key]

    # Class 5: High phase freq + High amp freq
    print("\nGenerating Class 5: High phase freq + High amp freq")
    signals_class5, metadata_class5 = generate_class_signals(
        data_gen,
        4,
        params["class5_pha_range"],
        params["class5_amp_range"],
        params,
        idx_end,
    )
    idx_start = idx_end
    idx_end = idx_start + len(signals_class5)
    signals_np[idx_start:idx_end] = signals_class5
    for key in metadata:
        metadata[key][idx_start:idx_end] = metadata_class5[key]

    # Shuffle the data while maintaining correspondence between signals and metadata
    print("\nShuffling the dataset...")
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    signals_np = signals_np[indices]
    for key in metadata:
        metadata[key] = metadata[key][indices]

    # Create PyTorch tensor
    signals_pt = torch.from_numpy(signals_np.astype(np.float32))

    # Return complete dataset
    return {
        "signals_np": signals_np,
        "signals_pt": signals_pt,
        "metadata": metadata,
        "params": params,
    }


def save_synthetic_data(data: Dict, output_dir: Path) -> None:
    """
    Save generated data to disk in multiple formats.

    Parameters
    ----------
    data : Dict
        Dictionary containing generated signals and metadata
    output_dir : Path
        Directory to save the output files
    """

    # Save numpy version
    # Create a dictionary with all the data to save
    save_data = {
        "signals": data["signals_np"],
        "class_labels": data["metadata"]["class_labels"],
        "pha_freqs": data["metadata"]["pha_freqs"],
        "amp_freqs": data["metadata"]["amp_freqs"],
        "noise_levels": data["metadata"]["noise_levels"],
        "coupling_strengths": data["metadata"]["coupling_strengths"],
        "sample_ids": data["metadata"]["sample_ids"],
    }
    # Add parameters with param_ prefix
    for k, v in data["params"].items():
        save_data[f"param_{k}"] = v

    # Save the data
    mngs.io.save(
        save_data,
        "./data/exp_01/synthetic_pac_signals.npz",
        symlink_from_cwd=True,
    )

    # Save PyTorch version (without numpy arrays to save space)
    mngs.io.save(
        {
            "signals": data["signals_pt"],
            "metadata": {
                "class_labels": torch.tensor(data["metadata"]["class_labels"]),
                "pha_freqs": torch.tensor(data["metadata"]["pha_freqs"]),
                "amp_freqs": torch.tensor(data["metadata"]["amp_freqs"]),
                "noise_levels": torch.tensor(data["metadata"]["noise_levels"]),
                "coupling_strengths": torch.tensor(
                    data["metadata"]["coupling_strengths"]
                ),
                "sample_ids": torch.tensor(data["metadata"]["sample_ids"]),
            },
            "params": data["params"],
        },
        "./data/exp_01/synthetic_pac_signals.pt",
        symlink_from_cwd=True,
    )

    # Save PyTorch dataset
    dataset = SyntheticPACDataset(
        signals=data["signals_pt"],
        labels=torch.tensor(data["metadata"]["class_labels"]),
        metadata={
            "pha_freqs": torch.tensor(data["metadata"]["pha_freqs"]),
            "amp_freqs": torch.tensor(data["metadata"]["amp_freqs"]),
            "noise_levels": torch.tensor(data["metadata"]["noise_levels"]),
            "coupling_strengths": torch.tensor(
                data["metadata"]["coupling_strengths"]
            ),
            "sample_ids": torch.tensor(data["metadata"]["sample_ids"]),
        },
    )

    # Use mngs.io.save for dataset
    mngs.io.save(
        dataset,
        "./data/exp_01/synthetic_pac_dataset.pt",
        symlink_from_cwd=True,
    )

    print(f"Data saved to {output_dir}")


def plot_example_signals(data: Dict, output_dir: Path) -> None:
    """
    Plot example signals from each class for visualization.

    Parameters
    ----------
    data : Dict
        Dictionary containing generated signals and metadata
    output_dir : Path
        Directory to save the output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get number of classes
    n_classes = data["params"]["n_classes"]

    # Plot examples from each class
    fig, axes = plt.subplots(n_classes, 2, figsize=(15, 5 * n_classes))
    fig.suptitle(
        "Example Synthetic PAC Signals for Multi-class Classification",
        fontsize=16,
    )

    # Class names for reference
    class_descriptions = {
        0: "Low phase + Low amp",
        1: "Low phase + High amp",
        2: "Medium phase + Medium amp",
        3: "High phase + Low amp",
        4: "High phase + High amp",
    }

    class_names = [
        f"Class {i+1}: {class_descriptions[i]}" for i in range(n_classes)
    ]

    # Handle empty dataset case
    if len(data["signals_np"]) == 0:
        for class_id in range(n_classes):
            for i in range(2):
                ax = axes[class_id, i]
                ax.text(
                    0.5,
                    0.5,
                    "No examples generated",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_title(f"{class_names[class_id]} (No examples)")
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        # Save figure using mngs.io.save
        mngs.io.save(
            fig,
            "./data/exp_01/synthetic_pac_examples.png",
            symlink_from_cwd=True,
        )
        plt.close(fig)
        return

    # Get indices for each class
    for class_id in range(n_classes):
        class_indices = np.where(data["metadata"]["class_labels"] == class_id)[
            0
        ]

        # Handle case where this class has no samples
        if len(class_indices) == 0:
            for i in range(2):
                ax = axes[class_id, i]
                ax.text(
                    0.5,
                    0.5,
                    f"No examples for {class_names[class_id]}",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.set_title(f"{class_names[class_id]} (No examples)")
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        # Handle case where this class has only one sample
        num_examples = min(2, len(class_indices))

        # Select random examples, with replacement if we don't have 2 examples
        np.random.seed(42 + class_id)  # Different seed for each class
        class_examples = np.random.choice(
            class_indices, num_examples, replace=(len(class_indices) < 2)
        )

        # If we only have one example, duplicate it for the second plot
        if len(class_examples) == 1:
            class_examples = np.array([class_examples[0], class_examples[0]])

        # Plot the examples
        for i, ax in enumerate(axes[class_id]):
            idx = class_examples[i]
            signal = data["signals_np"][
                idx, 0, 0, :
            ]  # First channel, first segment

            # Time vector
            fs = data["params"]["fs"]
            duration = data["params"]["duration"]
            time_vector = np.linspace(0, duration, len(signal))

            # Plot signal
            ax.plot(time_vector, signal)

            # Add metadata
            pha_freq = data["metadata"]["pha_freqs"][idx]
            amp_freq = data["metadata"]["amp_freqs"][idx]
            noise = data["metadata"]["noise_levels"][idx]
            coupling = data["metadata"]["coupling_strengths"][idx]

            ax.set_title(
                f"{class_names[class_id]}\nP:{pha_freq:.1f}Hz, A:{amp_freq:.1f}Hz, N:{noise:.2f}, C:{coupling:.2f}"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    # Save figure using mngs.io.save
    mngs.io.save(
        fig,
        "./data/exp_01/synthetic_pac_examples.png",
        symlink_from_cwd=True,
    )
    plt.close(fig)

    # Create a summary visualization showing the phase-amplitude frequency space
    fig, ax = plt.subplots(figsize=(10, 8))

    # Different color for each class (add more if n_classes > 5)
    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
    ]
    colors = colors[:n_classes]  # Use only as many colors as we have classes

    # Scatter plot of phase vs amplitude frequencies
    for class_id in range(n_classes):
        mask = data["metadata"]["class_labels"] == class_id
        pha_freqs = data["metadata"]["pha_freqs"][mask]
        amp_freqs = data["metadata"]["amp_freqs"][mask]

        # Plot up to 10% random samples to avoid overcrowding (at least 1)
        plot_size = max(1, len(pha_freqs) // 10)
        indices = np.random.choice(
            len(pha_freqs),
            size=plot_size,
            replace=(len(pha_freqs) < plot_size),
        )

        ax.scatter(
            pha_freqs[indices],
            amp_freqs[indices],
            c=colors[class_id],
            alpha=0.7,
            label=class_names[class_id],
        )

    # Add class regions - only draw if we have the standard 5 classes
    class_regions = []

    if n_classes >= 5 and all(
        k in data["params"]
        for k in [
            "class1_pha_range",
            "class1_amp_range",
            "class2_pha_range",
            "class2_amp_range",
            "class3_pha_range",
            "class3_amp_range",
            "class4_pha_range",
            "class4_amp_range",
            "class5_pha_range",
            "class5_amp_range",
        ]
    ):
        # Class 1: Low phase + Low amp
        rect1 = plt.Rectangle(
            (
                data["params"]["class1_pha_range"][0],
                data["params"]["class1_amp_range"][0],
            ),
            data["params"]["class1_pha_range"][1]
            - data["params"]["class1_pha_range"][0],
            data["params"]["class1_amp_range"][1]
            - data["params"]["class1_amp_range"][0],
            linewidth=1,
            edgecolor=colors[0],
            facecolor=colors[0],
            alpha=0.1,
        )
        class_regions.append(rect1)

        # Class 2: Low phase + High amp
        rect2 = plt.Rectangle(
            (
                data["params"]["class2_pha_range"][0],
                data["params"]["class2_amp_range"][0],
            ),
            data["params"]["class2_pha_range"][1]
            - data["params"]["class2_pha_range"][0],
            data["params"]["class2_amp_range"][1]
            - data["params"]["class2_amp_range"][0],
            linewidth=1,
            edgecolor=colors[1],
            facecolor=colors[1],
            alpha=0.1,
        )
        class_regions.append(rect2)

        # Class 3: Medium phase + Medium amp
        rect3 = plt.Rectangle(
            (
                data["params"]["class3_pha_range"][0],
                data["params"]["class3_amp_range"][0],
            ),
            data["params"]["class3_pha_range"][1]
            - data["params"]["class3_pha_range"][0],
            data["params"]["class3_amp_range"][1]
            - data["params"]["class3_amp_range"][0],
            linewidth=1,
            edgecolor=colors[2],
            facecolor=colors[2],
            alpha=0.1,
        )
        class_regions.append(rect3)

        # Class 4: High phase + Low amp
        rect4 = plt.Rectangle(
            (
                data["params"]["class4_pha_range"][0],
                data["params"]["class4_amp_range"][0],
            ),
            data["params"]["class4_pha_range"][1]
            - data["params"]["class4_pha_range"][0],
            data["params"]["class4_amp_range"][1]
            - data["params"]["class4_amp_range"][0],
            linewidth=1,
            edgecolor=colors[3],
            facecolor=colors[3],
            alpha=0.1,
        )
        class_regions.append(rect4)

        # Class 5: High phase + High amp
        rect5 = plt.Rectangle(
            (
                data["params"]["class5_pha_range"][0],
                data["params"]["class5_amp_range"][0],
            ),
            data["params"]["class5_pha_range"][1]
            - data["params"]["class5_pha_range"][0],
            data["params"]["class5_amp_range"][1]
            - data["params"]["class5_amp_range"][0],
            linewidth=1,
            edgecolor=colors[4],
            facecolor=colors[4],
            alpha=0.1,
        )
        class_regions.append(rect5)

    for rect in class_regions:
        ax.add_patch(rect)

    ax.set_title("Phase-Amplitude Frequency Space", fontsize=16)
    ax.set_xlabel("Phase Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Amplitude Frequency (Hz)", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    # Save figure using mngs.io.save
    mngs.io.save(
        fig,
        "./data/exp_01/frequency_space.png",
        symlink_from_cwd=True,
    )
    plt.close(fig)


def create_pac_dataset_splits(
    data: Dict,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    Create train, validation, and test dataset splits.

    Parameters
    ----------
    data : Dict
        Dictionary containing generated signals and metadata
    output_dir : Path
        Directory to save the output files
    train_ratio : float
        Proportion of data to use for training
    val_ratio : float
        Proportion of data to use for validation (rest goes to test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get total number of samples
    n_samples = len(data["signals_pt"])

    # Handle empty dataset
    if n_samples == 0:
        print("No samples available for dataset splits")
        empty_dataset = SyntheticPACDataset(
            signals=torch.zeros((0, 4, 1, 2000), dtype=torch.float32),
            labels=torch.zeros(0, dtype=torch.long),
            metadata={},
        )
        # Save empty datasets
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/train_dataset.pt",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/val_dataset.pt",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            empty_dataset,
            "./data/exp_01/test_dataset.pt",
            symlink_from_cwd=True,
        )
        mngs.io.save(
            {
                "train_indices": np.array([]),
                "val_indices": np.array([]),
                "test_indices": np.array([])
            },
            "./data/exp_01/dataset_splits.npz",
            symlink_from_cwd=True,
        )
        return

    # Create stratified split indices
    class_labels = data["metadata"]["class_labels"]
    unique_classes = np.unique(class_labels)
    n_classes = len(unique_classes)

    # Indices for each split by class
    train_indices = []
    val_indices = []
    test_indices = []

    for class_id in unique_classes:
        # Get indices for this class
        class_indices = np.where(class_labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Calculate split sizes
        n_class_samples = len(class_indices)

        # Handle small sample counts
        if n_class_samples <= 1:
            # If only one sample, use it for training
            train_indices.extend(class_indices)
            continue

        if n_class_samples == 2:
            # If two samples, use one for training, one for validation
            train_indices.append(class_indices[0])
            val_indices.append(class_indices[1])
            continue

        # Normal case - calculate proportional splits
        n_train = max(1, int(n_class_samples * train_ratio))
        n_val = max(1, int(n_class_samples * val_ratio))

        # Ensure we have at least one sample for test if there are enough samples
        if n_train + n_val >= n_class_samples and n_class_samples > 2:
            # Reduce validation to ensure at least one test sample
            n_val = max(1, n_class_samples - n_train - 1)

        # Adjust if we would exceed the number of samples
        if n_train + n_val > n_class_samples:
            if n_train > 1:
                n_train -= 1
            else:
                n_val -= 1

        # Split indices
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])

    # Shuffle the indices within each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Create the datasets
    train_dataset = SyntheticPACDataset(
        signals=data["signals_pt"][train_indices],
        labels=torch.tensor(class_labels[train_indices]),
        metadata={
            "pha_freqs": torch.tensor(
                data["metadata"]["pha_freqs"][train_indices]
            ),
            "amp_freqs": torch.tensor(
                data["metadata"]["amp_freqs"][train_indices]
            ),
            "noise_levels": torch.tensor(
                data["metadata"]["noise_levels"][train_indices]
            ),
            "coupling_strengths": torch.tensor(
                data["metadata"]["coupling_strengths"][train_indices]
            ),
            "sample_ids": torch.tensor(
                data["metadata"]["sample_ids"][train_indices]
            ),
        },
    )

    val_dataset = SyntheticPACDataset(
        signals=data["signals_pt"][val_indices],
        labels=torch.tensor(class_labels[val_indices]),
        metadata={
            "pha_freqs": torch.tensor(
                data["metadata"]["pha_freqs"][val_indices]
            ),
            "amp_freqs": torch.tensor(
                data["metadata"]["amp_freqs"][val_indices]
            ),
            "noise_levels": torch.tensor(
                data["metadata"]["noise_levels"][val_indices]
            ),
            "coupling_strengths": torch.tensor(
                data["metadata"]["coupling_strengths"][val_indices]
            ),
            "sample_ids": torch.tensor(
                data["metadata"]["sample_ids"][val_indices]
            ),
        },
    )

    test_dataset = SyntheticPACDataset(
        signals=data["signals_pt"][test_indices],
        labels=torch.tensor(class_labels[test_indices]),
        metadata={
            "pha_freqs": torch.tensor(
                data["metadata"]["pha_freqs"][test_indices]
            ),
            "amp_freqs": torch.tensor(
                data["metadata"]["amp_freqs"][test_indices]
            ),
            "noise_levels": torch.tensor(
                data["metadata"]["noise_levels"][test_indices]
            ),
            "coupling_strengths": torch.tensor(
                data["metadata"]["coupling_strengths"][test_indices]
            ),
            "sample_ids": torch.tensor(
                data["metadata"]["sample_ids"][test_indices]
            ),
        },
    )

    # Save the splits using mngs.io.save
    mngs.io.save(
        train_dataset,
        "./data/exp_01/train_dataset.pt",
        symlink_from_cwd=True,
    )
    mngs.io.save(
        val_dataset,
        "./data/exp_01/val_dataset.pt",
        symlink_from_cwd=True,
    )
    mngs.io.save(
        test_dataset,
        "./data/exp_01/test_dataset.pt",
        symlink_from_cwd=True,
    )

    # Save indices for reference using mngs.io.save
    mngs.io.save(
        {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        },
        "./data/exp_01/dataset_splits.npz",
        symlink_from_cwd=True,
    )

    print(f"Dataset splits created and saved to {output_dir}")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")


def main(args):
    """Main function to generate and save synthetic data."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Prepare parameters
    params = DEFAULT_PARAMS.copy()
    if args.n_samples:
        params["n_samples"] = args.n_samples
    if args.n_channels:
        params["n_channels"] = args.n_channels
    if args.duration:
        params["duration"] = args.duration

    # Create output directory
    output_dir = Path(__DIR__) / "data"

    # Generate synthetic data
    print("Generating synthetic PAC signals...")
    generated_data = generate_synthetic_pac_signals(params)

    # Save raw data
    print("Saving generated signals...")
    save_synthetic_data(generated_data, output_dir)

    # Generate visualizations
    print("Creating visualizations...")
    plot_example_signals(generated_data, output_dir)

    # Create dataset splits
    print("Creating train/val/test dataset splits...")
    create_pac_dataset_splits(generated_data, output_dir)

    # Report completion
    total_signals = len(generated_data["signals_np"])
    print(
        f"Generation completed successfully. {total_signals} signals generated."
    )
    class_counts = np.bincount(generated_data["metadata"]["class_labels"])
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} signals")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic PAC signals for multi-class classification"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help=f"Number of samples per class (default: {DEFAULT_PARAMS['n_samples']})",
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        help=f"Number of channels per sample (default: {DEFAULT_PARAMS['n_channels']})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help=f"Duration of each signal in seconds (default: {DEFAULT_PARAMS['duration']})",
    )

    args = parser.parse_args()
    return args


def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt

    import sys

    import matplotlib.pyplot as plt
    import mngs

    args = parse_args()

    # Initialize mngs framework
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    # Run main function
    exit_status = main(args)

    # Cleanup
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
