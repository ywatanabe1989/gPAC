#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-17 18:05:19 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/src/gpac/_SyntheticDataGenerator.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/gpac/_SyntheticDataGenerator.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


## Sample shape should be: (batch_size, n_chanels, n_segments, seq_len)
class SyntheticPACDataset(Dataset):
    """
    PyTorch Dataset for synthetic PAC signals.
    This dataset can be used across different experiments consistently.
    """

    def __init__(self, signals, labels, metadata=None):
        """
        Initialize the dataset with signals and labels.
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


class SyntheticDataGenerator:
    """
    Class for generating synthetic Phase-Amplitude Coupling (PAC) signals
    with different coupling properties.
    """

    def __init__(
        self,
        fs: float = 256.0,
        duration_sec: float = 2.0,
        n_samples: int = 200,
        n_channels: int = 4,
        n_segments: int = 16,
        n_classes: int = 5,
        random_seed: Optional[int] = 42,
        noise_levels: List[float] = None,
        coupling_strengths: List[float] = None,
        channel_variation: float = 0.02,
        segment_variation: float = 0.01,
    ):
        """
        Initialize the data generator with explicit parameters.
        """
        # Set default values for noise and coupling if not provided
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        if coupling_strengths is None:
            coupling_strengths = [0.6, 0.7, 0.8, 0.9]

        # Store parameters
        self._params = {
            "fs": fs,
            "duration_sec": duration_sec,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "n_segments": n_segments,
            "n_classes": n_classes,
            "random_seed": random_seed,
            "noise_levels": noise_levels,
            "coupling_strengths": coupling_strengths,
            "channel_variation": channel_variation,
            "segment_variation": segment_variation,
        }

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Class frequency definitions
        self.class_definitions = self._create_class_definitions(n_classes)

    def _create_class_definitions(self, n_classes: int) -> Dict:
        """
        Create class definitions based on the number of classes.
        """
        # Default class definitions for up to 5 classes
        class_defs = {
            # Class 0: Low phase freq + Low amp freq
            0: {
                "name": "Low-Low",
                "pha_range": [4.0, 8.0],
                "amp_range": [50.0, 70.0],
            },
            # Class 1: Low phase freq + High amp freq
            1: {
                "name": "Low-High",
                "pha_range": [4.0, 8.0],
                "amp_range": [150.0, 170.0],
            },
            # Class 2: Medium phase freq + Medium amp freq
            2: {
                "name": "Mid-Mid",
                "pha_range": [9.0, 14.0],
                "amp_range": [100.0, 120.0],
            },
            # Class 3: High phase freq + Low amp freq
            3: {
                "name": "High-Low",
                "pha_range": [15.0, 20.0],
                "amp_range": [50.0, 70.0],
            },
            # Class 4: High phase freq + High amp freq
            4: {
                "name": "High-High",
                "pha_range": [15.0, 20.0],
                "amp_range": [150.0, 170.0],
            },
        }

        # Limit to requested number of classes
        return {
            class_id: class_defs[class_id]
            for class_id in range(min(n_classes, 5))
        }

    def _get_params(self) -> Dict:
        """
        Get current parameter settings.
        """
        return self._params.copy()

    @property
    def params(self) -> Dict:
        return self._get_params()

    def _set_params(self, **params) -> None:
        """
        Update parameters.
        """
        self._params.update(params)

        # If number of classes changed, update class definitions
        if "n_classes" in params:
            self.class_definitions = self._create_class_definitions(
                self._params["n_classes"]
            )

    def _generate_pac_signal(
        self,
        pha_freq: float,
        amp_freq: float,
        coupling_strength: float = 0.8,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate a single PAC signal with specified coupling relationship.
        """
        # Get parameters
        duration_sec = self._params["duration_sec"]
        fs = self._params["fs"]

        # Create time vector
        time = np.arange(0, duration_sec, 1 / fs)

        # Create phase signal (slow oscillation)
        phase_signal = np.sin(2 * np.pi * pha_freq * time)

        # Create amplitude Modulation based on phase
        Modulation = (
            1 + coupling_strength * np.cos(2 * np.pi * pha_freq * time)
        ) / 2

        # Create carrier signal (fast oscillation)
        carrier = np.sin(2 * np.pi * amp_freq * time)

        # Apply amplitude Modulation to carrier
        modulated_carrier = Modulation * carrier

        # Create final signal with both components
        pac_signal = phase_signal + modulated_carrier

        # Add noise
        noise = np.random.normal(0, noise_level, len(time))
        signal = pac_signal + noise

        return signal

    def _generate_class_batch(
        self, class_id: int, n_samples: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate signals for a specific class.
        """
        # Get class definition
        class_def = self.class_definitions[class_id]
        pha_range = class_def["pha_range"]
        amp_range = class_def["amp_range"]

        # Get parameters
        duration_sec = self._params["duration_sec"]
        fs = self._params["fs"]
        n_channels = self._params["n_channels"]
        n_segments = self._params["n_segments"]
        noise_levels = self._params["noise_levels"]
        coupling_strengths = self._params["coupling_strengths"]
        channel_var = self._params["channel_variation"]
        segment_var = self._params["segment_variation"]

        # Initialize arrays
        seq_len = int(duration_sec * fs)
        signals = np.zeros((n_samples, n_channels, n_segments, seq_len))

        # Initialize metadata
        metadata = {
            "class_labels": np.full(n_samples, class_id, dtype=int),
            "pha_freqs": np.zeros(n_samples),
            "amp_freqs": np.zeros(n_samples),
            "noise_levels": np.zeros(n_samples),
            "coupling_strengths": np.zeros(n_samples),
        }

        # Generate signals with different parameter combinations
        for sample_idx in range(n_samples):
            # Select parameters - cycle through noise levels and coupling strengths
            noise_level = noise_levels[sample_idx % len(noise_levels)]
            coupling = coupling_strengths[
                (sample_idx // len(noise_levels)) % len(coupling_strengths)
            ]

            # Random frequencies from ranges
            pha_freq = np.random.uniform(pha_range[0], pha_range[1])
            amp_freq = np.random.uniform(amp_range[0], amp_range[1])

            # Generate base PAC signal
            pac_signal = self._generate_pac_signal(
                pha_freq=pha_freq,
                amp_freq=amp_freq,
                coupling_strength=coupling,
                noise_level=noise_level,
            )

            # Create channels and segments
            for channel_idx in range(n_channels):
                # Add channel variation
                channel_noise = np.random.normal(
                    0, channel_var, len(pac_signal)
                )
                channel_signal = pac_signal + channel_noise

                for segment_idx in range(n_segments):
                    if n_segments > 1:
                        # Add segment variation
                        segment_noise = np.random.normal(
                            0, segment_var, len(pac_signal)
                        )
                        signals[sample_idx, channel_idx, segment_idx] = (
                            channel_signal + segment_noise
                        )
                    else:
                        signals[sample_idx, channel_idx, segment_idx] = (
                            channel_signal
                        )

            # Store metadata
            metadata["pha_freqs"][sample_idx] = pha_freq
            metadata["amp_freqs"][sample_idx] = amp_freq
            metadata["noise_levels"][sample_idx] = noise_level
            metadata["coupling_strengths"][sample_idx] = coupling

            # Progress update
            if (sample_idx + 1) % 50 == 0:
                print(
                    f"Generated {sample_idx + 1}/{n_samples} signals for class {class_id}"
                )

        return signals, metadata

    def _generate_dataset_dict(
        self, custom_params: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a complete synthetic PAC dataset for multiple classes.
        """
        # Update parameters if provided
        if custom_params:
            self._set_params(**custom_params)

        # Get parameters
        n_classes = len(self.class_definitions)
        n_samples = self._params["n_samples"]
        total_samples = n_classes * n_samples

        print(
            f"Generating dataset with {total_samples} total samples ({n_samples} per class, {n_classes} classes)"
        )

        # Initialize arrays
        seq_len = int(self._params["duration_sec"] * self._params["fs"])
        all_signals = np.zeros(
            (
                total_samples,
                self._params["n_channels"],
                self._params["n_segments"],
                seq_len,
            )
        )

        # Initialize metadata
        all_metadata = {
            "class_labels": np.zeros(total_samples, dtype=int),
            "pha_freqs": np.zeros(total_samples),
            "amp_freqs": np.zeros(total_samples),
            "noise_levels": np.zeros(total_samples),
            "coupling_strengths": np.zeros(total_samples),
            "class_names": np.array([""] * total_samples, dtype=object),
        }

        # Generate data for each class
        for class_id in self.class_definitions:
            class_name = self.class_definitions[class_id]["name"]
            print(f"\nGenerating Class {class_id}: {class_name}")

            # Calculate indices
            start_idx = class_id * n_samples
            end_idx = start_idx + n_samples

            # Generate class batch
            signals, metadata = self._generate_class_batch(
                class_id=class_id, n_samples=n_samples
            )

            # Store signals
            all_signals[start_idx:end_idx] = signals

            # Store metadata
            for key in [
                "pha_freqs",
                "amp_freqs",
                "noise_levels",
                "coupling_strengths",
            ]:
                all_metadata[key][start_idx:end_idx] = metadata[key]

            all_metadata["class_labels"][start_idx:end_idx] = metadata[
                "class_labels"
            ]
            all_metadata["class_names"][start_idx:end_idx] = class_name

        # Shuffle the dataset
        print("\nShuffling dataset...")
        indices = np.random.permutation(total_samples)

        # Apply shuffle
        shuffled_signals = all_signals[indices]
        shuffled_metadata = {k: v[indices] for k, v in all_metadata.items()}

        # Create result dictionary
        result = {
            "signals": shuffled_signals,
            "class_labels": shuffled_metadata["class_labels"],
            "metadata": shuffled_metadata,
            "params": self._params,
            "class_info": self.class_definitions,
        }

        return result

    def _create_torch_datasets(
        self, data: Dict, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Dict:
        """
        Split generated data into train/val/test datasets.
        """
        # Convert data to torch tensors
        signals_tensor = torch.tensor(data["signals"], dtype=torch.float32)
        labels_tensor = torch.tensor(data["class_labels"], dtype=torch.long)

        # Initialize metadata tensors
        metadata_tensors = {
            "pha_freqs": torch.tensor(
                data["metadata"]["pha_freqs"], dtype=torch.float32
            ),
            "amp_freqs": torch.tensor(
                data["metadata"]["amp_freqs"], dtype=torch.float32
            ),
            "noise_levels": torch.tensor(
                data["metadata"]["noise_levels"], dtype=torch.float32
            ),
            "coupling_strengths": torch.tensor(
                data["metadata"]["coupling_strengths"], dtype=torch.float32
            ),
            "class_names": data["metadata"][
                "class_names"
            ],  # Keep as numpy array
        }

        # Create stratified split indices
        unique_classes = torch.unique(labels_tensor)
        train_indices = []
        val_indices = []
        test_indices = []

        for class_id in unique_classes:
            # Get indices for this class
            class_mask = labels_tensor == class_id
            class_indices = torch.where(class_mask)[0].numpy()
            np.random.shuffle(class_indices)

            # Calculate split sizes
            n_class_samples = len(class_indices)
            n_train = max(1, int(n_class_samples * train_ratio))
            n_val = max(1, int(n_class_samples * val_ratio))

            # Ensure test set has at least one sample
            if n_train + n_val >= n_class_samples and n_class_samples > 2:
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val -= 1

            # Split indices
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train : n_train + n_val])
            test_indices.extend(class_indices[n_train + n_val :])

        # Convert indices to arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Create datasets
        train_dataset = SyntheticPACDataset(
            signals=signals_tensor[train_indices],
            labels=labels_tensor[train_indices],
            metadata={
                k: v[train_indices] for k, v in metadata_tensors.items()
            },
        )

        val_dataset = SyntheticPACDataset(
            signals=signals_tensor[val_indices],
            labels=labels_tensor[val_indices],
            metadata={k: v[val_indices] for k, v in metadata_tensors.items()},
        )

        test_dataset = SyntheticPACDataset(
            signals=signals_tensor[test_indices],
            labels=labels_tensor[test_indices],
            metadata={k: v[test_indices] for k, v in metadata_tensors.items()},
        )

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "indices": {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            },
            "class_info": data["class_info"],
            "params": data["params"],
        }

    def generate_and_split(
        self,
        custom_params: Optional[Dict] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict:
        """
        Generate a dataset and split it into train/val/test in one step.
        """
        # Generate dataset
        dataset = self._generate_dataset_dict(custom_params)

        # Split into train/val/test
        return self._create_torch_datasets(dataset, train_ratio, val_ratio)

    def plot_sample(self, dataset, idx, figsize=(12, 6), show=False):
        """
        Plot a sample from the dataset for visualization.

        Parameters
        ----------
        dataset : SyntheticPACDataset
            The dataset containing the samples to plot
        idx : int
            Index of the sample to plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        show : bool, optional
            Whether to display the plot immediately

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object (useful when show=False)
        """

        # Get sample data
        sample, label, meta = dataset[idx]

        # Create figure
        fig = plt.figure(figsize=figsize)

        # For single channel data
        if sample.shape[0] == 1:
            # Get signal from first channel and first segment
            signal = sample[0, 0].numpy()

            # Plot time series
            plt.subplot(2, 1, 1)
            plt.plot(signal)
            plt.title(f"Class {label} ({meta['class_names']})")
            plt.xlabel("Time points")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)

            # Time-frequency representation could be added here
            plt.subplot(2, 1, 2)
            plt.text(
                0.5,
                0.5,
                f"Phase freq: {meta['pha_freqs']:.2f} Hz\n"
                f"Amplitude freq: {meta['amp_freqs']:.2f} Hz\n"
                f"Coupling strength: {meta['coupling_strengths']:.2f}\n"
                f"Noise level: {meta['noise_levels']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.axis("off")
        else:
            # For multi-channel data
            n_channels = min(4, sample.shape[0])  # Show max 4 channels

            for ch_idx in range(n_channels):
                plt.subplot(n_channels, 1, ch_idx + 1)
                plt.plot(sample[ch_idx, 0].numpy())
                plt.title(f"Channel {ch_idx+1}")
                plt.grid(True, alpha=0.3)

                # Only add xlabel to bottom subplot
                if ch_idx == n_channels - 1:
                    plt.xlabel("Time points")

                plt.ylabel("Amplitude")

            plt.suptitle(
                f"Class {label} ({meta['class_names']})\n"
                f"Phase: {meta['pha_freqs']:.2f} Hz, "
                f"Amplitude: {meta['amp_freqs']:.2f} Hz"
            )

        plt.tight_layout()

        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    from pprint import pprint

    import gpac

    data_generator = gpac.SyntheticDataGenerator()
    pprint(data_generator.class_definitions)

    dataset = data_generator.generate_and_split(
        custom_params={}, train_ratio=0.7, val_ratio=0.15
    )

    # dict_keys(['train', 'val', 'test', 'indices', 'class_info', 'params'])
    dataset.keys()

    i_sample = 0
    signal, labels, metadata = dataset["train"][i_sample]
    # In [12]: signal.shape
    # Out[12]: torch.Size([4, 16, 512])

    # pac_dataset = data_generator.generate_and_split(custom_params={}, train_ratio=0.7, val_ratio=0.15)
    # dataset = data_generator._generate_dataset_dict(custom_params={})

# EOF
