#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:18:42 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/dataset/_SyntheticDataGenerator.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/dataset/_SyntheticDataGenerator.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ._SyntheticPACDataset import SyntheticPACDataset

"""
Synthetic PAC data generator for testing and benchmarking.

This module provides tools to generate synthetic signals with known
phase-amplitude coupling properties for testing PAC algorithms.
"""


# ../../../../../../home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py . [  9%]
# ...F...F..                                                               [100%]

# =================================== FAILURES ===================================_________ TestSyntheticDataGenerator.test_dataset_generation_balanced __________/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py:89: in test_dataset_generation_balanced
#     assert abs(n_pac - n_no_pac) < n_samples * 0.5
# E   assert np.int64(62) < (100 * 0.5)
# E    +  where np.int64(62) = abs((np.int64(81) - np.int64(19)))
# _______________ TestSyntheticDataGenerator.test_reproducibility ________________/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py:141: in test_reproducibility
#     assert np.allclose(signal1, signal2)
# E   assert False
# E    +  where False = <function allclose at 0x14e92138d0f0>(array([ 0.04967142,  0.54976424,  0.8258099 , ..., -0.49725719,\n       -0.6512343 , -0.29847326], shape=(1024,)), array([ 0.17275432,  0.60722304,  0.7648414 , ..., -0.32778412,\n       -0.64877814, -0.51363032], shape=(1024,)))
# E    +    where <function allclose at 0x14e92138d0f0> = np.allclose
# =========================== short test summary info ============================FAILED ../../../../../../home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py::TestSyntheticDataGenerator::test_dataset_generation_balanced - assert np.int64(62) < (100 * 0.5)
# FAILED ../../../../../../home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py::TestSyntheticDataGenerator::test_reproducibility - assert False
# ========================= 2 failed, 9 passed in 0.12s ==========================(.env-3.10) (SpartanGPU) gPAC $


# class SyntheticDataGenerator:
#     """Generate synthetic Phase-Amplitude Coupling (PAC) signals."""

#     def __init__(
#         self,
#         fs: float = 512.0,
#         duration_sec: float = 2.0,
#         random_seed: Optional[int] = 42,
#     ):
#         self.fs = fs
#         self.duration_sec = duration_sec
#         self.n_samples = int(fs * duration_sec)

#         if random_seed is not None:
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)

#     def pac(
#         self,
#         phase_freq: float = 10.0,
#         amp_freq: float = 80.0,
#         coupling: float = 0.5,
#         noise: float = 0.1,
#     ) -> np.ndarray:
#         """Generate single PAC signal (simplified interface)."""
#         return self.generate_pac_signal(phase_freq, amp_freq, coupling, noise)

#     def no_pac(
#         self, freq1: float = 10.0, freq2: float = 80.0, noise: float = 0.1
#     ) -> np.ndarray:
#         """Generate control signal without PAC (simplified interface)."""
#         return self.generate_no_pac_signal(freq1, freq2, noise)

#     def dataset(
#         self, n_samples: int = 100, balanced: bool = True
#     ) -> SyntheticPACDataset:
#         """Generate complete dataset and return as PyTorch dataset (one-step)."""
#         data = self.generate_dataset(n_samples, include_no_pac=balanced)
#         return self.create_torch_dataset(
#             data["signals"], data["labels"], data["metadata"]
#         )

#     def quick_dataset(
#         self, n_pac: int = 50, n_no_pac: int = 50
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Generate dataset with specified counts, return arrays only."""
#         all_signals = []
#         all_labels = []

#         # Default parameters for quick generation
#         phase_freqs = [8, 12, 16]
#         amp_freqs = [60, 80, 100]
#         couplings = [0.3, 0.5, 0.7]

#         # Generate PAC signals
#         for idx in range(n_pac):
#             pf = np.random.choice(phase_freqs)
#             af = np.random.choice(amp_freqs)
#             coupling = np.random.choice(couplings)
#             signal = self.pac(pf, af, coupling)
#             all_signals.append(signal)
#             all_labels.append(1)

#         # Generate no-PAC signals
#         for idx in range(n_no_pac):
#             f1 = np.random.uniform(5, 20)
#             f2 = np.random.uniform(50, 120)
#             signal = self.no_pac(f1, f2)
#             all_signals.append(signal)
#             all_labels.append(0)

#         signals = np.array(all_signals)
#         labels = np.array(all_labels)

#         # Shuffle
#         indices = np.random.permutation(len(signals))
#         return signals[indices], labels[indices]

#     def generate_pac_signal(
#         self,
#         phase_freq: float,
#         amp_freq: float,
#         coupling_strength: float = 0.5,
#         noise_level: float = 0.1,
#         coupling_type: str = "amplitude",
#     ) -> np.ndarray:
#         """Generate PAC signal with detailed parameters."""
#         t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
#         phase_signal = np.sin(2 * np.pi * phase_freq * t)

#         if coupling_type == "amplitude":
#             modulation = 0.5 + 0.5 * coupling_strength * np.sin(
#                 2 * np.pi * phase_freq * t
#             )
#             carrier = np.sin(2 * np.pi * amp_freq * t)
#             amp_signal = modulation * carrier
#         elif coupling_type == "phase":
#             phase_mod = coupling_strength * np.sin(2 * np.pi * phase_freq * t)
#             amp_signal = np.sin(2 * np.pi * amp_freq * t + phase_mod)

#         signal = phase_signal + amp_signal

#         if noise_level > 0:
#             noise = np.random.normal(0, noise_level, self.n_samples)
#             signal += noise

#         return signal

#     def generate_no_pac_signal(
#         self,
#         freq1: float = 10.0,
#         freq2: float = 80.0,
#         noise_level: float = 0.1,
#     ) -> np.ndarray:
#         """Generate signal without PAC."""
#         t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
#         signal1 = np.sin(2 * np.pi * freq1 * t)
#         signal2 = np.sin(2 * np.pi * freq2 * t)
#         signal = signal1 + signal2

#         if noise_level > 0:
#             noise = np.random.normal(0, noise_level, self.n_samples)
#             signal += noise

#         return signal

#     def generate_multi_pac_signal(
#         self,
#         pac_pairs: List[Tuple[float, float, float]],
#         noise_level: float = 0.1,
#     ) -> np.ndarray:
#         """Generate signal with multiple PAC pairs."""
#         t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
#         signal = np.zeros(self.n_samples)

#         for phase_freq, amp_freq, coupling in pac_pairs:
#             phase_signal = np.sin(2 * np.pi * phase_freq * t)
#             modulation = 0.5 + 0.5 * coupling * np.sin(
#                 2 * np.pi * phase_freq * t
#             )
#             amp_signal = modulation * np.sin(2 * np.pi * amp_freq * t)
#             signal += phase_signal + amp_signal

#         signal = signal / len(pac_pairs)

#         if noise_level > 0:
#             noise = np.random.normal(0, noise_level, self.n_samples)
#             signal += noise

#         return signal

#     def generate_dataset(
#         self,
#         n_samples: int = 100,
#         pac_params: Optional[Dict] = None,
#         include_no_pac: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         """Generate complete dataset."""
#         if pac_params is None:
#             pac_params = {
#                 "phase_freqs": [6, 10, 15],
#                 "amp_freqs": [60, 80, 100],
#                 "coupling_strengths": [0.3, 0.5, 0.7],
#                 "noise_levels": [0.05, 0.1, 0.2],
#             }

#         n_pac_combinations = (
#             len(pac_params["phase_freqs"])
#             * len(pac_params["amp_freqs"])
#             * len(pac_params["coupling_strengths"])
#         )

#         if include_no_pac:
#             n_per_condition = n_samples // (n_pac_combinations + 1)
#             n_no_pac = n_samples - (n_per_condition * n_pac_combinations)
#         else:
#             n_per_condition = n_samples // n_pac_combinations
#             n_no_pac = 0

#         all_signals = []
#         all_labels = []
#         metadata = {
#             "phase_freq": [],
#             "amp_freq": [],
#             "coupling_strength": [],
#             "noise_level": [],
#             "has_pac": [],
#         }

#         sample_idx = 0
#         for phase_freq in pac_params["phase_freqs"]:
#             for amp_freq in pac_params["amp_freqs"]:
#                 for coupling in pac_params["coupling_strengths"]:
#                     for _ in range(n_per_condition):
#                         if sample_idx >= n_samples:
#                             break

#                         noise = np.random.choice(pac_params["noise_levels"])
#                         signal = self.generate_pac_signal(
#                             phase_freq, amp_freq, coupling, noise
#                         )

#                         all_signals.append(signal)
#                         all_labels.append(1)
#                         metadata["phase_freq"].append(phase_freq)
#                         metadata["amp_freq"].append(amp_freq)
#                         metadata["coupling_strength"].append(coupling)
#                         metadata["noise_level"].append(noise)
#                         metadata["has_pac"].append(True)
#                         sample_idx += 1

#         if include_no_pac:
#             for _ in range(n_no_pac):
#                 noise = np.random.choice(pac_params["noise_levels"])
#                 freq1 = np.random.uniform(5, 20)
#                 freq2 = np.random.uniform(50, 120)
#                 signal = self.generate_no_pac_signal(freq1, freq2, noise)

#                 all_signals.append(signal)
#                 all_labels.append(0)
#                 metadata["phase_freq"].append(freq1)
#                 metadata["amp_freq"].append(freq2)
#                 metadata["coupling_strength"].append(0.0)
#                 metadata["noise_level"].append(noise)
#                 metadata["has_pac"].append(False)

#         signals = np.array(all_signals)
#         labels = np.array(all_labels)

#         for key in metadata:
#             metadata[key] = np.array(metadata[key])

#         indices = np.random.permutation(len(signals))
#         signals = signals[indices]
#         labels = labels[indices]
#         for key in metadata:
#             metadata[key] = metadata[key][indices]

#         return {"signals": signals, "labels": labels, "metadata": metadata}

#     def create_torch_dataset(
#         self,
#         signals: np.ndarray,
#         labels: np.ndarray,
#         metadata: Optional[Dict] = None,
#     ) -> SyntheticPACDataset:
#         """Create PyTorch dataset."""
#         signals_tensor = torch.tensor(signals, dtype=torch.float32)
#         labels_tensor = torch.tensor(labels, dtype=torch.long)

#         if signals_tensor.ndim == 2:
#             signals_tensor = signals_tensor.unsqueeze(1)

#         metadata_tensors = None
#         if metadata is not None:
#             metadata_tensors = {}
#             for key, value in metadata.items():
#                 if isinstance(value[0], (int, float, np.number)):
#                     metadata_tensors[key] = torch.tensor(
#                         value, dtype=torch.float32
#                     )
#                 else:
#                     metadata_tensors[key] = value

#         return SyntheticPACDataset(
#             signals_tensor, labels_tensor, metadata_tensors
#         )

#     @property
#     def info(self) -> Dict:
#         """Generator configuration and recommendations."""
#         return {
#             "config": {
#                 "fs": self.fs,
#                 "duration_sec": self.duration_sec,
#                 "n_samples": self.n_samples,
#             },
#             "limits": {
#                 "max_phase_freq": self.fs / 10,
#                 "max_amp_freq": self.fs / 3,
#                 "nyquist": self.fs / 2,
#             },
#             "recommended": {
#                 "phase_freq": [5, 20],
#                 "amp_freq": [30, 150],
#                 "coupling": [0.3, 0.7],
#                 "noise": [0.05, 0.2],
#             },
#         }


class SyntheticDataGenerator:
    """Generate synthetic Phase-Amplitude Coupling (PAC) signals."""

    def __init__(
        self,
        fs: float = 512.0,
        duration_sec: float = 2.0,
        random_seed: Optional[int] = 42,
    ):
        self.fs = fs
        self.duration_sec = duration_sec
        self.n_samples = int(fs * duration_sec)
        self.random_seed = random_seed
        if random_seed is not None:
            self._set_seeds(random_seed)

    def _set_seeds(self, seed: int):
        """Set all random seeds consistently."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def pac(
        self,
        phase_freq: float = 10.0,
        amp_freq: float = 80.0,
        coupling: float = 0.5,
        noise: float = 0.1,
    ) -> np.ndarray:
        """Generate single PAC signal (simplified interface)."""
        return self.generate_pac_signal(phase_freq, amp_freq, coupling, noise)

    def no_pac(
        self, freq1: float = 10.0, freq2: float = 80.0, noise: float = 0.1
    ) -> np.ndarray:
        """Generate control signal without PAC (simplified interface)."""
        return self.generate_no_pac_signal(freq1, freq2, noise)

    def dataset(
        self, n_samples: int = 100, balanced: bool = True
    ) -> SyntheticPACDataset:
        """Generate complete dataset and return as PyTorch dataset (one-step)."""
        if balanced:
            n_pac = n_samples // 2
            n_no_pac = n_samples - n_pac
            signals, labels = self.quick_dataset(n_pac, n_no_pac)
        else:
            data = self.generate_dataset(n_samples, include_no_pac=True)
            signals = data["signals"]
            labels = data["labels"]

        metadata = {
            "phase_freq": np.zeros(len(signals)),
            "amp_freq": np.zeros(len(signals)),
            "coupling_strength": np.zeros(len(signals)),
            "noise_level": np.full(len(signals), 0.1),
            "has_pac": labels.astype(bool),
        }

        return self.create_torch_dataset(signals, labels, metadata)

    def quick_dataset(
        self, n_pac: int = 50, n_no_pac: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dataset with specified counts, return arrays only."""
        if self.random_seed is not None:
            self._set_seeds(self.random_seed)

        all_signals = []
        all_labels = []

        phase_freqs = [8, 12, 16]
        amp_freqs = [60, 80, 100]
        couplings = [0.3, 0.5, 0.7]

        for idx in range(n_pac):
            pf = np.random.choice(phase_freqs)
            af = np.random.choice(amp_freqs)
            coupling = np.random.choice(couplings)
            signal = self.pac(pf, af, coupling)
            all_signals.append(signal)
            all_labels.append(1)

        for idx in range(n_no_pac):
            f1 = np.random.uniform(5, 20)
            f2 = np.random.uniform(50, 120)
            signal = self.no_pac(f1, f2)
            all_signals.append(signal)
            all_labels.append(0)

        signals = np.array(all_signals)
        labels = np.array(all_labels)

        indices = np.random.permutation(len(signals))
        return signals[indices], labels[indices]

    # def generate_pac_signal(
    #     self,
    #     phase_freq: float,
    #     amp_freq: float,
    #     coupling_strength: float = 0.5,
    #     noise_level: float = 0.1,
    #     coupling_type: str = "amplitude",
    # ) -> np.ndarray:
    #     """Generate PAC signal with detailed parameters."""
    #     t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
    #     phase_signal = np.sin(2 * np.pi * phase_freq * t)

    #     if coupling_type == "amplitude":
    #         modulation = 0.5 + 0.5 * coupling_strength * np.sin(
    #             2 * np.pi * phase_freq * t
    #         )
    #         carrier = np.sin(2 * np.pi * amp_freq * t)
    #         amp_signal = modulation * carrier
    #     elif coupling_type == "phase":
    #         phase_mod = coupling_strength * np.sin(2 * np.pi * phase_freq * t)
    #         amp_signal = np.sin(2 * np.pi * amp_freq * t + phase_mod)

    #     signal = phase_signal + amp_signal

    #     if noise_level > 0:
    #         noise = np.random.normal(0, noise_level, self.n_samples)
    #         signal += noise

    #     return signal

    # def generate_no_pac_signal(
    #     self,
    #     freq1: float = 10.0,
    #     freq2: float = 80.0,
    #     noise_level: float = 0.1,
    # ) -> np.ndarray:
    #     """Generate signal without PAC."""
    #     t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
    #     signal1 = np.sin(2 * np.pi * freq1 * t)
    #     signal2 = np.sin(2 * np.pi * freq2 * t)
    #     signal = signal1 + signal2

    #     if noise_level > 0:
    #         noise = np.random.normal(0, noise_level, self.n_samples)
    #         signal += noise

    #     return signal

    def generate_pac_signal(
        self,
        phase_freq: float,
        amp_freq: float,
        coupling_strength: float = 0.5,
        noise_level: float = 0.1,
        coupling_type: str = "amplitude",
    ) -> np.ndarray:
        """Generate PAC signal with detailed parameters."""
        if self.random_seed is not None:
            self._set_seeds(self.random_seed)

        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        phase_signal = np.sin(2 * np.pi * phase_freq * t)

        if coupling_type == "amplitude":
            modulation = 0.5 + 0.5 * coupling_strength * np.sin(
                2 * np.pi * phase_freq * t
            )
            carrier = np.sin(2 * np.pi * amp_freq * t)
            amp_signal = modulation * carrier
        elif coupling_type == "phase":
            phase_mod = coupling_strength * np.sin(2 * np.pi * phase_freq * t)
            amp_signal = np.sin(2 * np.pi * amp_freq * t + phase_mod)

        signal = phase_signal + amp_signal

        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise

        return signal

    def generate_no_pac_signal(
        self,
        freq1: float = 10.0,
        freq2: float = 80.0,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Generate signal without PAC."""
        if self.random_seed is not None:
            self._set_seeds(self.random_seed)

        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        signal1 = np.sin(2 * np.pi * freq1 * t)
        signal2 = np.sin(2 * np.pi * freq2 * t)
        signal = signal1 + signal2

        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise

        return signal

    def generate_multi_pac_signal(
        self,
        pac_pairs: List[Tuple[float, float, float]],
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Generate signal with multiple PAC pairs."""
        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        signal = np.zeros(self.n_samples)

        for phase_freq, amp_freq, coupling in pac_pairs:
            phase_signal = np.sin(2 * np.pi * phase_freq * t)
            modulation = 0.5 + 0.5 * coupling * np.sin(2 * np.pi * phase_freq * t)
            amp_signal = modulation * np.sin(2 * np.pi * amp_freq * t)
            signal += phase_signal + amp_signal

        signal = signal / len(pac_pairs)

        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise

        return signal

    def generate_dataset(
        self,
        n_samples: int = 100,
        pac_params: Optional[Dict] = None,
        include_no_pac: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate complete dataset."""
        if pac_params is None:
            pac_params = {
                "phase_freqs": [6, 10, 15],
                "amp_freqs": [60, 80, 100],
                "coupling_strengths": [0.3, 0.5, 0.7],
                "noise_levels": [0.05, 0.1, 0.2],
            }

        n_pac_combinations = (
            len(pac_params["phase_freqs"])
            * len(pac_params["amp_freqs"])
            * len(pac_params["coupling_strengths"])
        )

        if include_no_pac:
            n_per_condition = n_samples // (n_pac_combinations + 1)
            n_no_pac = n_samples - (n_per_condition * n_pac_combinations)
        else:
            n_per_condition = n_samples // n_pac_combinations
            n_no_pac = 0

        all_signals = []
        all_labels = []
        metadata = {
            "phase_freq": [],
            "amp_freq": [],
            "coupling_strength": [],
            "noise_level": [],
            "has_pac": [],
        }

        sample_idx = 0
        for phase_freq in pac_params["phase_freqs"]:
            for amp_freq in pac_params["amp_freqs"]:
                for coupling in pac_params["coupling_strengths"]:
                    for _ in range(n_per_condition):
                        if sample_idx >= n_samples:
                            break
                        noise = np.random.choice(pac_params["noise_levels"])
                        signal = self.generate_pac_signal(
                            phase_freq, amp_freq, coupling, noise
                        )
                        all_signals.append(signal)
                        all_labels.append(1)
                        metadata["phase_freq"].append(phase_freq)
                        metadata["amp_freq"].append(amp_freq)
                        metadata["coupling_strength"].append(coupling)
                        metadata["noise_level"].append(noise)
                        metadata["has_pac"].append(True)
                        sample_idx += 1

        if include_no_pac:
            for _ in range(n_no_pac):
                noise = np.random.choice(pac_params["noise_levels"])
                freq1 = np.random.uniform(5, 20)
                freq2 = np.random.uniform(50, 120)
                signal = self.generate_no_pac_signal(freq1, freq2, noise)
                all_signals.append(signal)
                all_labels.append(0)
                metadata["phase_freq"].append(freq1)
                metadata["amp_freq"].append(freq2)
                metadata["coupling_strength"].append(0.0)
                metadata["noise_level"].append(noise)
                metadata["has_pac"].append(False)

        signals = np.array(all_signals)
        labels = np.array(all_labels)
        for key in metadata:
            metadata[key] = np.array(metadata[key])

        indices = np.random.permutation(len(signals))
        signals = signals[indices]
        labels = labels[indices]
        for key in metadata:
            metadata[key] = metadata[key][indices]

        return {"signals": signals, "labels": labels, "metadata": metadata}

    def create_torch_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> SyntheticPACDataset:
        """Create PyTorch dataset."""
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if signals_tensor.ndim == 2:
            signals_tensor = signals_tensor.unsqueeze(1)

        metadata_tensors = None
        if metadata is not None:
            metadata_tensors = {}
            for key, value in metadata.items():
                if isinstance(value[0], (int, float, np.number)):
                    metadata_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    metadata_tensors[key] = value

        return SyntheticPACDataset(signals_tensor, labels_tensor, metadata_tensors)

    @property
    def info(self) -> Dict:
        """Generator configuration and recommendations."""
        return {
            "config": {
                "fs": self.fs,
                "duration_sec": self.duration_sec,
                "n_samples": self.n_samples,
            },
            "limits": {
                "max_phase_freq": self.fs / 10,
                "max_amp_freq": self.fs / 3,
                "nyquist": self.fs / 2,
            },
            "recommended": {
                "phase_freq": [5, 20],
                "amp_freq": [30, 150],
                "coupling": [0.3, 0.7],
                "noise": [0.05, 0.2],
            },
        }


# EOF
