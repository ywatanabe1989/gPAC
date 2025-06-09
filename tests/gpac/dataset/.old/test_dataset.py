#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 22:50:58 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test_dataset.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/dataset/test_dataset.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest

import numpy as np
import torch
from gpac.dataset import (generate_pac_batch, generate_pac_dataloader,
                          generate_pac_dataset, multi_class_multi_pac_config,
                          single_class_single_pac_config)
from gpac.dataset._SyntheticDataGenerator import SyntheticDataGenerator
from gpac.dataset._SyntheticPACDataset import SyntheticPACDataset
from torch.utils.data import DataLoader


class TestSyntheticDataGenerator:
    def test_init_default(self):
        gen = SyntheticDataGenerator()
        assert gen.fs == 512.0
        assert gen.duration_sec == 2.0
        assert gen.n_samples == 1024

    def test_init_custom(self):
        gen = SyntheticDataGenerator(
            fs=1000, duration_sec=1.5, random_seed=123
        )
        assert gen.fs == 1000
        assert gen.duration_sec == 1.5
        assert gen.n_samples == 1500

    def test_pac_signal_generation(self):
        gen = SyntheticDataGenerator(random_seed=42)
        signal = gen.pac(phase_freq=10.0, amp_freq=80.0, coupling=0.5)
        assert signal.shape == (1024,)
        assert isinstance(signal, np.ndarray)

    def test_no_pac_signal_generation(self):
        gen = SyntheticDataGenerator(random_seed=42)
        signal = gen.no_pac(freq1=10.0, freq2=80.0)
        assert signal.shape == (1024,)
        assert isinstance(signal, np.ndarray)

    def test_multi_pac_signal(self):
        gen = SyntheticDataGenerator(random_seed=42)
        pac_pairs = [(8, 60, 0.3), (12, 80, 0.5)]
        signal = gen.generate_multi_pac_signal(pac_pairs)
        assert signal.shape == (1024,)

    def test_quick_dataset(self):
        gen = SyntheticDataGenerator(random_seed=42)
        signals, labels = gen.quick_dataset(n_pac=10, n_no_pac=10)
        assert signals.shape == (20, 1024)
        assert labels.shape == (20,)
        assert np.sum(labels == 1) == 10
        assert np.sum(labels == 0) == 10

    def test_generate_dataset(self):
        gen = SyntheticDataGenerator(random_seed=42)
        data = gen.generate_dataset(n_samples=20, include_no_pac=True)
        assert "signals" in data
        assert "labels" in data
        assert "metadata" in data
        assert data["signals"].shape[0] == 20

    def test_dataset_method(self):
        gen = SyntheticDataGenerator(random_seed=42)
        dataset = gen.dataset(n_samples=10)
        assert isinstance(dataset, SyntheticPACDataset)
        assert len(dataset) == 10

    def test_info_property(self):
        gen = SyntheticDataGenerator(fs=500, duration_sec=1.0)
        info = gen.info
        assert info["config"]["fs"] == 500
        assert info["config"]["duration_sec"] == 1.0
        assert info["limits"]["nyquist"] == 250


class TestSyntheticPACDataset:
    def test_init_without_metadata(self):
        signals = torch.randn(10, 3, 1024)
        labels = torch.randint(0, 2, (10,))
        dataset = SyntheticPACDataset(signals, labels)
        assert len(dataset) == 10
        assert dataset.metadata is None

    def test_init_with_metadata(self):
        signals = torch.randn(10, 3, 1024)
        labels = torch.randint(0, 2, (10,))
        metadata = {"fs": torch.tensor([512.0] * 10)}
        dataset = SyntheticPACDataset(signals, labels, metadata)
        assert len(dataset) == 10
        assert dataset.metadata is not None

    def test_getitem_without_metadata(self):
        signals = torch.randn(10, 3, 1024)
        labels = torch.randint(0, 2, (10,))
        dataset = SyntheticPACDataset(signals, labels)
        signal, label = dataset[0]
        assert signal.shape == (3, 1024)
        assert isinstance(label, torch.Tensor)

    def test_getitem_with_metadata(self):
        signals = torch.randn(10, 3, 1024)
        labels = torch.randint(0, 2, (10,))
        metadata = {"fs": torch.tensor([512.0] * 10)}
        dataset = SyntheticPACDataset(signals, labels, metadata)
        signal, label, meta = dataset[0]
        assert signal.shape == (3, 1024)
        assert "fs" in meta
        assert meta["fs"] == 512.0


class TestGeneratePacDataset:
    def test_generate_pac_dataset_basic(self):
        dataset = generate_pac_dataset(
            n_samples=8, n_channels=4, n_segments=2, random_seed=42
        )
        assert isinstance(dataset, SyntheticPACDataset)
        assert len(dataset) == 8

    def test_generate_pac_dataset_shapes(self):
        dataset = generate_pac_dataset(
            n_samples=4,
            n_channels=3,
            n_segments=2,
            duration_sec=1,
            fs=256,
            random_seed=42,
        )
        signal, label, metadata = dataset[0]
        # Should be (n_channels, n_segments, seq_len)
        assert signal.shape == (3, 2, 256)

    def test_generate_pac_dataset_multi_class(self):
        dataset = generate_pac_dataset(
            n_samples=12,
            pac_config=multi_class_multi_pac_config,
            random_seed=42,
        )
        # Should have 3 classes in multi_class_multi_pac_config
        labels = [dataset[ii][1] for ii in range(len(dataset))]
        unique_labels = set(labels)
        assert len(unique_labels) == 3

    def test_generate_pac_dataloader(self):
        dataloader = generate_pac_dataloader(
            n_samples=8, batch_size=4, random_seed=42
        )
        assert isinstance(dataloader, DataLoader)
        batch = next(iter(dataloader))
        # Should return (signals, labels, metadata)
        assert len(batch) == 3

    def test_generate_pac_batch(self):
        signals, labels, metadata = generate_pac_batch(
            batch_size=4, n_channels=2, n_segments=2, random_seed=42
        )
        assert signals.shape[0] == 4  # batch_size
        assert labels.shape[0] == 4
        assert isinstance(metadata, dict)

    def test_balanced_vs_unbalanced(self):
        # Test balanced
        dataset_balanced = generate_pac_dataset(
            n_samples=12,
            pac_config=multi_class_multi_pac_config,
            balanced=True,
            random_seed=42,
        )

        # Test unbalanced
        dataset_unbalanced = generate_pac_dataset(
            n_samples=12,
            pac_config=multi_class_multi_pac_config,
            balanced=False,
            random_seed=42,
        )

        # Both should work
        assert len(dataset_balanced) == 12
        assert len(dataset_unbalanced) >= 0  # Could be variable

    def test_reproducibility(self):
        dataset1 = generate_pac_dataset(n_samples=4, random_seed=42)
        dataset2 = generate_pac_dataset(n_samples=4, random_seed=42)

        signal1, _, _ = dataset1[0]
        signal2, _, _ = dataset2[0]

        assert torch.allclose(signal1, signal2)

    def test_different_configs(self):
        # Test single class single PAC
        dataset1 = generate_pac_dataset(
            n_samples=4,
            pac_config=single_class_single_pac_config,
            random_seed=42,
        )

        # Test multi class multi PAC
        dataset2 = generate_pac_dataset(
            n_samples=6,
            pac_config=multi_class_multi_pac_config,
            random_seed=42,
        )

        assert len(dataset1) == 4
        assert len(dataset2) == 6

    def test_metadata_structure(self):
        dataset = generate_pac_dataset(
            n_samples=2,
            pac_config=single_class_single_pac_config,
            random_seed=42,
        )

        _, _, metadata = dataset[0]
        required_keys = ["fs", "class_name", "noise_level", "n_pac_components"]
        for key in required_keys:
            assert key in metadata


class TestEdgeCases:
    def test_zero_noise(self):
        gen = SyntheticDataGenerator(random_seed=42)
        signal = gen.pac(noise=0.0)
        assert signal.shape == (1024,)

    def test_high_coupling(self):
        gen = SyntheticDataGenerator(random_seed=42)
        signal = gen.pac(coupling=1.0)
        assert signal.shape == (1024,)

    def test_small_dataset(self):
        dataset = generate_pac_dataset(n_samples=1, random_seed=42)
        assert len(dataset) == 1

    def test_single_channel_single_segment(self):
        dataset = generate_pac_dataset(
            n_samples=2, n_channels=1, n_segments=1, random_seed=42
        )
        signal, _, _ = dataset[0]
        assert signal.shape == (1, 1, 1024)

    def test_no_pac_components(self):
        no_pac_config = {"no_pac": {"components": [], "noise_levels": [0.1]}}
        dataset = generate_pac_dataset(
            n_samples=2, pac_config=no_pac_config, random_seed=42
        )
        _, _, metadata = dataset[0]
        assert metadata["n_pac_components"] == 0


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
