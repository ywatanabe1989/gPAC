#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 16:07:35 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__generate_pac_dataset.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/dataset/test__generate_pac_dataset.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
import torch
from gpac.dataset import (generate_pac_batch, generate_pac_dataloader,
                          generate_pac_dataset, multi_class_multi_pac_config,
                          multi_class_single_pac_config,
                          single_class_multi_pac_config,
                          single_class_single_pac_config)
from torch.utils.data import DataLoader


class TestGeneratePACDataset:
    """Test PAC dataset generation functions."""

    def test_generate_pac_dataset_basic(self):
        """Test basic dataset generation."""
        n_samples = 16
        n_channels = 8
        n_segments = 4
        duration_sec = 2
        fs = 512.0
        dataset = generate_pac_dataset(
            n_samples=n_samples,
            n_channels=n_channels,
            n_segments=n_segments,
            duration_sec=duration_sec,
            fs=fs,
            balanced=True,
            random_seed=42,
        )

        assert len(dataset) == n_samples
        signal, label, metadata = dataset[0]
        expected_seq_len = int(fs * duration_sec)
        assert signal.shape == (n_channels, n_segments, expected_seq_len)
        assert isinstance(label, torch.Tensor)
        assert isinstance(metadata, dict)

    def test_generate_pac_dataset_single_class(self):
        """Test dataset generation with single class config."""
        dataset = generate_pac_dataset(
            n_samples=20,
            pac_config=single_class_single_pac_config,
            random_seed=42,
        )

        labels = []
        for ii in range(len(dataset)):
            _, label, _ = dataset[ii]
            labels.append(label.item())

        assert all(label == 0 for label in labels)

    def test_generate_pac_dataset_multi_class(self):
        """Test dataset generation with multi-class config."""
        dataset = generate_pac_dataset(
            n_samples=30,
            pac_config=multi_class_single_pac_config,
            balanced=True,
            random_seed=42,
        )

        labels = []
        for ii in range(len(dataset)):
            _, label, _ = dataset[ii]
            labels.append(label.item())

        unique_labels = set(labels)
        assert len(unique_labels) == len(multi_class_single_pac_config)

    def test_generate_pac_dataloader(self):
        """Test DataLoader generation."""
        batch_size = 8
        n_samples = 32
        dataloader = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            random_seed=42,
        )

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == batch_size

        batch = next(iter(dataloader))
        assert len(batch) == 3
        signals_batch, labels_batch, meta_batch = batch
        assert signals_batch.shape[0] == batch_size
        assert labels_batch.shape[0] == batch_size

    def test_generate_pac_batch(self):
        """Test single batch generation."""
        batch_size = 16
        n_channels = 8
        n_segments = 4
        duration_sec = 2
        fs = 512.0
        signals, labels, metadata = generate_pac_batch(
            batch_size=batch_size,
            n_channels=n_channels,
            n_segments=n_segments,
            duration_sec=duration_sec,
            fs=fs,
            random_seed=42,
        )

        expected_seq_len = int(fs * duration_sec)
        assert signals.shape == (
            batch_size,
            n_channels,
            n_segments,
            expected_seq_len,
        )
        assert labels.shape == (batch_size,)
        assert isinstance(metadata, dict)

    def test_metadata_structure(self):
        """Test metadata structure in generated dataset."""
        dataset = generate_pac_dataset(
            n_samples=10,
            pac_config=multi_class_multi_pac_config,
            random_seed=42,
        )

        _, _, metadata = dataset[0]
        assert "fs" in metadata
        assert "class_name" in metadata
        assert "noise_level" in metadata
        assert "n_pac_components" in metadata
        assert "pac_components" in metadata

    def test_balanced_vs_unbalanced(self):
        """Test balanced vs unbalanced dataset generation."""
        n_samples = 90
        config = multi_class_single_pac_config

        balanced_dataset = generate_pac_dataset(
            n_samples=n_samples,
            pac_config=config,
            balanced=True,
            random_seed=42,
        )

        unbalanced_dataset = generate_pac_dataset(
            n_samples=n_samples,
            pac_config=config,
            balanced=False,
            random_seed=42,
        )

        balanced_labels = []
        for ii in range(len(balanced_dataset)):
            _, label, _ = balanced_dataset[ii]
            balanced_labels.append(label.item())

        unbalanced_labels = []
        for ii in range(len(unbalanced_dataset)):
            _, label, _ = unbalanced_dataset[ii]
            unbalanced_labels.append(label.item())

        balanced_counts = [balanced_labels.count(ii) for ii in range(3)]
        assert max(balanced_counts) - min(balanced_counts) <= 3

        unbalanced_counts = [unbalanced_labels.count(ii) for ii in range(3)]
        assert sum(unbalanced_counts) == len(unbalanced_dataset)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with random seed."""
        dataset1 = generate_pac_dataset(n_samples=10, random_seed=42)
        dataset2 = generate_pac_dataset(n_samples=10, random_seed=42)

        signal1, label1, _ = dataset1[0]
        signal2, label2, _ = dataset2[0]
        assert torch.allclose(signal1, signal2)
        assert label1 == label2

        dataset3 = generate_pac_dataset(n_samples=10, random_seed=123)
        signal3, label3, _ = dataset3[0]

        if label1 == label3:
            assert not torch.allclose(signal1, signal3)

    def test_multi_pac_components(self):
        """Test generation with multiple PAC components."""
        dataset = generate_pac_dataset(
            n_samples=10,
            pac_config=single_class_multi_pac_config,
            random_seed=42,
        )

        _, _, metadata = dataset[0]
        assert metadata["n_pac_components"].item() > 1
        assert len(metadata["pac_components"]) > 1

    def test_no_pac_class(self):
        """Test generation of no-PAC signals."""
        dataset = generate_pac_dataset(
            n_samples=10,
            pac_config={"no_pac": {"components": [], "noise_levels": [0.1]}},
            random_seed=42,
        )

        _, label, metadata = dataset[0]
        assert metadata["n_pac_components"].item() == 0
        assert len(metadata["pac_components"]) == 0

    def test_batch_metadata_handling(self):
        """Test metadata handling in batch generation."""
        batch_size = 6
        signals, labels, metadata = generate_pac_batch(
            batch_size=batch_size,
            pac_config=multi_class_multi_pac_config,
            random_seed=42,
        )

        assert "fs" in metadata
        assert "class_name" in metadata
        assert "noise_level" in metadata
        assert "n_pac_components" in metadata
        assert "pac_components" in metadata
        assert len(metadata["class_name"]) == batch_size
        assert metadata["fs"].shape[0] == batch_size
        assert isinstance(metadata["pac_components"], list)
        assert len(metadata["pac_components"]) == batch_size

    def test_dataloader_reproducible_shuffling(self):
        """Test DataLoader shuffling reproducibility with seed."""
        batch_size = 8
        n_samples = 32

        # Create two dataloaders with same seed
        dataloader1 = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=True,
            random_seed=42,
        )

        dataloader2 = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=True,
            random_seed=42,
        )

        # Get first batches from both dataloaders
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        signals1, labels1, _ = batch1
        signals2, labels2, _ = batch2

        # Should be identical due to same seed
        assert torch.allclose(signals1, signals2)
        assert torch.equal(labels1, labels2)

        # Create dataloader with different seed
        dataloader3 = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=True,
            random_seed=123,
        )

        batch3 = next(iter(dataloader3))
        signals3, labels3, _ = batch3

        # Should be different due to different seed
        assert not torch.allclose(signals1, signals3) or not torch.equal(
            labels1, labels3
        )

        # Test without shuffle - should be deterministic regardless of seed
        dataloader_no_shuffle1 = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=False,
            random_seed=42,
        )

        dataloader_no_shuffle2 = generate_pac_dataloader(
            n_samples=n_samples,
            batch_size=batch_size,
            shuffle=False,
            random_seed=123,
        )

        batch_ns1 = next(iter(dataloader_no_shuffle1))
        batch_ns2 = next(iter(dataloader_no_shuffle2))

        signals_ns1, labels_ns1, _ = batch_ns1
        signals_ns2, labels_ns2, _ = batch_ns2

        # Should be identical since no shuffling
        assert torch.equal(labels_ns1, labels_ns2)


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
