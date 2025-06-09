#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:03:21 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticPACDataset.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/dataset/test__SyntheticPACDataset.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
import torch
from gpac.dataset._SyntheticPACDataset import SyntheticPACDataset
from torch.utils.data import DataLoader

class TestSyntheticPACDataset:
    """Test PyTorch dataset for synthetic PAC signals."""

    def setup_method(self):
        """Set up test data."""
        self.n_samples = 100
        self.n_channels = 1
        self.seq_len = 512
        self.signals = torch.randn(
            self.n_samples, self.n_channels, self.seq_len
        )
        self.labels = torch.randint(0, 2, (self.n_samples,))
        self.metadata = {
            "phase_freq": torch.rand(self.n_samples) * 20,
            "amp_freq": torch.rand(self.n_samples) * 100 + 50,
            "coupling_strength": torch.rand(self.n_samples),
            "has_pac": self.labels.bool(),
        }

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = SyntheticPACDataset(self.signals, self.labels, self.metadata)
        assert len(dataset) == self.n_samples
        assert dataset.signals.shape == self.signals.shape
        assert dataset.labels.shape == self.labels.shape
        assert dataset.metadata is not None

    def test_getitem_with_metadata(self):
        """Test getting items with metadata."""
        dataset = SyntheticPACDataset(self.signals, self.labels, self.metadata)
        signal, label, meta = dataset[0]

        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(meta, dict)
        assert signal.shape == (self.n_channels, self.seq_len)
        assert label.shape == ()
        assert "phase_freq" in meta
        assert "amp_freq" in meta
        assert "coupling_strength" in meta
        assert "has_pac" in meta

    def test_getitem_without_metadata(self):
        """Test getting items without metadata."""
        dataset = SyntheticPACDataset(self.signals, self.labels, metadata=None)
        result = dataset[0]

        assert len(result) == 2
        signal, label = result
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert signal.shape == (self.n_channels, self.seq_len)

    def test_len(self):
        """Test dataset length."""
        dataset = SyntheticPACDataset(self.signals, self.labels)
        assert len(dataset) == self.n_samples

    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader."""
        dataset = SyntheticPACDataset(self.signals, self.labels, self.metadata)
        batch_size = 16
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        batch = next(iter(dataloader))

        assert len(batch) == 3
        signals_batch, labels_batch, meta_batch = batch
        assert signals_batch.shape == (
            batch_size,
            self.n_channels,
            self.seq_len,
        )
        assert labels_batch.shape == (batch_size,)
        assert isinstance(meta_batch, dict)
        for key in self.metadata.keys():
            assert key in meta_batch
            assert meta_batch[key].shape == (batch_size,)

    def test_indexing(self):
        """Test various indexing operations."""
        dataset = SyntheticPACDataset(self.signals, self.labels, self.metadata)
        last_item = dataset[-1]
        assert len(last_item) == 3

        with pytest.raises(IndexError):
            _ = dataset[self.n_samples]
        with pytest.raises(IndexError):
            _ = dataset[-self.n_samples - 1]

    def test_iteration(self):
        """Test iterating through dataset."""
        dataset = SyntheticPACDataset(self.signals, self.labels, self.metadata)
        count = 0
        for item in dataset:
            count += 1
            assert len(item) == 3
        assert count == len(dataset)

    def test_different_data_types(self):
        """Test dataset with different data types."""
        signals_float64 = self.signals.double()
        labels_int64 = self.labels.long()
        dataset = SyntheticPACDataset(signals_float64, labels_int64)
        signal, label = dataset[0]
        assert signal.dtype == torch.float64
        assert label.dtype == torch.int64

    def test_metadata_types(self):
        """Test dataset with different metadata types."""
        metadata = {
            "numeric": torch.rand(self.n_samples),
            "boolean": torch.rand(self.n_samples) > 0.5,
            "integer": torch.randint(0, 10, (self.n_samples,)),
        }
        dataset = SyntheticPACDataset(self.signals, self.labels, metadata)
        _, _, meta = dataset[0]

        assert isinstance(meta["numeric"], torch.Tensor)
        assert isinstance(meta["boolean"], torch.Tensor)
        assert isinstance(meta["integer"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
