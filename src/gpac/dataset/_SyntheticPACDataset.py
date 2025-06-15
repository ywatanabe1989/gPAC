#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:20:42 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/dataset/_SyntheticPACDataset.py
# ----------------------------------------
import os

__FILE__ = "./src/gpac/dataset/_SyntheticPACDataset.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from torch.utils.data import Dataset


class SyntheticPACDataset(Dataset):
    """
    PyTorch Dataset for synthetic PAC signals.

    This dataset can be used across different experiments consistently.
    """

    def __init__(self, signals, labels, metadata=None):
        """Initialize the dataset with signals and labels."""
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


# EOF
