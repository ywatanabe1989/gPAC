#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-07 23:07:03 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/tests/gpac/dataset/test__SyntheticDataGenerator.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/gpac/dataset/test__SyntheticDataGenerator.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pytest
import torch
from gpac.dataset._SyntheticDataGenerator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    def setup_method(self):
        """Set up test generator."""
        self.fs = 512.0
        self.duration = 2.0
        self.gen = SyntheticDataGenerator(
            fs=self.fs, duration_sec=self.duration, random_seed=42
        )

    def test_initialization(self):
        """Test generator initialization."""
        assert self.gen.fs == self.fs
        assert self.gen.duration_sec == self.duration
        assert self.gen.n_samples == int(self.fs * self.duration)

    def test_pac_signal_generation(self):
        """Test PAC signal generation."""
        phase_freq = 10.0
        amp_freq = 80.0
        coupling = 0.5
        noise = 0.1
        signal = self.gen.pac(phase_freq, amp_freq, coupling, noise)

        assert signal.shape == (self.gen.n_samples,)
        assert isinstance(signal, np.ndarray)
        assert signal.std() > 0

    def test_no_pac_signal_generation(self):
        """Test no-PAC signal generation."""
        freq1 = 10.0
        freq2 = 80.0
        noise = 0.1
        signal = self.gen.no_pac(freq1, freq2, noise)

        assert signal.shape == (self.gen.n_samples,)
        assert isinstance(signal, np.ndarray)
        assert signal.std() > 0

    def test_quick_dataset_generation(self):
        """Test quick dataset generation."""
        n_pac = 50
        n_no_pac = 50
        signals, labels = self.gen.quick_dataset(n_pac, n_no_pac)

        assert signals.shape == (n_pac + n_no_pac, self.gen.n_samples)
        assert labels.shape == (n_pac + n_no_pac,)
        assert np.sum(labels == 1) == n_pac
        assert np.sum(labels == 0) == n_no_pac

    def test_dataset_generation_balanced(self):
        """Test balanced dataset generation."""
        n_samples = 100
        dataset = self.gen.dataset(n_samples, balanced=True)

        assert len(dataset) == n_samples

        labels = []
        for ii in range(len(dataset)):
            sample = dataset[ii]
            if len(sample) == 2:
                _, label = sample
            else:
                _, label, _ = sample
            labels.append(label.item())
        labels = np.array(labels)

        n_pac = np.sum(labels == 1)
        n_no_pac = np.sum(labels == 0)
        assert abs(n_pac - n_no_pac) < n_samples * 0.5

    def test_multi_pac_signal(self):
        """Test multi-PAC signal generation."""
        pac_pairs = [
            (8.0, 60.0, 0.5),
            (12.0, 100.0, 0.3),
        ]
        noise = 0.1
        signal = self.gen.generate_multi_pac_signal(pac_pairs, noise)

        assert signal.shape == (self.gen.n_samples,)
        assert isinstance(signal, np.ndarray)
        assert signal.std() > 0

    def test_detailed_pac_generation(self):
        """Test detailed PAC generation with different coupling types."""
        phase_freq = 10.0
        amp_freq = 80.0
        coupling = 0.5
        noise = 0.1

        signal_amp = self.gen.generate_pac_signal(
            phase_freq, amp_freq, coupling, noise, coupling_type="amplitude"
        )
        assert signal_amp.shape == (self.gen.n_samples,)

        signal_phase = self.gen.generate_pac_signal(
            phase_freq, amp_freq, coupling, noise, coupling_type="phase"
        )
        assert signal_phase.shape == (self.gen.n_samples,)
        assert not np.allclose(signal_amp, signal_phase)

    def test_generator_info(self):
        """Test generator info property."""
        info = self.gen.info
        assert "config" in info
        assert "limits" in info
        assert "recommended" in info
        assert info["config"]["fs"] == self.fs
        assert info["config"]["duration_sec"] == self.duration
        assert info["limits"]["nyquist"] == self.fs / 2
        assert info["limits"]["max_phase_freq"] == self.fs / 10
        assert info["limits"]["max_amp_freq"] == self.fs / 3

    def test_reproducibility(self):
        """Test random seed reproducibility."""
        gen1 = SyntheticDataGenerator(fs=512, duration_sec=2.0, random_seed=42)
        gen2 = SyntheticDataGenerator(fs=512, duration_sec=2.0, random_seed=42)

        signal1 = gen1.pac(10.0, 80.0, 0.5, 0.1)
        signal2 = gen2.pac(10.0, 80.0, 0.5, 0.1)
        assert np.allclose(signal1, signal2)

        gen3 = SyntheticDataGenerator(
            fs=512, duration_sec=2.0, random_seed=123
        )
        signal3 = gen3.pac(10.0, 80.0, 0.5, 0.1)
        assert not np.allclose(signal1, signal3)

    def test_generate_dataset_with_custom_params(self):
        """Test dataset generation with custom parameters."""
        pac_params = {
            "phase_freqs": [5, 10],
            "amp_freqs": [50, 100],
            "coupling_strengths": [0.3, 0.7],
            "noise_levels": [0.1, 0.2],
        }
        data = self.gen.generate_dataset(
            n_samples=50, pac_params=pac_params, include_no_pac=True
        )
        assert "signals" in data
        assert "labels" in data
        assert "metadata" in data
        assert "phase_freq" in data["metadata"]
        assert "amp_freq" in data["metadata"]
        assert "coupling_strength" in data["metadata"]
        assert "noise_level" in data["metadata"]
        assert "has_pac" in data["metadata"]

    def test_create_torch_dataset(self):
        """Test PyTorch dataset creation."""
        signals = np.random.randn(10, 1024)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        metadata = {
            "phase_freq": np.array([10.0] * 10),
            "amp_freq": np.array([80.0] * 10),
        }
        dataset = self.gen.create_torch_dataset(signals, labels, metadata)

        assert len(dataset) == 10
        signal, label, meta = dataset[0]
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(meta, dict)
        assert signal.shape == (1, 1024)
        assert label.item() == 0
        assert "phase_freq" in meta
        assert "amp_freq" in meta


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
