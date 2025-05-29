#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-28 21:30:00 (ywatanabe)"
# File: ./src/gpac/_SyntheticDataGenerator.py

"""
Synthetic PAC data generator for testing and benchmarking.

This module provides tools to generate synthetic signals with known
phase-amplitude coupling properties for testing PAC algorithms.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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


class SyntheticDataGenerator:
    """
    Generate synthetic Phase-Amplitude Coupling (PAC) signals.
    
    This generator creates signals with controlled PAC properties for
    testing and validation of PAC analysis methods.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    duration_sec : float
        Duration of each signal in seconds
    random_seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        fs: float = 256.0,
        duration_sec: float = 2.0,
        random_seed: Optional[int] = 42,
    ):
        """Initialize the data generator."""
        self.fs = fs
        self.duration_sec = duration_sec
        self.n_samples = int(fs * duration_sec)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def generate_pac_signal(
        self,
        phase_freq: float,
        amp_freq: float,
        coupling_strength: float = 0.5,
        noise_level: float = 0.1,
        coupling_type: str = "amplitude"
    ) -> np.ndarray:
        """
        Generate a single PAC signal with specified coupling.
        
        Parameters
        ----------
        phase_freq : float
            Frequency of the phase signal (Hz)
        amp_freq : float
            Frequency of the amplitude signal (Hz)
        coupling_strength : float
            Strength of coupling (0-1)
        noise_level : float
            Standard deviation of additive Gaussian noise
        coupling_type : str
            Type of coupling: "amplitude" or "phase"
            
        Returns
        -------
        signal : np.ndarray
            Generated signal with PAC
        """
        # Create time vector
        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        
        # Generate phase signal (low frequency)
        phase_signal = np.sin(2 * np.pi * phase_freq * t)
        
        if coupling_type == "amplitude":
            # Amplitude modulation by phase
            # Modulation varies between (1-coupling_strength) and 1
            modulation = 0.5 + 0.5 * coupling_strength * np.sin(2 * np.pi * phase_freq * t)
            
            # High frequency carrier
            carrier = np.sin(2 * np.pi * amp_freq * t)
            
            # Apply amplitude modulation
            amp_signal = modulation * carrier
            
        elif coupling_type == "phase":
            # Phase modulation (less common but possible)
            phase_mod = coupling_strength * np.sin(2 * np.pi * phase_freq * t)
            amp_signal = np.sin(2 * np.pi * amp_freq * t + phase_mod)
            
        else:
            raise ValueError(f"Unknown coupling type: {coupling_type}")
        
        # Combine signals
        signal = phase_signal + amp_signal
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise
            
        return signal

    def generate_no_pac_signal(
        self,
        freq1: float = 10.0,
        freq2: float = 80.0,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Generate a signal with no PAC (control signal).
        
        Parameters
        ----------
        freq1 : float
            First frequency component
        freq2 : float
            Second frequency component
        noise_level : float
            Standard deviation of additive noise
            
        Returns
        -------
        signal : np.ndarray
            Signal without PAC
        """
        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        
        # Independent oscillations
        signal1 = np.sin(2 * np.pi * freq1 * t)
        signal2 = np.sin(2 * np.pi * freq2 * t)
        
        # Combine without coupling
        signal = signal1 + signal2
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise
            
        return signal

    def generate_multi_pac_signal(
        self,
        pac_pairs: List[Tuple[float, float, float]],
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Generate signal with multiple PAC pairs.
        
        Parameters
        ----------
        pac_pairs : list of tuples
            Each tuple contains (phase_freq, amp_freq, coupling_strength)
        noise_level : float
            Noise level
            
        Returns
        -------
        signal : np.ndarray
            Signal with multiple PAC components
        """
        t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        signal = np.zeros(self.n_samples)
        
        for phase_freq, amp_freq, coupling in pac_pairs:
            # Phase component
            phase_signal = np.sin(2 * np.pi * phase_freq * t)
            
            # Modulated amplitude component
            modulation = 0.5 + 0.5 * coupling * np.sin(2 * np.pi * phase_freq * t)
            amp_signal = modulation * np.sin(2 * np.pi * amp_freq * t)
            
            # Add to signal
            signal += phase_signal + amp_signal
            
        # Normalize
        signal = signal / len(pac_pairs)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, self.n_samples)
            signal += noise
            
        return signal

    def generate_dataset(
        self,
        n_samples: int = 100,
        pac_params: Optional[Dict] = None,
        include_no_pac: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate a dataset of PAC signals.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        pac_params : dict, optional
            Parameters for PAC generation. If None, uses defaults.
        include_no_pac : bool
            Whether to include no-PAC control signals
            
        Returns
        -------
        dataset : dict
            Dictionary containing 'signals', 'labels', and 'metadata'
        """
        if pac_params is None:
            # Default PAC parameters
            pac_params = {
                'phase_freqs': [6, 10, 15],
                'amp_freqs': [60, 80, 100],
                'coupling_strengths': [0.3, 0.5, 0.7],
                'noise_levels': [0.05, 0.1, 0.2]
            }
            
        # Calculate total samples
        n_pac_combinations = (
            len(pac_params['phase_freqs']) * 
            len(pac_params['amp_freqs']) * 
            len(pac_params['coupling_strengths'])
        )
        
        if include_no_pac:
            n_per_condition = n_samples // (n_pac_combinations + 1)
            n_no_pac = n_samples - (n_per_condition * n_pac_combinations)
        else:
            n_per_condition = n_samples // n_pac_combinations
            n_no_pac = 0
            
        # Initialize storage
        all_signals = []
        all_labels = []
        metadata = {
            'phase_freq': [],
            'amp_freq': [],
            'coupling_strength': [],
            'noise_level': [],
            'has_pac': []
        }
        
        # Generate PAC signals
        sample_idx = 0
        for phase_freq in pac_params['phase_freqs']:
            for amp_freq in pac_params['amp_freqs']:
                for coupling in pac_params['coupling_strengths']:
                    for _ in range(n_per_condition):
                        if sample_idx >= n_samples:
                            break
                            
                        # Random noise level
                        noise = np.random.choice(pac_params['noise_levels'])
                        
                        # Generate signal
                        signal = self.generate_pac_signal(
                            phase_freq=phase_freq,
                            amp_freq=amp_freq,
                            coupling_strength=coupling,
                            noise_level=noise
                        )
                        
                        all_signals.append(signal)
                        all_labels.append(1)  # Has PAC
                        
                        # Store metadata
                        metadata['phase_freq'].append(phase_freq)
                        metadata['amp_freq'].append(amp_freq)
                        metadata['coupling_strength'].append(coupling)
                        metadata['noise_level'].append(noise)
                        metadata['has_pac'].append(True)
                        
                        sample_idx += 1
                        
        # Generate no-PAC signals
        if include_no_pac:
            for _ in range(n_no_pac):
                noise = np.random.choice(pac_params['noise_levels'])
                
                # Random frequencies
                freq1 = np.random.uniform(5, 20)
                freq2 = np.random.uniform(50, 120)
                
                signal = self.generate_no_pac_signal(
                    freq1=freq1,
                    freq2=freq2,
                    noise_level=noise
                )
                
                all_signals.append(signal)
                all_labels.append(0)  # No PAC
                
                metadata['phase_freq'].append(freq1)
                metadata['amp_freq'].append(freq2)
                metadata['coupling_strength'].append(0.0)
                metadata['noise_level'].append(noise)
                metadata['has_pac'].append(False)
                
        # Convert to arrays
        signals = np.array(all_signals)
        labels = np.array(all_labels)
        
        # Convert metadata
        for key in metadata:
            metadata[key] = np.array(metadata[key])
            
        # Shuffle
        indices = np.random.permutation(len(signals))
        signals = signals[indices]
        labels = labels[indices]
        for key in metadata:
            metadata[key] = metadata[key][indices]
            
        return {
            'signals': signals,
            'labels': labels,
            'metadata': metadata
        }

    def create_torch_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> SyntheticPACDataset:
        """
        Create a PyTorch dataset from numpy arrays.
        
        Parameters
        ----------
        signals : np.ndarray
            Signal data
        labels : np.ndarray
            Labels
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        dataset : SyntheticPACDataset
            PyTorch dataset
        """
        # Convert to tensors
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Add channel and batch dimensions if needed
        if signals_tensor.ndim == 1:
            signals_tensor = signals_tensor.unsqueeze(0).unsqueeze(0)
        elif signals_tensor.ndim == 2:
            # (n_samples, n_timepoints) -> (n_samples, 1, n_timepoints)
            signals_tensor = signals_tensor.unsqueeze(1)
            
        # Convert metadata
        if metadata is not None:
            metadata_tensors = {}
            for key, value in metadata.items():
                if isinstance(value[0], (int, float, np.number)):
                    metadata_tensors[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    metadata_tensors[key] = value
        else:
            metadata_tensors = None
            
        return SyntheticPACDataset(
            signals=signals_tensor,
            labels=labels_tensor,
            metadata=metadata_tensors
        )


# Convenience function for quick signal generation
def generate_pac_signal(
    duration: float = 2.0,
    fs: float = 256.0,
    phase_freq: float = 10.0,
    amp_freq: float = 80.0,
    coupling_strength: float = 0.5,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Quick function to generate a single PAC signal.
    
    Parameters
    ----------
    duration : float
        Signal duration in seconds
    fs : float
        Sampling frequency
    phase_freq : float
        Phase frequency in Hz
    amp_freq : float
        Amplitude frequency in Hz
    coupling_strength : float
        Coupling strength (0-1)
    noise_level : float
        Noise level
    random_seed : int, optional
        Random seed
        
    Returns
    -------
    signal : np.ndarray
        Generated PAC signal
    """
    generator = SyntheticDataGenerator(fs=fs, duration_sec=duration, random_seed=random_seed)
    return generator.generate_pac_signal(
        phase_freq=phase_freq,
        amp_freq=amp_freq,
        coupling_strength=coupling_strength,
        noise_level=noise_level
    )