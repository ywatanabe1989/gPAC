#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 14:30:00"
# Author: Claude
# Filename: diagnose_gpac_slowness.py

"""
Diagnose why gPAC is slower than TensorPAC
Break down timing into components
"""

import time
import numpy as np
import torch
import torch.profiler
from gpac import PAC as gPAC_PAC
from gpac import SyntheticDataGenerator
from tensorpac import Pac as TensorPAC_Pac
import warnings
warnings.filterwarnings('ignore')

def profile_gpac_components():
    """Profile different components of gPAC execution"""
    print("gPAC Performance Breakdown")
    print("=" * 80)
    
    # Config
    config = {
        'batch_size': 1,
        'n_channels': 64,
        'duration': 10,
        'fs': 512,
        'pha_n_bands': 30,
        'amp_n_bands': 30,
    }
    
    n_samples = int(config['duration'] * config['fs'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {config}")
    print()
    
    # Generate data
    generator = SyntheticDataGenerator(fs=config['fs'], duration_sec=config['duration'])
    batch_signals = []
    
    for b in range(config['batch_size']):
        signals = []
        for ch in range(config['n_channels']):
            signal = generator.generate_pac_signal(
                phase_freq=6.0,
                amp_freq=80.0,
                coupling_strength=0.8,
                noise_level=0.2
            )
            signals.append(signal)
        batch_signals.append(signals)
    
    batch_signals = np.array(batch_signals)
    print(f"Data shape: {batch_signals.shape}")
    print(f"Total samples: {batch_signals.size:,}")
    print()
    
    # Initialize model
    print("1. Model Initialization")
    init_start = time.time()
    pac_model = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=250,
        amp_n_bands=config['amp_n_bands'],
        trainable=False
    ).to(device)
    init_time = time.time() - init_start
    print(f"  Initialization time: {init_time:.3f}s")
    print()
    
    # Profile components
    print("2. Component Timing (5 runs each)")
    
    # Data transfer to GPU
    times = []
    for _ in range(5):
        start = time.time()
        data_torch = torch.from_numpy(batch_signals).float().to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f"  Data to GPU: {np.mean(times)*1000:.2f}±{np.std(times)*1000:.2f}ms")
    
    # Forward pass only
    data_torch = torch.from_numpy(batch_signals).float().to(device)
    times = []
    for _ in range(5):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            results = pac_model(data_torch)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f"  Forward pass: {np.mean(times):.3f}±{np.std(times):.3f}s")
    
    # Data transfer back to CPU
    times = []
    for _ in range(5):
        with torch.no_grad():
            results = pac_model(data_torch)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        pac_cpu = results["pac"].cpu().numpy()
        times.append(time.time() - start)
    print(f"  Data to CPU: {np.mean(times)*1000:.2f}±{np.std(times)*1000:.2f}ms")
    
    # Full pipeline
    times = []
    for _ in range(5):
        start = time.time()
        data_torch = torch.from_numpy(batch_signals).float().to(device)
        with torch.no_grad():
            results = pac_model(data_torch)
        pac_cpu = results["pac"].cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    print(f"  Full pipeline: {np.mean(times):.3f}±{np.std(times):.3f}s")
    print()
    
    # Test with different batch sizes
    print("3. Batch Size Scaling")
    for batch_size in [1, 2, 4, 8, 16]:
        # Generate data
        batch_signals = np.random.randn(batch_size, config['n_channels'], n_samples)
        
        times = []
        for _ in range(3):
            start = time.time()
            data_torch = torch.from_numpy(batch_signals).float().to(device)
            with torch.no_grad():
                results = pac_model(data_torch)
            pac_cpu = results["pac"].cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times)
        throughput = batch_size * config['n_channels'] * n_samples / mean_time
        print(f"  Batch {batch_size}: {mean_time:.3f}s ({throughput/1e6:.1f}M samples/s)")
    print()
    
    # Test with different number of bands
    print("4. Frequency Band Scaling")
    for n_bands in [10, 20, 30, 40, 50]:
        pac_model_bands = gPAC_PAC(
            seq_len=n_samples,
            fs=config['fs'],
            pha_start_hz=2,
            pha_end_hz=20,
            pha_n_bands=n_bands,
            amp_start_hz=30,
            amp_end_hz=250,
            amp_n_bands=n_bands,
            trainable=False
        ).to(device)
        
        batch_signals = np.random.randn(1, config['n_channels'], n_samples)
        
        times = []
        for _ in range(3):
            start = time.time()
            data_torch = torch.from_numpy(batch_signals).float().to(device)
            with torch.no_grad():
                results = pac_model_bands(data_torch)
            pac_cpu = results["pac"].cpu().numpy()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times)
        print(f"  {n_bands}x{n_bands} bands: {mean_time:.3f}s")
    print()
    
    # Compare with TensorPAC
    print("5. TensorPAC Comparison")
    pha_edges = np.linspace(2, 20, config['pha_n_bands'] + 1)
    amp_edges = np.linspace(30, 250, config['amp_n_bands'] + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    pac_tp = TensorPAC_Pac(
        idpac=(2, 0, 0),
        f_pha=pha_bands,
        f_amp=amp_bands,
        verbose=False
    )
    
    batch_signals = np.random.randn(config['batch_size'], config['n_channels'], n_samples)
    signals_reshaped = batch_signals.transpose(0, 2, 1).reshape(n_samples, -1)
    
    times = []
    for _ in range(3):
        start = time.time()
        pac_matrix = pac_tp.filterfit(config['fs'], signals_reshaped, n_jobs=64)
        times.append(time.time() - start)
    
    print(f"  TensorPAC (64 cores): {np.mean(times):.3f}±{np.std(times):.3f}s")
    
    # Single core comparison
    times = []
    for _ in range(3):
        start = time.time()
        pac_matrix = pac_tp.filterfit(config['fs'], signals_reshaped, n_jobs=1)
        times.append(time.time() - start)
    
    print(f"  TensorPAC (1 core): {np.mean(times):.3f}±{np.std(times):.3f}s")

if __name__ == "__main__":
    profile_gpac_components()