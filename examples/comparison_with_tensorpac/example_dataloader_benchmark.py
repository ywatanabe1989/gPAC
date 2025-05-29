#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-01-29 14:40:00"
# Author: Claude
# Filename: example_dataloader_benchmark.py

"""
Realistic benchmark assuming DataLoader usage (data already on GPU)
Separates timing for:
1. Core computation (GPU)
2. Result retrieval (GPU→CPU) - often necessary for analysis
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gpac import PAC as gPAC_PAC
from gpac import SyntheticDataGenerator
from tensorpac import Pac as TensorPAC_Pac
import warnings
warnings.filterwarnings('ignore')

def benchmark_with_dataloader():
    """Benchmark assuming DataLoader workflow"""
    print("DataLoader-Style gPAC vs TensorPAC Benchmark")
    print("=" * 80)
    
    # Config
    config = {
        'batch_size': 8,  # More realistic batch size
        'n_channels': 64,
        'duration': 10,
        'fs': 512,
        'pha_n_bands': 30,
        'amp_n_bands': 30,
        'n_jobs': 64,
    }
    
    n_samples = int(config['duration'] * config['fs'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {config}")
    print()
    
    # Generate dataset (normally would be loaded from files)
    print("Generating dataset...")
    generator = SyntheticDataGenerator(fs=config['fs'], duration_sec=config['duration'])
    all_signals = []
    
    for _ in range(32):  # 32 total samples
        batch_signals = []
        for ch in range(config['n_channels']):
            signal = generator.generate_pac_signal(
                phase_freq=6.0,
                amp_freq=80.0,
                coupling_strength=0.8,
                noise_level=0.2
            )
            batch_signals.append(signal)
        all_signals.append(batch_signals)
    
    all_signals = torch.FloatTensor(np.array(all_signals))
    dataset = TensorDataset(all_signals)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize models
    print("\nInitializing models...")
    nyquist = config['fs'] / 2
    amp_end_hz = min(150, nyquist - 10)
    
    # gPAC
    pac_gpac = gPAC_PAC(
        seq_len=n_samples,
        fs=config['fs'],
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=config['pha_n_bands'],
        amp_start_hz=30,
        amp_end_hz=amp_end_hz,
        amp_n_bands=config['amp_n_bands'],
        trainable=False
    ).to(device)
    
    # TensorPAC
    pha_edges = np.linspace(2, 20, config['pha_n_bands'] + 1)
    amp_edges = np.linspace(30, amp_end_hz, config['amp_n_bands'] + 1)
    pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
    amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
    
    pac_tp = TensorPAC_Pac(
        idpac=(2, 0, 0),
        f_pha=pha_bands,
        f_amp=amp_bands,
        verbose=False
    )
    
    print("\n" + "="*80)
    print("SCENARIO 1: DataLoader with GPU computation only (no CPU transfer)")
    print("="*80)
    
    # Warmup
    for batch in dataloader:
        data_gpu = batch[0].to(device)
        with torch.no_grad():
            _ = pac_gpac(data_gpu)
        break
    
    # Time GPU computation only
    gpu_times = []
    for batch in dataloader:
        data_gpu = batch[0].to(device)  # In practice, dataloader can put directly on GPU
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            results = pac_gpac(data_gpu)
        
        torch.cuda.synchronize()
        end = time.time()
        
        gpu_times.append(end - start)
    
    total_samples = len(all_signals) * config['n_channels'] * n_samples
    gpu_mean = np.mean(gpu_times)
    gpu_total = np.sum(gpu_times)
    
    print(f"\ngPAC (GPU compute only):")
    print(f"  Per batch: {gpu_mean:.4f}s")
    print(f"  Total time: {gpu_total:.4f}s")
    print(f"  Throughput: {total_samples/gpu_total/1e6:.2f}M samples/s")
    
    print("\n" + "="*80)
    print("SCENARIO 2: Full pipeline with CPU transfer (typical analysis)")
    print("="*80)
    
    # Time full pipeline (including CPU transfer)
    full_times = []
    cpu_transfer_times = []
    
    for batch in dataloader:
        data_gpu = batch[0].to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            results = pac_gpac(data_gpu)
        
        torch.cuda.synchronize()
        compute_end = time.time()
        
        # Transfer to CPU (necessary for analysis/saving)
        pac_cpu = results["pac"].cpu().numpy()
        
        end = time.time()
        
        full_times.append(end - start)
        cpu_transfer_times.append(end - compute_end)
    
    full_mean = np.mean(full_times)
    full_total = np.sum(full_times)
    transfer_mean = np.mean(cpu_transfer_times)
    
    print(f"\ngPAC (with CPU transfer):")
    print(f"  Per batch: {full_mean:.4f}s (compute: {gpu_mean:.4f}s + transfer: {transfer_mean:.4f}s)")
    print(f"  Total time: {full_total:.4f}s")
    print(f"  Throughput: {total_samples/full_total/1e6:.2f}M samples/s")
    print(f"  Transfer overhead: {transfer_mean/full_mean*100:.1f}%")
    
    print("\n" + "="*80)
    print("SCENARIO 3: TensorPAC baseline")
    print("="*80)
    
    # Time TensorPAC
    tp_times = []
    
    for batch in dataloader:
        data_np = batch[0].numpy()
        
        start = time.time()
        
        # Reshape for TensorPAC
        batch_size = data_np.shape[0]
        signals_reshaped = data_np.transpose(0, 2, 1).reshape(n_samples, -1)
        
        # Compute PAC
        pac_matrix = pac_tp.filterfit(
            config["fs"], 
            signals_reshaped,
            n_jobs=config['n_jobs']
        )
        
        end = time.time()
        tp_times.append(end - start)
    
    tp_mean = np.mean(tp_times)
    tp_total = np.sum(tp_times)
    
    print(f"\nTensorPAC ({config['n_jobs']} cores):")
    print(f"  Per batch: {tp_mean:.4f}s")
    print(f"  Total time: {tp_total:.4f}s")
    print(f"  Throughput: {total_samples/tp_total/1e6:.2f}M samples/s")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"gPAC vs TensorPAC speedup:")
    print(f"  GPU compute only: {tp_total/gpu_total:.2f}x")
    print(f"  With CPU transfer: {tp_total/full_total:.2f}x")
    print(f"\nGPU→CPU transfer adds {transfer_mean/gpu_mean*100:.1f}% overhead to gPAC")
    
    # Test different batch sizes
    print("\n" + "="*80)
    print("BATCH SIZE SCALING (GPU compute only)")
    print("="*80)
    
    for batch_size in [1, 2, 4, 8, 16, 32]:
        # Create batch
        batch_data = torch.randn(batch_size, config['n_channels'], n_samples).to(device)
        
        # Time computation
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = pac_gpac(batch_data)
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        mean_time = np.mean(times)
        samples = batch_size * config['n_channels'] * n_samples
        throughput = samples / mean_time / 1e6
        
        print(f"  Batch {batch_size:2d}: {mean_time:.4f}s ({throughput:.2f}M samples/s)")

if __name__ == "__main__":
    benchmark_with_dataloader()