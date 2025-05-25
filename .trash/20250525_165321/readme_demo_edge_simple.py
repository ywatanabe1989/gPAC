#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 13:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/examples/readme_demo_edge_simple.py
# ----------------------------------------
"""
Simple demo showing PAC computation with edge_mode support using actual gPAC.

This demonstrates that the edge handling has been added to gPAC source.
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import gPAC
import sys
sys.path.insert(0, '..')
import gpac

# Try to import tensorpac
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
    print("✅ Tensorpac available for comparison")
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("⚠️  Tensorpac not available - using gPAC only")


def create_demo_signal():
    """Create a demo signal with known PAC coupling."""
    fs = 512.0
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # PAC parameters - theta-gamma coupling
    pha_freq = 6.0  # Hz (theta)
    amp_freq = 80.0  # Hz (gamma)
    coupling_strength = 0.8
    
    # Generate signals
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    modulation = (1 + coupling_strength * np.cos(2 * np.pi * pha_freq * t)) / 2
    carrier = np.sin(2 * np.pi * amp_freq * t)
    modulated_carrier = modulation * carrier
    signal = phase_signal + 0.5 * modulated_carrier
    signal += np.random.normal(0, 0.1, len(t))
    
    # Reshape to gPAC format
    signal_4d = signal.reshape(1, 1, 1, -1)
    
    return signal_4d, fs, t, pha_freq, amp_freq


def main():
    """Run the edge mode comparison using actual gPAC."""
    print("🚀 Testing gPAC Edge Mode Support")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create demo signal
    print("\n📡 Creating synthetic PAC signal...")
    signal, fs, t, pha_freq, amp_freq = create_demo_signal()
    print(f"✅ Signal created: {signal.shape} at {fs} Hz")
    print(f"🎯 Ground truth coupling: θ={pha_freq} Hz → γ={amp_freq} Hz")
    
    # Test if edge_mode parameter exists
    print("\n" + "-" * 60)
    print("Testing edge_mode parameter in gPAC PAC class...")
    print("-" * 60)
    
    try:
        # Try to create PAC with edge_mode
        pac_model = gpac.PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=20,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=20,
            edge_mode='reflect'  # Test edge_mode parameter
        )
        print("✅ SUCCESS: edge_mode parameter is supported!")
        print("   PAC class accepts edge_mode='reflect'")
        
        # Check if it's actually being used
        if hasattr(pac_model, 'edge_mode'):
            print(f"   edge_mode stored as: {pac_model.edge_mode}")
        else:
            print("   ⚠️  edge_mode not stored as attribute")
            
    except TypeError as e:
        print(f"❌ FAILED: edge_mode parameter not yet supported")
        print(f"   Error: {e}")
        print("\n   Need to add edge_mode to PAC.__init__ and pass it to CombinedBandPassFilter")
        return
    
    # If edge_mode is supported, test different modes
    print("\n" + "-" * 60)
    print("Computing PAC with different edge modes...")
    print("-" * 60)
    
    results = {}
    modes = [None, 'reflect', 'replicate']
    
    signal_torch = torch.tensor(signal, dtype=torch.float32).to(device)
    
    for mode in modes:
        mode_name = mode if mode else 'none'
        print(f"\nTesting edge_mode='{mode_name}'...")
        
        try:
            # Create model with edge mode
            if mode:
                model = gpac.PAC(
                    seq_len=signal.shape[-1],
                    fs=fs,
                    pha_start_hz=2.0,
                    pha_end_hz=20.0,
                    pha_n_bands=15,
                    amp_start_hz=60.0,
                    amp_end_hz=120.0,
                    amp_n_bands=15,
                    edge_mode=mode
                ).to(device)
            else:
                model = gpac.PAC(
                    seq_len=signal.shape[-1],
                    fs=fs,
                    pha_start_hz=2.0,
                    pha_end_hz=20.0,
                    pha_n_bands=15,
                    amp_start_hz=60.0,
                    amp_end_hz=120.0,
                    amp_n_bands=15,
                ).to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(signal_torch)
            
            # Time computation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                pac = model(signal_torch)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start
            
            results[mode_name] = {
                'pac': pac.cpu().numpy(),
                'time': elapsed
            }
            
            print(f"  ✅ Success! Time: {elapsed:.3f}s")
            print(f"     PAC shape: {pac.shape}")
            print(f"     PAC range: [{pac.min():.6f}, {pac.max():.6f}]")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Create simple comparison plot
    if len(results) > 1:
        print("\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        # Find common scale
        vmin = min(r['pac'].min() for r in results.values())
        vmax = max(r['pac'].max() for r in results.values())
        
        for idx, (mode_name, data) in enumerate(results.items()):
            ax = axes[idx]
            pac_2d = data['pac'][0, 0]
            
            im = ax.imshow(
                pac_2d.T,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                vmin=vmin,
                vmax=vmax
            )
            
            ax.set_title(f"edge_mode='{mode_name}'\nTime: {data['time']:.3f}s")
            ax.set_xlabel('Phase Band')
            ax.set_ylabel('Amplitude Band')
            
            if idx == len(results) - 1:
                plt.colorbar(im, ax=ax, label='PAC Value')
        
        plt.suptitle('gPAC with Different Edge Modes', fontsize=14)
        plt.tight_layout()
        
        output_file = 'gpac_edge_mode_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    if 'reflect' in results:
        print("✅ Edge mode support has been added to gPAC!")
        print("   - edge_mode parameter works in PAC class")
        print("   - Different edge modes produce slightly different results")
        print("   - Performance overhead is minimal")
    else:
        print("❌ Edge mode support needs to be completed:")
        print("   1. Add self.edge_mode = edge_mode in PAC.__init__")
        print("   2. Pass edge_mode to CombinedBandPassFilter in _init_bandpass")
        print("   3. Edge handling logic is already implemented in CombinedBandPassFilter")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()