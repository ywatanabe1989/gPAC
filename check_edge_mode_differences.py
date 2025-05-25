#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-25 14:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/check_edge_mode_differences.py
# ----------------------------------------
"""
Check if edge modes actually produce different results and measure timing correctly.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Import gPAC
import sys
sys.path.insert(0, '.')
import gpac


def create_test_signal():
    """Create a test signal."""
    fs = 512.0
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create signal with PAC
    pha_freq = 6.0
    amp_freq = 80.0
    phase_signal = np.sin(2 * np.pi * pha_freq * t)
    modulation = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
    carrier = np.sin(2 * np.pi * amp_freq * t)
    signal = phase_signal + 0.5 * modulation * carrier
    signal += np.random.normal(0, 0.1, len(t))
    
    return signal.reshape(1, 1, 1, -1), fs


def measure_timing_correctly():
    """Measure timing with proper initialization separation."""
    print("=" * 80)
    print("TIMING ANALYSIS WITH INITIALIZATION SEPARATION")
    print("=" * 80)
    
    signal, fs = create_test_signal()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signal_torch = torch.tensor(signal, dtype=torch.float32).to(device)
    
    # Test configurations
    configs = [
        ("Standard", False, None),
        ("Edge reflect", False, 'reflect'),
        ("Filtfilt + edge reflect", True, 'reflect'),
    ]
    
    results = {}
    
    for name, filtfilt, edge_mode in configs:
        print(f"\n{name}:")
        
        # Measure initialization time
        init_start = time.time()
        model = gpac.PAC(
            seq_len=signal.shape[-1],
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=20,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=20,
            filtfilt_mode=filtfilt,
            edge_mode=edge_mode
        ).to(device)
        init_time = time.time() - init_start
        
        # Warm up (important for GPU)
        with torch.no_grad():
            _ = model(signal_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Measure computation time (average of 10 runs)
        n_runs = 10
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        comp_start = time.time()
        
        for _ in range(n_runs):
            with torch.no_grad():
                pac = model(signal_torch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        comp_time = (time.time() - comp_start) / n_runs
        
        results[name] = {
            'init_time': init_time,
            'comp_time': comp_time,
            'total_time': init_time + comp_time,
            'pac': pac.cpu().numpy()
        }
        
        print(f"  Initialization: {init_time:.4f}s")
        print(f"  Computation: {comp_time:.4f}s (avg of {n_runs} runs)")
        print(f"  Total (first run): {init_time + comp_time:.4f}s")
    
    return results


def check_differences():
    """Check if different edge modes produce different results."""
    print("\n" + "=" * 80)
    print("CHECKING DIFFERENCES BETWEEN EDGE MODES")
    print("=" * 80)
    
    signal, fs = create_test_signal()
    
    # Calculate PAC with different modes
    modes = {
        'None': None,
        'reflect': 'reflect',
        'replicate': 'replicate',
        'circular': 'circular'
    }
    
    results = {}
    
    for mode_name, edge_mode in modes.items():
        pac, _, _ = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=20.0,
            pha_n_bands=15,
            amp_start_hz=60.0,
            amp_end_hz=120.0,
            amp_n_bands=15,
            edge_mode=edge_mode
        )
        results[mode_name] = pac.cpu().numpy()
    
    # Compare results
    print("\nPairwise comparisons:")
    print("-" * 60)
    
    mode_names = list(modes.keys())
    for i in range(len(mode_names)):
        for j in range(i+1, len(mode_names)):
            mode1, mode2 = mode_names[i], mode_names[j]
            pac1, pac2 = results[mode1], results[mode2]
            
            # Calculate differences
            abs_diff = np.abs(pac1 - pac2)
            rel_diff = abs_diff / (np.abs(pac1) + 1e-10)
            
            max_abs_diff = abs_diff.max()
            mean_abs_diff = abs_diff.mean()
            max_rel_diff = rel_diff.max()
            
            print(f"\n{mode1} vs {mode2}:")
            print(f"  Max absolute difference: {max_abs_diff:.6e}")
            print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
            print(f"  Max relative difference: {max_rel_diff:.3%}")
            
            if max_abs_diff < 1e-10:
                print("  ⚠️  IDENTICAL - edge mode might not be working!")
            elif max_abs_diff < 1e-6:
                print("  ✅ Nearly identical (numerical precision)")
            else:
                print("  ✅ Different results - edge mode is working")
    
    # Visualize differences
    print("\n📊 Creating difference visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot PAC for each mode
    for idx, (mode_name, pac) in enumerate(results.items()):
        if idx < 4:
            ax = axes[idx]
            im = ax.imshow(pac[0, 0], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"edge_mode='{mode_name}'")
            ax.set_xlabel('Amplitude Band')
            ax.set_ylabel('Phase Band')
            plt.colorbar(im, ax=ax)
    
    # Plot differences
    ax = axes[4]
    diff_none_reflect = results['None'] - results['reflect']
    im = ax.imshow(diff_none_reflect[0, 0], aspect='auto', origin='lower', 
                   cmap='RdBu_r', vmin=-diff_none_reflect.max(), vmax=diff_none_reflect.max())
    ax.set_title("Difference: None - reflect")
    ax.set_xlabel('Amplitude Band')
    ax.set_ylabel('Phase Band')
    plt.colorbar(im, ax=ax, label='Difference')
    
    ax = axes[5]
    diff_reflect_replicate = results['reflect'] - results['replicate']
    im = ax.imshow(diff_reflect_replicate[0, 0], aspect='auto', origin='lower',
                   cmap='RdBu_r', vmin=-diff_reflect_replicate.max(), vmax=diff_reflect_replicate.max())
    ax.set_title("Difference: reflect - replicate")
    ax.set_xlabel('Amplitude Band')
    ax.set_ylabel('Phase Band')
    plt.colorbar(im, ax=ax, label='Difference')
    
    plt.suptitle('Edge Mode Comparison and Differences', fontsize=16)
    plt.tight_layout()
    plt.savefig('edge_mode_differences.png', dpi=150, bbox_inches='tight')
    print("💾 Saved to: edge_mode_differences.png")
    
    return results


def check_edge_effects_on_edges():
    """Check if edge effects are visible at signal boundaries."""
    print("\n" + "=" * 80)
    print("CHECKING EDGE EFFECTS AT SIGNAL BOUNDARIES")
    print("=" * 80)
    
    # Create a signal with strong discontinuity at edges
    fs = 512.0
    duration = 2.0  # Shorter signal to emphasize edges
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create signal with strong edge discontinuity
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)
    # Add strong values at edges
    signal[:50] = 2.0  # Strong positive at start
    signal[-50:] = -2.0  # Strong negative at end
    
    signal_4d = signal.reshape(1, 1, 1, -1)
    
    # Test with and without edge mode
    print("\nTesting with signal that has strong edge discontinuities...")
    
    pac_no_edge, _, _ = gpac.calculate_pac(
        signal_4d, fs=fs,
        pha_start_hz=5.0, pha_end_hz=15.0, pha_n_bands=5,
        amp_start_hz=70.0, amp_end_hz=90.0, amp_n_bands=5,
        edge_mode=None
    )
    
    pac_with_edge, _, _ = gpac.calculate_pac(
        signal_4d, fs=fs,
        pha_start_hz=5.0, pha_end_hz=15.0, pha_n_bands=5,
        amp_start_hz=70.0, amp_end_hz=90.0, amp_n_bands=5,
        edge_mode='reflect'
    )
    
    diff = pac_no_edge - pac_with_edge
    max_diff = torch.abs(diff).max().item()
    mean_diff = torch.abs(diff).mean().item()
    
    print(f"\nWith edge discontinuities:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    if max_diff > 1e-4:
        print("  ✅ Significant differences with edge discontinuities - edge mode is working!")
    else:
        print("  ⚠️  No significant differences even with edge discontinuities")
    
    return pac_no_edge, pac_with_edge


def main():
    """Run all checks."""
    print("🔍 INVESTIGATING EDGE MODE FUNCTIONALITY")
    print("=" * 80)
    
    # 1. Check timing with proper initialization separation
    timing_results = measure_timing_correctly()
    
    # 2. Check if different edge modes produce different results
    edge_differences = check_differences()
    
    # 3. Check edge effects with discontinuous signal
    pac_no_edge, pac_with_edge = check_edge_effects_on_edges()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n1. TIMING ANALYSIS:")
    for name, data in timing_results.items():
        print(f"   {name}: init={data['init_time']:.4f}s, compute={data['comp_time']:.4f}s")
    
    print("\n2. EDGE MODE FUNCTIONALITY:")
    # Check if any differences were found
    max_diff = 0
    for mode1 in edge_differences:
        for mode2 in edge_differences:
            if mode1 != mode2:
                diff = np.abs(edge_differences[mode1] - edge_differences[mode2]).max()
                max_diff = max(max_diff, diff)
    
    if max_diff < 1e-10:
        print("   ❌ Edge modes produce IDENTICAL results - not working properly!")
    else:
        print(f"   ✅ Edge modes produce different results (max diff: {max_diff:.6e})")
    
    print("\n3. RECOMMENDATIONS:")
    print("   - The first timing was misleading due to initialization overhead")
    print("   - Actual computation times are very similar across modes")
    print("   - Edge mode differences may be subtle with smooth signals")
    print("   - Stronger differences expected with discontinuous signals or longer filters")


if __name__ == "__main__":
    main()