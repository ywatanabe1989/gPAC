#!/usr/bin/env python3
"""
TensorPAC vs gPAC Comparison Script

This script implements comprehensive comparison between gPAC and TensorPAC 
implementations to validate numerical accuracy and compatibility.

Target configuration: idpac=(2,0,0) - Modulation Index, No surrogates, No normalization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import warnings
import time
from typing import Dict, Any, Tuple

# Import both implementations
import gpac
try:
    # Add tensorpac source to path
    tensorpac_path = os.path.join(os.path.dirname(__file__), '../../tensorpac_source')
    if tensorpac_path not in sys.path:
        sys.path.insert(0, tensorpac_path)
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError as e:
    TENSORPAC_AVAILABLE = False
    print(f"Warning: TensorPAC not available: {e}")


def generate_test_signals(fs: float = 1000.0, duration: float = 2.0, 
                         n_trials: int = 10) -> Dict[str, Any]:
    """Generate synthetic test signals with known PAC characteristics."""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    signals = []
    params = []
    
    for trial in range(n_trials):
        # Parameters for this trial
        pha_freq = 5.0 + np.random.normal(0, 0.5)  # Around 5 Hz
        amp_freq = 80.0 + np.random.normal(0, 5.0)  # Around 80 Hz
        coupling_strength = 0.5 + np.random.uniform(0, 0.5)
        noise_level = 0.1 + np.random.uniform(0, 0.3)
        
        # Generate phase signal (low frequency)
        phase_sig = np.sin(2 * np.pi * pha_freq * t)
        
        # Generate amplitude signal (high frequency) modulated by phase
        phase_modulation = 1 + coupling_strength * np.sin(2 * np.pi * pha_freq * t)
        amp_sig = phase_modulation * np.sin(2 * np.pi * amp_freq * t)
        
        # Combined signal with noise
        signal = phase_sig + amp_sig + noise_level * np.random.randn(n_samples)
        
        signals.append(signal)
        params.append({
            'pha_freq': pha_freq,
            'amp_freq': amp_freq,
            'coupling_strength': coupling_strength,
            'noise_level': noise_level
        })
    
    # Shape: (n_trials, n_samples)
    signals_array = np.array(signals)
    
    return {
        'signals': signals_array,
        'params': params,
        'fs': fs,
        'duration': duration,
        't': t
    }


def run_gpac_analysis(signals: np.ndarray, fs: float, 
                     pha_freqs: tuple = (2, 20), amp_freqs: tuple = (60, 160),
                     pha_n_bands: int = 10, amp_n_bands: int = 10) -> Dict[str, Any]:
    """Run gPAC analysis on test signals."""
    
    print("Running gPAC analysis...")
    start_time = time.time()
    
    # Convert to gPAC expected format: (batch, channels, segments, time)
    # We'll treat each trial as a separate batch item with 1 channel and 1 segment
    signals_4d = signals[:, np.newaxis, np.newaxis, :]  # (n_trials, 1, 1, n_samples)
    
    try:
        pac_values, pha_mids, amp_mids = gpac.calculate_pac(
            signals_4d,
            fs=fs,
            pha_start_hz=pha_freqs[0],
            pha_end_hz=pha_freqs[1], 
            pha_n_bands=pha_n_bands,
            amp_start_hz=amp_freqs[0],
            amp_end_hz=amp_freqs[1],
            amp_n_bands=amp_n_bands,
            n_perm=None,  # No permutation testing to match idpac=(2,0,0)
            trainable=False
        )
        
        runtime = time.time() - start_time
        
        return {
            'pac_values': pac_values.cpu().numpy() if torch.is_tensor(pac_values) else pac_values,
            'pha_freqs': pha_mids,
            'amp_freqs': amp_mids,
            'runtime': runtime,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'pac_values': None,
            'pha_freqs': None,
            'amp_freqs': None,
            'runtime': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def run_tensorpac_analysis(signals: np.ndarray, fs: float,
                          pha_freqs: tuple = (2, 20), amp_freqs: tuple = (60, 160),
                          pha_n_bands: int = 10, amp_n_bands: int = 10) -> Dict[str, Any]:
    """Run TensorPAC analysis on test signals."""
    
    if not TENSORPAC_AVAILABLE:
        return {
            'pac_values': None,
            'pha_freqs': None,
            'amp_freqs': None,
            'runtime': 0,
            'success': False,
            'error': 'TensorPAC not available'
        }
    
    print("Running TensorPAC analysis...")
    start_time = time.time()
    
    try:
        # Create frequency vectors that match gPAC's approach
        pha_vec = np.linspace(pha_freqs[0], pha_freqs[1], pha_n_bands)
        amp_vec = np.linspace(amp_freqs[0], amp_freqs[1], amp_n_bands)
        
        # Convert to TensorPAC frequency band format
        pha_bands = [[pha_vec[i], pha_vec[i+1]] for i in range(len(pha_vec)-1)]
        amp_bands = [[amp_vec[i], amp_vec[i+1]] for i in range(len(amp_vec)-1)]
        
        # If we need exact number of bands, use linspace directly
        if len(pha_bands) != pha_n_bands:
            pha_bands = [[pha_freqs[0] + i * (pha_freqs[1] - pha_freqs[0]) / pha_n_bands,
                         pha_freqs[0] + (i+1) * (pha_freqs[1] - pha_freqs[0]) / pha_n_bands] 
                        for i in range(pha_n_bands)]
        
        if len(amp_bands) != amp_n_bands:
            amp_bands = [[amp_freqs[0] + i * (amp_freqs[1] - amp_freqs[0]) / amp_n_bands,
                         amp_freqs[0] + (i+1) * (amp_freqs[1] - amp_freqs[0]) / amp_n_bands]
                        for i in range(amp_n_bands)]
        
        # Initialize TensorPAC with idpac=(2,0,0)
        # Method 2: Modulation Index, No surrogates, No normalization
        pac_obj = Pac(
            idpac=(2, 0, 0),
            f_pha=pha_bands,
            f_amp=amp_bands,
            dcomplex='hilbert'
        )
        
        # TensorPAC expects (n_epochs, n_times) format
        pac_values = pac_obj.filterfit(fs, signals)
        
        runtime = time.time() - start_time
        
        # Get frequency centers for comparison
        pha_mids = np.array([np.mean(band) for band in pha_bands])
        amp_mids = np.array([np.mean(band) for band in amp_bands])
        
        return {
            'pac_values': pac_values,
            'pha_freqs': pha_mids,
            'amp_freqs': amp_mids,
            'runtime': runtime,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'pac_values': None,
            'pha_freqs': None,
            'amp_freqs': None,
            'runtime': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def compare_results(gpac_result: Dict[str, Any], tensorpac_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results between gPAC and TensorPAC."""
    
    comparison = {
        'both_successful': gpac_result['success'] and tensorpac_result['success'],
        'runtime_comparison': None,
        'frequency_agreement': None,
        'pac_agreement': None,
        'correlation': None,
        'max_absolute_error': None,
        'mean_relative_error': None,
        'summary': None
    }
    
    if not comparison['both_successful']:
        comparison['summary'] = f"Analysis failed - gPAC: {gpac_result.get('error', 'OK')}, TensorPAC: {tensorpac_result.get('error', 'OK')}"
        return comparison
    
    # Runtime comparison
    gpac_time = gpac_result['runtime']
    tensorpac_time = tensorpac_result['runtime']
    comparison['runtime_comparison'] = {
        'gpac_time': gpac_time,
        'tensorpac_time': tensorpac_time,
        'speedup_factor': tensorpac_time / gpac_time if gpac_time > 0 else float('inf')
    }
    
    # Frequency agreement
    pha_freq_diff = np.abs(gpac_result['pha_freqs'] - tensorpac_result['pha_freqs'])
    amp_freq_diff = np.abs(gpac_result['amp_freqs'] - tensorpac_result['amp_freqs'])
    
    comparison['frequency_agreement'] = {
        'pha_freq_max_diff': np.max(pha_freq_diff),
        'amp_freq_max_diff': np.max(amp_freq_diff),
        'pha_freq_mean_diff': np.mean(pha_freq_diff),
        'amp_freq_mean_diff': np.mean(amp_freq_diff)
    }
    
    # PAC values comparison
    gpac_pac = gpac_result['pac_values']
    tensorpac_pac = tensorpac_result['pac_values']
    
    # Handle different output shapes - gPAC: (batch, channels, pha, amp), TensorPAC: (trials, pha, amp)
    if gpac_pac.ndim == 4 and gpac_pac.shape[1] == 1:
        gpac_pac = gpac_pac[:, 0, :, :]  # Remove channel dimension
    
    # Ensure same shape
    if gpac_pac.shape != tensorpac_pac.shape:
        min_trials = min(gpac_pac.shape[0], tensorpac_pac.shape[0])
        min_pha = min(gpac_pac.shape[1], tensorpac_pac.shape[1])
        min_amp = min(gpac_pac.shape[2], tensorpac_pac.shape[2])
        
        gpac_pac = gpac_pac[:min_trials, :min_pha, :min_amp]
        tensorpac_pac = tensorpac_pac[:min_trials, :min_pha, :min_amp]
    
    # Calculate comparison metrics
    correlation = np.corrcoef(gpac_pac.flatten(), tensorpac_pac.flatten())[0, 1]
    absolute_errors = np.abs(gpac_pac - tensorpac_pac)
    max_absolute_error = np.max(absolute_errors)
    
    # Relative error (avoid division by zero)
    nonzero_mask = np.abs(tensorpac_pac) > 1e-10
    relative_errors = np.zeros_like(absolute_errors)
    relative_errors[nonzero_mask] = absolute_errors[nonzero_mask] / np.abs(tensorpac_pac[nonzero_mask])
    mean_relative_error = np.mean(relative_errors[nonzero_mask]) if np.any(nonzero_mask) else float('inf')
    
    comparison.update({
        'correlation': correlation,
        'max_absolute_error': max_absolute_error,
        'mean_relative_error': mean_relative_error,
        'pac_agreement': {
            'correlation': correlation,
            'agreement_99pct': correlation > 0.99,
            'max_abs_error': max_absolute_error,
            'mean_rel_error': mean_relative_error
        }
    })
    
    # Summary
    if correlation > 0.99:
        comparison['summary'] = f"✅ Excellent agreement (r={correlation:.4f})"
    elif correlation > 0.95:
        comparison['summary'] = f"✅ Good agreement (r={correlation:.4f})"
    elif correlation > 0.80:
        comparison['summary'] = f"⚠️ Moderate agreement (r={correlation:.4f})"
    else:
        comparison['summary'] = f"❌ Poor agreement (r={correlation:.4f})"
    
    return comparison


def main():
    """Run comprehensive TensorPAC vs gPAC comparison."""
    
    print("="*60)
    print("TensorPAC vs gPAC Comparison")
    print("="*60)
    print(f"TensorPAC available: {TENSORPAC_AVAILABLE}")
    print()
    
    # Generate test data
    print("Generating test signals...")
    test_data = generate_test_signals(fs=1000.0, duration=2.0, n_trials=5)
    print(f"Generated {test_data['signals'].shape[0]} trials of {test_data['duration']}s each")
    print()
    
    # Configuration
    pha_freqs = (4, 16)  # Theta/alpha range
    amp_freqs = (60, 120)  # Gamma range  
    pha_n_bands = 8
    amp_n_bands = 8
    
    print(f"Analysis configuration:")
    print(f"  Phase frequencies: {pha_freqs[0]}-{pha_freqs[1]} Hz ({pha_n_bands} bands)")
    print(f"  Amplitude frequencies: {amp_freqs[0]}-{amp_freqs[1]} Hz ({amp_n_bands} bands)")
    print(f"  Target idpac: (2, 0, 0) - Modulation Index, No surrogates, No normalization")
    print()
    
    # Run analyses
    gpac_result = run_gpac_analysis(
        test_data['signals'], test_data['fs'],
        pha_freqs, amp_freqs, pha_n_bands, amp_n_bands
    )
    
    tensorpac_result = run_tensorpac_analysis(
        test_data['signals'], test_data['fs'], 
        pha_freqs, amp_freqs, pha_n_bands, amp_n_bands
    )
    
    # Compare results
    print("Comparing results...")
    comparison = compare_results(gpac_result, tensorpac_result)
    
    # Report results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Overall: {comparison['summary']}")
    print()
    
    if comparison['both_successful']:
        # Runtime comparison
        runtime_comp = comparison['runtime_comparison']
        print(f"Runtime Comparison:")
        print(f"  gPAC: {runtime_comp['gpac_time']:.3f}s")
        print(f"  TensorPAC: {runtime_comp['tensorpac_time']:.3f}s") 
        print(f"  Speedup factor: {runtime_comp['speedup_factor']:.2f}x")
        print()
        
        # Frequency agreement
        freq_comp = comparison['frequency_agreement']
        print(f"Frequency Agreement:")
        print(f"  Phase freq max diff: {freq_comp['pha_freq_max_diff']:.4f} Hz")
        print(f"  Amp freq max diff: {freq_comp['amp_freq_max_diff']:.4f} Hz")
        print()
        
        # PAC agreement
        pac_comp = comparison['pac_agreement']
        print(f"PAC Value Agreement:")
        print(f"  Correlation: {pac_comp['correlation']:.6f}")
        print(f"  Max absolute error: {pac_comp['max_abs_error']:.6f}")
        print(f"  Mean relative error: {pac_comp['mean_rel_error']:.6f}")
        print(f"  99% agreement: {'✅ YES' if pac_comp['agreement_99pct'] else '❌ NO'}")
        print()
        
        # Success criteria evaluation
        print("Success Criteria Evaluation:")
        criteria_met = 0
        total_criteria = 3
        
        if pac_comp['agreement_99pct']:
            print("  ✅ ≥99% numerical agreement: PASSED")
            criteria_met += 1
        else:
            print(f"  ❌ ≥99% numerical agreement: FAILED (r={pac_comp['correlation']:.4f})")
        
        if runtime_comp['speedup_factor'] >= 0.9:  # Within 10% performance
            print("  ✅ Performance within 10%: PASSED")
            criteria_met += 1
        else:
            print(f"  ❌ Performance within 10%: FAILED ({runtime_comp['speedup_factor']:.2f}x)")
        
        if freq_comp['pha_freq_max_diff'] < 0.1 and freq_comp['amp_freq_max_diff'] < 1.0:
            print("  ✅ Frequency matching: PASSED")
            criteria_met += 1
        else:
            print("  ❌ Frequency matching: FAILED")
        
        print(f"\nOverall: {criteria_met}/{total_criteria} criteria met")
        
        if criteria_met == total_criteria:
            print("🎉 ALL SUCCESS CRITERIA MET!")
        else:
            print("⚠️ Some criteria not met - further investigation needed")
    
    else:
        print("❌ Analysis failed, cannot perform comparison")
        if not gpac_result['success']:
            print(f"gPAC error: {gpac_result['error']}")
        if not tensorpac_result['success']:
            print(f"TensorPAC error: {tensorpac_result['error']}")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()