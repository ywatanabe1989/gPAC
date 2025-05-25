#!/usr/bin/env python3
"""
Comprehensive Performance Test Suite for gPAC v1.0.0

Tests:
1. Various signal lengths (short to very long)
2. Different batch sizes
3. Memory usage patterns
4. Edge cases and error handling
5. CPU vs GPU performance comparison
"""

import time
import gc
import warnings
from typing import Dict, List, Tuple
import psutil
import torch
import numpy as np
import pytest
import gpac


class PerformanceMetrics:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.compute_times = []
        self.memory_usage = []
        self.gpu_memory = []
        self.throughput = []
        self.errors = []
    
    def add_computation(self, compute_time: float, memory_mb: float, 
                       gpu_memory_mb: float = 0, n_signals: int = 1):
        """Add metrics from a computation."""
        self.compute_times.append(compute_time)
        self.memory_usage.append(memory_mb)
        self.gpu_memory.append(gpu_memory_mb)
        self.throughput.append(n_signals / compute_time if compute_time > 0 else 0)
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'compute_time_mean': np.mean(self.compute_times) if self.compute_times else 0,
            'compute_time_std': np.std(self.compute_times) if self.compute_times else 0,
            'memory_mean_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_max_mb': np.max(self.memory_usage) if self.memory_usage else 0,
            'gpu_memory_mean_mb': np.mean(self.gpu_memory) if self.gpu_memory else 0,
            'gpu_memory_max_mb': np.max(self.gpu_memory) if self.gpu_memory else 0,
            'throughput_mean': np.mean(self.throughput) if self.throughput else 0,
            'n_computations': len(self.compute_times),
            'n_errors': len(self.errors)
        }


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def create_complex_signal(n_samples: int, n_channels: int = 1, 
                         fs: float = 512.0, complexity: str = 'medium') -> np.ndarray:
    """Create complex test signal with controlled properties."""
    t = np.linspace(0, n_samples/fs, n_samples)
    
    if complexity == 'simple':
        # Single PAC coupling
        pha_freq, amp_freq = 6.0, 80.0
        phase_signal = np.sin(2 * np.pi * pha_freq * t)
        amplitude_mod = (1 + 0.8 * np.cos(2 * np.pi * pha_freq * t)) / 2
        carrier = np.sin(2 * np.pi * amp_freq * t)
        signal = phase_signal + amplitude_mod * carrier * 0.5
        
    elif complexity == 'medium':
        # Multiple PAC couplings
        signal = np.zeros_like(t)
        pac_pairs = [(4, 60), (8, 90), (12, 110)]
        for pha_freq, amp_freq in pac_pairs:
            phase_signal = np.sin(2 * np.pi * pha_freq * t)
            amplitude_mod = (1 + 0.6 * np.cos(2 * np.pi * pha_freq * t)) / 2
            carrier = np.sin(2 * np.pi * amp_freq * t)
            signal += phase_signal + amplitude_mod * carrier * 0.3
    
    elif complexity == 'complex':
        # Multiple PAC with time-varying coupling
        signal = np.zeros_like(t)
        pac_pairs = [(4, 60), (8, 90), (12, 110), (16, 130)]
        for i, (pha_freq, amp_freq) in enumerate(pac_pairs):
            # Time-varying coupling strength
            coupling_strength = 0.3 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
            phase_signal = np.sin(2 * np.pi * pha_freq * t)
            amplitude_mod = (1 + coupling_strength * np.cos(2 * np.pi * pha_freq * t)) / 2
            carrier = np.sin(2 * np.pi * amp_freq * t)
            signal += phase_signal + amplitude_mod * carrier * 0.25
    
    # Add realistic noise
    noise_level = {'simple': 0.1, 'medium': 0.2, 'complex': 0.3}[complexity]
    noise = np.random.normal(0, noise_level, len(t))
    signal += noise
    
    # Create multi-channel signal if requested
    if n_channels > 1:
        signals = []
        for ch in range(n_channels):
            ch_signal = signal + np.random.normal(0, 0.05, len(signal))
            signals.append(ch_signal)
        signal = np.stack(signals, axis=0)
    else:
        signal = signal.reshape(1, -1)
    
    return signal.reshape(1, n_channels, 1, -1)


class TestPerformanceVaryingSignalLength:
    """Test performance with different signal lengths."""
    
    def test_signal_length_scaling(self):
        """Test how performance scales with signal length."""
        fs = 512.0
        signal_durations = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]  # seconds
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        results = {}
        
        for duration in signal_durations:
            print(f"\nTesting {duration}s signal...")
            n_samples = int(fs * duration)
            signal = create_complex_signal(n_samples, n_channels=1, complexity='medium')
            
            metrics = PerformanceMetrics()
            
            # Initialize model
            model = gpac.PAC(
                seq_len=n_samples,
                fs=fs,
                pha_n_bands=20,
                amp_n_bands=15,
                trainable=False
            ).to(device)
            
            signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)
            
            # Warm-up
            _ = model(signal_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Run multiple computations
            n_runs = 10
            for _ in range(n_runs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                mem_before = get_memory_usage()
                gpu_mem_before = get_gpu_memory_usage()
                
                start_time = time.time()
                _ = model(signal_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = time.time() - start_time
                
                mem_after = get_memory_usage()
                gpu_mem_after = get_gpu_memory_usage()
                
                metrics.add_computation(
                    compute_time,
                    mem_after - mem_before,
                    gpu_mem_after - gpu_mem_before
                )
            
            results[duration] = metrics.summary()
            print(f"  Mean compute time: {results[duration]['compute_time_mean']:.4f}s")
            print(f"  Throughput: {results[duration]['throughput_mean']:.2f} signals/s")
            
        # Verify scaling is reasonable
        times = [results[d]['compute_time_mean'] for d in signal_durations]
        # Check that computation time increases roughly linearly with signal length
        # Allow for some non-linearity due to FFT algorithms
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            duration_ratio = signal_durations[i] / signal_durations[i-1]
            # Allow up to 2x worse than linear scaling
            assert ratio < duration_ratio * 2, \
                f"Performance scaling worse than expected: {ratio:.2f}x for {duration_ratio:.2f}x signal length"


class TestPerformanceBatchProcessing:
    """Test performance with different batch sizes."""
    
    def test_batch_size_scaling(self):
        """Test how performance scales with batch size."""
        fs = 512.0
        n_samples = 1024
        batch_sizes = [1, 4, 8, 16, 32, 64]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Skip large batches on CPU to avoid excessive test time
        if device == 'cpu':
            batch_sizes = [1, 4, 8, 16]
        
        results = {}
        
        # Initialize model once
        model = gpac.PAC(
            seq_len=n_samples,
            fs=fs,
            pha_n_bands=15,
            amp_n_bands=10,
            trainable=False
        ).to(device)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            
            # Create batch of signals
            signals = []
            for _ in range(batch_size):
                signal = create_complex_signal(n_samples, n_channels=1, complexity='simple')
                signals.append(signal)
            
            batch_signal = np.concatenate(signals, axis=0)
            batch_tensor = torch.tensor(batch_signal, dtype=torch.float32).to(device)
            
            metrics = PerformanceMetrics()
            
            # Warm-up
            _ = model(batch_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Run multiple computations
            n_runs = 5
            for _ in range(n_runs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                mem_before = get_memory_usage()
                gpu_mem_before = get_gpu_memory_usage()
                
                start_time = time.time()
                _ = model(batch_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = time.time() - start_time
                
                mem_after = get_memory_usage()
                gpu_mem_after = get_gpu_memory_usage()
                
                metrics.add_computation(
                    compute_time,
                    mem_after - mem_before,
                    gpu_mem_after - gpu_mem_before,
                    n_signals=batch_size
                )
            
            results[batch_size] = metrics.summary()
            print(f"  Mean compute time: {results[batch_size]['compute_time_mean']:.4f}s")
            print(f"  Per-signal time: {results[batch_size]['compute_time_mean']/batch_size:.4f}s")
            print(f"  Throughput: {results[batch_size]['throughput_mean']:.2f} signals/s")
        
        # Verify batch processing efficiency
        single_time = results[1]['compute_time_mean']
        for batch_size in batch_sizes[1:]:
            batch_time = results[batch_size]['compute_time_mean']
            efficiency = (single_time * batch_size) / batch_time
            print(f"\nBatch size {batch_size} efficiency: {efficiency:.2f}x")
            # Batch processing should be at least somewhat efficient
            assert efficiency > 0.5, f"Batch processing inefficient for size {batch_size}"


class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def test_memory_scaling(self):
        """Test memory usage with increasing problem size."""
        fs = 512.0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test different problem sizes
        configs = [
            (1024, 10, 10),    # Small
            (2048, 20, 15),    # Medium
            (4096, 30, 20),    # Large
            (8192, 40, 25),    # Very large
        ]
        
        if device == 'cpu':
            # Reduce problem size for CPU
            configs = configs[:3]
        
        results = {}
        
        for seq_len, pha_bands, amp_bands in configs:
            print(f"\nTesting config: {seq_len} samples, {pha_bands}x{amp_bands} bands...")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            mem_before = get_memory_usage()
            gpu_mem_before = get_gpu_memory_usage()
            
            # Create model and signal
            model = gpac.PAC(
                seq_len=seq_len,
                fs=fs,
                pha_n_bands=pha_bands,
                amp_n_bands=amp_bands,
                trainable=False
            ).to(device)
            
            signal = create_complex_signal(seq_len, n_channels=1)
            signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)
            
            # Compute PAC
            _ = model(signal_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            mem_after = get_memory_usage()
            gpu_mem_after = get_gpu_memory_usage()
            
            total_elements = seq_len * pha_bands * amp_bands
            memory_used = mem_after - mem_before
            gpu_memory_used = gpu_mem_after - gpu_mem_before
            
            results[(seq_len, pha_bands, amp_bands)] = {
                'memory_mb': memory_used,
                'gpu_memory_mb': gpu_memory_used,
                'total_elements': total_elements,
                'memory_per_element': memory_used * 1024 * 1024 / total_elements if total_elements > 0 else 0
            }
            
            print(f"  Memory used: {memory_used:.2f} MB")
            print(f"  GPU memory used: {gpu_memory_used:.2f} MB")
            print(f"  Memory per element: {results[(seq_len, pha_bands, amp_bands)]['memory_per_element']:.2f} bytes")
            
            # Clean up
            del model, signal_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_short_signals(self):
        """Test with very short signals."""
        fs = 512.0
        min_samples = 64  # Very short signal
        
        signal = create_complex_signal(min_samples, n_channels=1)
        
        # Should handle short signals gracefully
        pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=5,
            amp_n_bands=5
        )
        
        assert pac_values.shape == (1, 1, 5, 5)
        # Convert to numpy if it's a tensor
        if torch.is_tensor(pac_values):
            pac_values_np = pac_values.cpu().numpy()
        else:
            pac_values_np = pac_values
        assert not np.any(np.isnan(pac_values_np))
        assert not np.any(np.isinf(pac_values_np))
    
    def test_high_frequency_resolution(self):
        """Test with very high frequency resolution."""
        fs = 1024.0
        n_samples = 2048
        
        signal = create_complex_signal(n_samples, n_channels=1)
        
        # High resolution
        pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=50,  # High resolution
            amp_n_bands=40
        )
        
        assert pac_values.shape == (1, 1, 50, 40)
        # Convert to numpy if it's a tensor
        if torch.is_tensor(pac_values):
            pac_values_np = pac_values.cpu().numpy()
        else:
            pac_values_np = pac_values
        assert not np.any(np.isnan(pac_values_np))
        assert not np.any(np.isinf(pac_values_np))
    
    def test_multichannel_signals(self):
        """Test with multi-channel signals."""
        fs = 512.0
        n_samples = 1024
        n_channels = 8
        
        signal = create_complex_signal(n_samples, n_channels=n_channels)
        
        pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_n_bands=10,
            amp_n_bands=10
        )
        
        assert pac_values.shape == (1, n_channels, 10, 10)
        # Convert to numpy if it's a tensor
        if torch.is_tensor(pac_values):
            pac_values_np = pac_values.cpu().numpy()
        else:
            pac_values_np = pac_values
        assert not np.any(np.isnan(pac_values_np))
        
    def test_extreme_frequency_ranges(self):
        """Test with extreme frequency ranges."""
        fs = 1000.0
        n_samples = 2000
        
        signal = create_complex_signal(n_samples, n_channels=1)
        
        # Test near Nyquist frequency
        pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
            signal,
            fs=fs,
            pha_start_hz=2.0,
            pha_end_hz=50.0,
            amp_start_hz=100.0,
            amp_end_hz=450.0,  # Close to Nyquist (500 Hz)
            pha_n_bands=10,
            amp_n_bands=10
        )
        
        assert pac_values.shape == (1, 1, 10, 10)
        # Convert to numpy if it's a tensor
        if torch.is_tensor(pac_values):
            pac_values_np = pac_values.cpu().numpy()
        else:
            pac_values_np = pac_values
        assert not np.any(np.isnan(pac_values_np))
        assert amp_freqs[-1] <= fs/2  # Should not exceed Nyquist


class TestGPUvsCPU:
    """Compare GPU vs CPU performance."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_vs_cpu_performance(self):
        """Compare performance between GPU and CPU."""
        fs = 512.0
        test_configs = [
            (512, 10, 10),     # Small
            (1024, 20, 15),    # Medium
            (2048, 30, 20),    # Large
        ]
        
        results = {}
        
        for seq_len, pha_bands, amp_bands in test_configs:
            print(f"\n\nTesting GPU vs CPU: {seq_len} samples, {pha_bands}x{amp_bands} bands")
            
            signal = create_complex_signal(seq_len, n_channels=1)
            n_runs = 5
            
            for device in ['cpu', 'cuda']:
                print(f"\n{device.upper()} Performance:")
                
                # Initialize model
                model = gpac.PAC(
                    seq_len=seq_len,
                    fs=fs,
                    pha_n_bands=pha_bands,
                    amp_n_bands=amp_bands,
                    trainable=False
                ).to(device)
                
                signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)
                
                # Warm-up
                _ = model(signal_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                # Time computations
                times = []
                for _ in range(n_runs):
                    start_time = time.time()
                    _ = model(signal_tensor)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                
                mean_time = np.mean(times)
                std_time = np.std(times)
                
                config_key = f"{seq_len}_{pha_bands}x{amp_bands}"
                if config_key not in results:
                    results[config_key] = {}
                
                results[config_key][device] = {
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'throughput': 1.0 / mean_time
                }
                
                print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f}s")
                print(f"  Throughput: {1.0/mean_time:.2f} signals/s")
            
            # Calculate speedup
            config_key = f"{seq_len}_{pha_bands}x{amp_bands}"
            cpu_time = results[config_key]['cpu']['mean_time']
            gpu_time = results[config_key]['cuda']['mean_time']
            speedup = cpu_time / gpu_time
            
            print(f"\n🚀 GPU Speedup: {speedup:.2f}x")
            
            # GPU should generally be faster for non-trivial problem sizes
            if pha_bands * amp_bands >= 150:  # Medium to large problems
                assert speedup > 1.0, f"GPU should be faster for problem size {pha_bands}x{amp_bands}"


def run_all_performance_tests():
    """Run all performance tests and generate report."""
    print("=" * 80)
    print("gPAC v1.0.0 COMPREHENSIVE PERFORMANCE TEST SUITE")
    print("=" * 80)
    
    test_classes = [
        TestPerformanceVaryingSignalLength(),
        TestPerformanceBatchProcessing(),
        TestMemoryUsage(),
        TestEdgeCases(),
    ]
    
    if torch.cuda.is_available():
        test_classes.append(TestGPUvsCPU())
        print(f"GPU Available: {torch.cuda.get_device_name()}")
    else:
        print("GPU: Not available (CPU-only tests)")
    
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print("=" * 80)
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*60}")
        print(f"Running {class_name}")
        print(f"{'='*60}")
        
        # Run all test methods
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                print(f"\n>>> {method_name}")
                method = getattr(test_class, method_name)
                try:
                    method()
                    print("✅ PASSED")
                except Exception as e:
                    print(f"❌ FAILED: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST SUITE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_all_performance_tests()