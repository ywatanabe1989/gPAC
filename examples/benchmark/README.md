# Performance Examples

This directory contains focused, essential performance testing examples for gPAC multi-GPU functionality.

## Structure

### `multiple_gpus/` - Multi-GPU Performance Testing
Core multi-GPU benchmarking suite with modular design:

- **`utils.py`** - Shared utilities and helper functions for all multi-GPU tests
- **`speed.py`** - Computational speedup testing (same data size, faster processing)
- **`throughput.py`** - Data scaling benefits (4x larger datasets with 4 GPUs - key value proposition)
- **`vram.py`** - Memory scaling tests to find VRAM capacity limits
- **`comodulogram.py`** - Real-world PAC workflow demonstration with high-resolution analysis

### `parameter_sweep/` - Systematic Performance Analysis
Comprehensive parameter analysis and comparisons:

- **`parameter_sweep_benchmark.py`** - Systematic variation of PAC parameters to understand performance scaling
- **`performance_comparison.py`** - Direct gPAC vs TensorPAC performance comparison

## Key Features

✅ **Clean, focused examples** - Each file serves a unique purpose
✅ **Stable API usage** - Uses only public gPAC APIs
✅ **Modular design** - Shared utilities prevent code duplication
✅ **Comprehensive coverage** - Tests all aspects of multi-GPU benefits
✅ **Real-world relevance** - Includes practical workflow examples

## Usage

All examples follow the MNGS template format and can be run independently:

```bash
# Multi-GPU speed comparison
python examples/performance/multiple_gpus/speed.py --n_perm 50

# Data throughput scaling (key multi-GPU benefit)
python examples/performance/multiple_gpus/throughput.py --n_perm 100

# Memory scaling analysis
python examples/performance/multiple_gpus/vram.py

# Real-world comodulogram workflow
python examples/performance/multiple_gpus/comodulogram.py --resolution high

# Parameter sweep analysis
python examples/performance/parameter_sweep/parameter_sweep_benchmark.py --quick

# gPAC vs TensorPAC comparison
python examples/performance/parameter_sweep/performance_comparison.py
```

## Archived Scripts

Redundant or overly complex scripts have been moved to `.old/` directory:
- `performance_test.py` - Redundant basic testing (covered by parameter_sweep_benchmark.py)
- `_run_benchmark.py` - Complex internal API usage (unstable dependencies)

## Multi-GPU Value Proposition

These examples demonstrate the key benefits of multi-GPU PAC computation:

1. **Throughput Scaling**: Process 4x larger datasets with 4 GPUs
2. **Computational Speedup**: Faster processing of identical workloads  
3. **Memory Scaling**: Access to combined VRAM for larger analyses
4. **Real-world Workflows**: High-resolution comodulogram analysis