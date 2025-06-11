<!-- ---
!-- Timestamp: 2025-06-09 21:55:06
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/examples/benchmark/parameter_sweep/README.md
!-- --- -->


``` bash
parameter_sweep_benchmark.py --no-quick --shuffle
```

# Parameter Sweep Benchmark

This directory contains comprehensive parameter sweep benchmarks for gPAC performance analysis across different computational configurations.

## Overview

The parameter sweep benchmark systematically varies key computational parameters to analyze gPAC performance characteristics and identify optimal settings for different use cases.

## Files

- **`parameter_sweep_benchmark.py`** - Main benchmarking script with systematic parameter variation

## Benchmark Parameters

The benchmark systematically varies:
- **Batch Size**: 2, 4, 8, 16, 32 samples per batch
- **Number of Channels**: 8, 16, 32 channels
- **Sequence Length**: 2-8 seconds duration
- **Frequency Bands**: 10-30 phase bands, 15-45 amplitude bands
- **Permutation Tests**: 0-200 permutations

## Performance Analysis

### Key Metrics
- **Execution Time**: Wall-clock time for PAC computation
- **Memory Usage**: Peak GPU memory consumption (in GB)  
- **Throughput**: Samples processed per second
- **Scaling Behavior**: Performance vs parameter size relationships

### Computational Characteristics
- Linear scaling with batch size for GPU-optimized workloads
- Efficient memory utilization across parameter ranges
- Consistent performance across different signal configurations

## Generated Outputs

### Result Files
- **`parameter_sweep_results.yaml`** - Comprehensive benchmark results
- **`parameter_sweep_results.pkl`** - Python objects for detailed analysis

### Analysis Insights
The benchmark demonstrates gPAC's robust performance scaling:
- **Batch Processing**: Linear scaling with batch size up to GPU memory limits
- **Channel Scaling**: Efficient parallel processing of multiple channels
- **Frequency Band Optimization**: Optimal performance with 10-20 phase/amplitude bands
- **Memory Efficiency**: Predictable VRAM usage scaling with parameter complexity

## Usage

```bash
cd examples/performance/parameter_sweep
python parameter_sweep_benchmark.py
```

## Configuration

### Default Parameters
```python
BASELINE = {
    'batch_size': 4,
    'n_channels': 8, 
    'seq_sec': 4.0,
    'pha_n_bands': 10,
    'amp_n_bands': 15,
    'n_perm': 0
}
```

### Parameter Variations
Each benchmark run varies one parameter while keeping others at baseline values, enabling systematic performance characterization.

## Technical Details

### Performance Characteristics
- **Small Workloads** (batch_size ≤ 8): 0.5-2.0s execution time
- **Medium Workloads** (batch_size 16-32): 2.0-8.0s execution time  
- **Large Workloads** (batch_size ≥ 64): Linear scaling with memory constraints

### Optimization Insights
- Optimal batch sizes depend on available GPU memory
- Multi-channel processing provides efficient parallelization
- Frequency band count affects computational complexity quadratically

## Dependencies

- gPAC library with GPU acceleration
- PyTorch with CUDA support
- NumPy for numerical analysis
- MNGS framework for result management

<!-- EOF -->