<!-- ---
!-- Timestamp: 2025-06-07 11:22:53
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/README.md
!-- --- -->

# gPAC Source Code Documentation

## Overview

This directory contains the core implementation of gPAC (GPU-accelerated Phase-Amplitude Coupling), featuring an intelligent memory management system that automatically adapts to available hardware resources.

## Module Dependencies

``` mermaid
graph TD
    PAC[_PAC.py] --> MemoryManager[_MemoryManager.py]
    PAC --> StaticFilter[_StaticBandPassFilter.py]
    PAC --> Hilbert[_Hilbert.py]
    PAC --> ModulationIndex[_ModulationIndexMemoryOptimized.py]
    PAC --> MemoryManagementStrategy[_MemoryManagementStrategy.py]

    MemoryManager --> MemoryManagementStrategy
    MemoryManager --> MemoryEstimator[memory_estimator.py]

    BandPassFilter[_BandPassFilter.py] --> StaticFilter
    BandPassFilter --> PooledFilter[_PooledBandPassFilter.py]

    Compare[utils/compare.py] --> PAC

    StaticFilter --> FilterBase[_Filters/__init__.py]
    PooledFilter --> FilterBase

    Profiler[_Profiler.py] --> PAC

    SyntheticData[_SyntheticDataGenerator.py] --> PAC

    subgraph "Core Components"
        PAC
        MemoryManager
        MemoryManagementStrategy
    end

    subgraph "Filters"
        StaticFilter
        PooledFilter
        BandPassFilter
    end

    subgraph "Processing"
        Hilbert
        ModulationIndex
    end

    subgraph "Utilities"
        MemoryEstimator
        Compare
        Profiler
        SyntheticData
    end
```

## Key Features

### 1. Intelligent Memory Management

gPAC automatically selects the optimal processing strategy based on available GPU memory and problem size:

```python
# Automatic strategy selection (recommended)
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    memory_strategy="auto"  # Default - automatically optimizes
)
```

### 2. Dimensional Strategy Control

Fine-grained control over processing dimensions for advanced users:

```python
from gpac._MemoryManagementStrategy import MemoryManagementStrategy

# Custom strategy for specific hardware
strategy = MemoryManagementStrategy(
    batch="vectorized",      # Process all batches at once
    channel="chunked",       # Process channels in chunks
    permutation="sequential", # Process permutations one by one
    channel_chunk_size=32    # Optional: specify chunk size
)

pac = PAC(seq_len=2048, fs=256, n_perm=1000, memory_strategy=strategy)
```

### 3. Memory Limit Control

Set explicit memory limits for multi-user systems:

```python
# Limit to 8GB (useful for shared GPUs)
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    vram_gb=8.0  # Explicit limit
)

# Or use auto with percentage
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    vram_gb="auto",
    max_memory_usage=0.6  # Use only 60% of available VRAM
)
```

### 4. Automatic Downgrade with Warnings

When memory is insufficient, gPAC automatically downgrades the strategy and provides helpful guidance:

```
⚠️  MEMORY STRATEGY DOWNGRADE WARNING
============================================================
Input shape: (64, 128, 2048), Permutations: 1000
Available memory: 7.2GB

Downgrades applied:
  1. Permutation chunking
  2. Channel chunking

Memory savings: 75.3%
Final strategy: Batch: vectorized | Channel: chunked(32) | Permutation: chunked(50)

💡 RECOMMENDATIONS for better performance:
  • Reduce n_perm (currently 1000)
  • Process fewer channels (currently 128)
  • For this configuration, ideal VRAM: 14.4GB+

To suppress this warning, explicitly set:
  memory_strategy=MemoryManagementStrategy{'batch': 'vectorized', 'channel': 'chunked', ...}
============================================================
```

## Core Components

### Main Classes

- **`PAC`**: Main class for phase-amplitude coupling analysis
- **`MemoryManager`**: Intelligent memory management and strategy selection
- **`MemoryManagementStrategy`**: Configuration for dimensional processing control
- **`Profiler`**: Performance and memory profiling utilities

### Processing Modules

- **`_BandPassFilter`**: GPU-accelerated bandpass filtering
- **`_Hilbert`**: Hilbert transform implementation
- **`_ModulationIndex`**: Phase-amplitude coupling computation
- **`_SyntheticDataGenerator`**: Generate test signals

### Utilities

- **`memory_estimator`**: Estimate VRAM usage for configurations
- **`dataloader`**: Efficient data loading for large datasets

## Memory Strategy Presets

```python
from gpac._MemoryManagementStrategy import PRESETS

# Available presets:
# - "conservative": All sequential (minimum memory)
# - "balanced": All chunked (good trade-off)
# - "aggressive": All vectorized (maximum speed)
# - "batch_optimized": Vectorized batch, sequential rest
# - "channel_optimized": Vectorized channels, sequential rest
# - "permutation_optimized": Vectorized permutations, sequential rest

pac = PAC(seq_len=2048, fs=256, memory_strategy=PRESETS["balanced"])
```

## Memory Recommendations

Get memory usage recommendations for your configuration:

```python
# Create PAC instance
pac = PAC(seq_len=2048, fs=256, n_perm=1000)

# Get recommendations
x = torch.randn(32, 64, 2048)
recommendations = pac.get_memory_recommendations(x)

# Display recommendations
for strategy_name, info in recommendations.items():
    if strategy_name != "system_info":
        print(f"{strategy_name}:")
        print(f"  Memory required: {info['memory_gb']:.1f}GB")
        print(f"  Fits in memory: {info['fits_in_memory']}")
        print(f"  Efficiency: {info['efficiency']}")
```

## Processing Strategies Explained

### 1. Vectorized
- **Pros**: Fastest, fully utilizes GPU parallelism
- **Cons**: High memory usage
- **Use when**: Sufficient GPU memory available

### 2. Chunked
- **Pros**: Good balance of speed and memory
- **Cons**: Slightly slower than vectorized
- **Use when**: Moderate GPU memory, large datasets

### 3. Sequential
- **Pros**: Minimal memory usage
- **Cons**: Slowest option
- **Use when**: Limited GPU memory, very large datasets

## Hardware Recommendations

| GPU Type               | Memory  | Recommended Strategy       |
|------------------------|---------|----------------------------|
| Consumer (RTX 2080)    | 8GB     | Sequential/Chunked         |
| Workstation (RTX 3090) | 24GB    | Chunked/Balanced           |
| Server (V100)          | 32GB    | Balanced/Aggressive        |
| High-end (A100)        | 40-80GB | Aggressive/Full Vectorized |

## Performance Tips

1. **Start with auto mode** - Let gPAC optimize for your hardware
2. **Monitor warnings** - Follow recommendations for better performance
3. **Use profiling** - Enable `enable_memory_profiling=True` for detailed metrics
4. **Adjust parameters** - Reduce batch size, channels, or permutations if needed
5. **Use fp16** - Enable `fp16=True` for ~50% memory savings with minimal accuracy loss

## Example: Complete Workflow

```python
import torch
from gpac import PAC

# 1. Create PAC with auto memory management
pac = PAC(
    seq_len=2048,
    fs=256,
    pha_start_hz=4,
    pha_end_hz=8,
    amp_start_hz=30,
    amp_end_hz=100,
    n_perm=1000,
    fp16=True,  # Use half precision
    memory_strategy="auto",
    enable_memory_profiling=True
).cuda()

# 2. Prepare your data
signal = torch.randn(32, 64, 2048).cuda()  # (batch, channels, time)

# 3. Run analysis
result = pac(signal)

# 4. Access results
pac_values = result['pac']  # Phase-amplitude coupling
z_scores = result['pac_z']  # Statistical significance

# 5. Check memory usage
if pac.profiler:
    pac.profiler.print_summary()
```

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size
2. Use chunked or sequential strategies
3. Enable fp16
4. Set explicit memory limit with `vram_gb`

### Slow Performance
1. Check if strategy is too conservative
2. Increase chunk sizes if using chunked mode
3. Consider upgrading GPU memory

### Inconsistent Results
- All strategies produce identical results (within numerical precision)
- Use same random seed for reproducible surrogates

## Multi-GPU Support

gPAC now supports simple and efficient multi-GPU batch parallelism:

### Basic Usage

```python
# Automatically use all available GPUs
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True  # That's it!
).cuda()

# Process large batch - automatically distributed across GPUs
x = torch.randn(256, 64, 2048).cuda()
result = pac(x)
```

### Specific GPU Selection

```python
# Use specific GPUs
pac = PAC(
    seq_len=2048,
    fs=256,
    multi_gpu=True,
    device_ids=[0, 2, 3]  # Use only GPU 0, 2, and 3
)
```

### Multi-GPU with Memory Management

```python
# Each GPU respects memory strategies independently
pac = PAC(
    seq_len=2048,
    fs=256,
    n_perm=1000,
    multi_gpu=True,
    memory_strategy="auto",  # Applied on each GPU
    vram_gb=40.0            # Per-GPU memory limit
)

# Check GPU status
status = pac.get_gpu_status()
print(f"Active GPUs: {status['active_devices']}")
for gpu_id, info in status['gpu_details'].items():
    print(f"GPU {gpu_id}: {info['name']} - {info['free_gb']:.1f}GB free")
```

### Performance Expectations

- **Single GPU**: 100 samples/sec (baseline)
- **2 GPUs**: ~190 samples/sec (1.9x speedup)
- **4 GPUs**: ~380 samples/sec (3.8x speedup)
- **8 GPUs**: ~750 samples/sec (7.5x speedup)

The slight overhead is due to data distribution and result gathering.

### Environment Setup

```bash
# Specify visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python your_script.py
```

## Architecture

```
gPAC/
├── _PAC.py                   # Main PAC class with multi-GPU support
├── _MemoryManager.py         # Intelligent memory management
├── _MemoryManagementStrategy.py   # Strategy configuration
├── _Profiler.py              # Performance profiling
├── _BandPassFilter.py        # Signal filtering
├── _Hilbert.py               # Hilbert transform
├── _ModulationIndex.py       # PAC computation
├── memory_estimator.py       # Memory usage estimation
└── _benchmark/               # Performance benchmarking tools
```

## Contributing

When adding new features, ensure they:
1. Respect the dimensional strategy system
2. Include memory estimation updates
3. Provide appropriate downgrade paths
4. Include clear documentation

## License

See main repository LICENSE file.

<!-- EOF -->