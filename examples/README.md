# gPAC Examples

This directory contains example scripts demonstrating various features of the gPAC (GPU-accelerated Phase-Amplitude Coupling) package.

## Overview

All examples follow the mngs framework conventions for consistent output handling and visualization.

## Main Examples

### 1. `example_pac_analysis.py`
**Basic PAC Analysis**
- Demonstrates core PAC computation using gPAC
- Generates synthetic data with known PAC coupling
- Visualizes PAC comodulogram
- Compares with TensorPAC (if available)

```bash
python example_pac_analysis.py
```

### 2. `example_bandpass_filter.py`
**Bandpass Filtering Demo**
- Shows how to use gPAC's bandpass filter
- Visualizes filter frequency responses
- Demonstrates filtering for phase and amplitude bands
- Compares different filter configurations

```bash
python example_bandpass_filter.py
```

### 3. `example_profiler.py`
**Performance Profiling**
- Profiles PAC computation performance
- Tracks GPU memory usage (VRAM)
- Tests different batch sizes
- Generates performance reports and visualizations

```bash
python example_profiler.py
```

## Output Structure

All examples save their outputs using mngs conventions:
```
examples/
└── outputs/
    ├── example_pac_analysis/
    │   ├── pac_analysis.png
    │   └── pac_results.pkl
    ├── example_bandpass_filter/
    │   ├── filter_frequency_response.png
    │   ├── filtering_results.png
    │   └── filter_info.yaml
    └── example_profiler/
        ├── profiling_results.png
        ├── profiling_data.yaml
        └── performance_report.txt
```

## Advanced Examples

### Trainability Examples (`trainability/`)
Demonstrates gPAC's differentiability for gradient-based optimization:
- `example_pac_trainability.py` - Full training example
- `example_pac_trainability_simple.py` - Simplified version
- `example_pac_trainability_working.py` - Working implementation with outputs

### Comparison Examples (`comparison_with_tensorpac/`)
Detailed comparisons between gPAC and TensorPAC implementations.

## Requirements

- PyTorch with CUDA support (for GPU acceleration)
- mngs >= 1.0.0
- numpy, scipy, matplotlib
- Optional: tensorpac (for comparison examples)

## Tips

1. **GPU Usage**: Examples automatically detect and use GPU if available
2. **Batch Processing**: Adjust batch sizes based on your GPU memory
3. **Visualization**: All plots are saved to `outputs/` directory
4. **Caching**: Some examples use `@mngs.io.decorator.cache` for faster reruns

## Common Issues

- **Out of Memory**: Reduce batch size in profiler example
- **No GPU**: Examples fall back to CPU automatically
- **Missing TensorPAC**: Comparison features are optional

## Contributing

When adding new examples:
1. Follow mngs conventions for output handling
2. Use descriptive filenames starting with `example_`
3. Include docstrings explaining the example's purpose
4. Save all outputs to `outputs/<script_name>/`