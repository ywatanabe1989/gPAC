# gPAC Examples

<<<<<<< HEAD
This directory contains comprehensive examples demonstrating various features and use cases of gPAC (GPU-accelerated Phase-Amplitude Coupling).

## 📁 Directory Structure

### 🧠 `motor_imagery/`
- **motor_imagery_demo.py**: Analyze motor imagery (left/right hand) using PAC features
- **motor_imagery_demo_zscore.py**: Z-score normalized analysis with multiple classifiers

### 🔄 `ComparisonBenchmarkers/`
- **quick_comparison_demo.py**: Quick comparison between gPAC and TensorPAC
- **example_scientific_comparison.py**: Comprehensive scientific comparison with benchmarks

### 🪫 `epilepsty/`
- **epilepsy_classification_demo.py**: Epileptic seizure detection using PAC features

### 🔧 `gpac/`
Core functionality examples:
- **simple_pac_demo.py**: Basic PAC computation example
- **example_pac_analysis.py**: Comprehensive PAC analysis workflow
- **example_hilbert.py**: Hilbert transform usage
- **example_modulation_index.py**: Modulation Index computation
- **example_bandpass_filter.py**: Bandpass filtering examples
- **example_profiler.py**: Performance profiling utilities
- **example__SyntheticDataGenerator.py**: Generate synthetic PAC signals

#### `gpac/_Filters/`
Filter-specific examples:
- **example_StaticBandPassFilter.py**: Static (non-trainable) filter usage
- **example_DifferentiableBandPassFilter.py**: Trainable filter for deep learning
- **simple_static_filter_demo.py**: Basic static filtering
- **simple_differentiable_filter_demo.py**: Gradient-based filter optimization

### ✋ `handgrasping/`
- **hand_grasping_demo.py**: Analyze hand grasping movements using PAC

### 🚀 `performance/`
- **performance_test.py**: Basic performance benchmarks
- **comprehensive_benchmark.py**: Detailed performance analysis
- **performance_comparison.py**: Compare different configurations

### 📖 `readme/`
- **readme_demo.py**: Generate figures for README documentation

### 🎯 `trainability/`
Deep learning integration examples:
- **example_pac_trainability.py**: Train frequency bands via backpropagation
- **example_pac_trainability_simple.py**: Simplified trainable PAC
- **example_pac_trainability_working.py**: Working example with convergence

## 🚀 Running Examples

All examples follow the STX framework and can be run directly:

```bash
# Run a single example
python examples/gpac/simple_pac_demo.py

# Run all examples (from project root)
./.playground/run_examples.sh
```
=======
This directory contains example scripts demonstrating various features of the gPAC (GPU-accelerated Phase-Amplitude Coupling) package.

## Overview

All examples follow the stx framework conventions for consistent output handling and visualization.
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf

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

<<<<<<< HEAD
## 📊 Output Structure

Each example creates an output directory following STX conventions:
```
example_name_out/
├── RUNNING/          # Active runs
├── FINISHED_SUCCESS/ # Completed runs
├── FINISHED_ERROR/   # Failed runs
└── figures/          # Generated visualizations
```

## 🔗 Key Features Demonstrated

1. **GPU Acceleration**: All examples utilize CUDA when available
2. **Comparison Framework**: Side-by-side comparison with other implementations
3. **Real-world Data**: Examples using MNE sample datasets
4. **Deep Learning**: Trainable frequency bands and differentiable operations
5. **Visualization**: Publication-quality figures with proper formatting
=======
## Output Structure

All examples save their outputs using stx conventions:
```
examples/
└── outputs/
    ├── example_pac_analysis/
    │   ├── pac_analysis.gif
    │   └── pac_results.pkl
    ├── example_bandpass_filter/
    │   ├── filter_frequency_response.gif
    │   ├── filtering_results.gif
    │   └── filter_info.yaml
    └── example_profiler/
        ├── profiling_results.gif
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
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf

## Requirements

- PyTorch with CUDA support (for GPU acceleration)
- stx >= 1.0.0
- numpy, scipy, matplotlib
- Optional: tensorpac (for comparison examples)

<<<<<<< HEAD
## 💡 Tips

- Set `matplotlib.use('Agg')` for headless environments (already configured)
- Use `--device cpu` flag if CUDA is unavailable
- Check `*_out/` directories for results and logs
- Review STX logs for debugging information
=======
## Tips

1. **GPU Usage**: Examples automatically detect and use GPU if available
2. **Batch Processing**: Adjust batch sizes based on your GPU memory
3. **Visualization**: All plots are saved to `outputs/` directory
4. **Caching**: Some examples use `@stx.io.decorator.cache` for faster reruns
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf

## Common Issues

- **Out of Memory**: Reduce batch size in profiler example
- **No GPU**: Examples fall back to CPU automatically
- **Missing TensorPAC**: Comparison features are optional

<<<<<<< HEAD
## 📝 Creating New Examples

Follow the STX template:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
__FILE__ = "./examples/category/your_example.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scitex as stx

def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments
    return parser.parse_args()

@stx.gen.decorators.flow
def run_main(args):
    CONFIG, CC, sdir = stx.gen.start(__file__, args=args, pyplot_backend="Agg")
    
    # Your code here
    
    stx.gen.close()

if __name__ == "__main__":
    args = parse_args()
    run_main(args)
```

<!-- EOF -->
=======
## Contributing

When adding new examples:
1. Follow stx conventions for output handling
2. Use descriptive filenames starting with `example_`
3. Include docstrings explaining the example's purpose
4. Save all outputs to `outputs/<script_name>/`
>>>>>>> 4a22432c6a307d7609df622a94c48133160cf1bf
