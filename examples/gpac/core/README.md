# gPAC Examples

This directory contains demonstration examples for the gPAC (GPU-accelerated Phase-Amplitude Coupling) package components.

## Examples Overview

### 1. `example__BandPassFilter.py`
Demonstrates the BandPassFilter functionality for PAC analysis:
- **Static filtering**: Fixed frequency bands for phase and amplitude
- **Trainable filtering**: Adaptive frequency bands that can be optimized
- **Outputs**:
  - `01_static_bandpass_filter.gif`: Visualization of filtered signals with fixed bands
  - `02_trainable_bandpass_filter.gif`: Frequency band distribution for trainable filters

### 2. `example__Hilbert.py`
Demonstrates the Hilbert transform for instantaneous phase and amplitude extraction:
- **Signal analysis**: Phase, amplitude, and instantaneous frequency extraction
- **Gradient testing**: Validates differentiability for neural network training
- **Performance benchmarking**: Batch processing and throughput measurements
- **Outputs**:
  - `01_hilbert_analysis.gif`: Comprehensive signal analysis visualization
  - `02_hilbert_performance.gif`: Performance metrics and batch processing

## Running the Examples

Each example can be run independently:

```bash
# Run individual examples
python examples/gpac/example__BandPassFilter.py
python examples/gpac/example__Hilbert.py

# Run all examples in this directory
../../run_examples.sh gpac
```

## Output Structure

All examples follow the stx framework and create output directories automatically:
- `example__BandPassFilter_out/`: Contains bandpass filter visualizations
- `example__Hilbert_out/`: Contains Hilbert transform analysis results

## Requirements

- PyTorch with CUDA support (optional but recommended)
- NumPy, Matplotlib
- stx framework
- gpac package

## Notes

- Examples use synthetic data generation from `gpac.dataset`
- All visualizations are saved as static GIF files (not animations)
- The stx framework handles logging and output directory management automatically