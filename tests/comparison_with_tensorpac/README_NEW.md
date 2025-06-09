# gPAC vs TensorPAC Comprehensive Comparison

This directory contains comprehensive tests comparing gPAC and TensorPAC implementations module by module in terms of both speed and accuracy.

## Test Modules

### 1. `test_comprehensive_comparison.py`
Complete module-by-module comparison including:
- **BandPass Filter**: Compares FIR (gPAC) vs Butterworth (TensorPAC) filters
- **Hilbert Transform**: Tests scipy-compatible implementations
- **Modulation Index**: Compares Tort MI method implementations
- **Full PAC Pipeline**: End-to-end PAC computation comparison

### 2. `run_comparison.py`
Script to run all comparisons and generate detailed reports.

## Usage

### Quick Test
```bash
# Run all comparisons with console output
python test_comprehensive_comparison.py
```

### Generate Report
```bash
# Run comparisons and save detailed report
python run_comparison.py --save-report

# Specify output directory
python run_comparison.py --save-report --output-dir results/
```

## Key Features

### Accuracy Testing
- Correlation metrics (Pearson, Spearman)
- Error metrics (MAE, MSE, RMSE)
- Shape verification and compatibility checks
- Scale factor analysis

### Performance Testing
- Speed comparisons with timing measurements
- Memory profiling (CPU/GPU)
- Batch processing benchmarks
- Multi-channel/epoch efficiency tests

### Utilities Used
- `gpac.utils.compare`: Comparison utilities for shape verification and metrics
- `gpac.utils._profiler`: Performance profiling with CPU/GPU tracking
- `gpac.dataset._SyntheticDataGenerator`: Consistent synthetic data generation

## Expected Results

### Accuracy
- **BandPass Filter**: >0.8 correlation (different filter types)
- **Hilbert Transform**: >0.999 correlation (near-perfect match)
- **Modulation Index**: ~10-15% scale difference (acceptable)
- **Full PAC**: >0.7 correlation overall

### Performance
- **Single-channel**: 2-5x speedup
- **Multi-channel batching**: 5-10x speedup
- **GPU acceleration**: Additional 2-3x speedup
- **Memory efficiency**: Better VRAM utilization

## Key Differences

1. **Filter Types**
   - gPAC: FIR filters with Kaiser window
   - TensorPAC: Butterworth IIR filters
   - Impact: ~3.4x power ratio difference

2. **Implementation**
   - gPAC: PyTorch-based, GPU-accelerated
   - TensorPAC: NumPy/SciPy-based, CPU-only
   - Impact: Significant speedup with batching

3. **API Design**
   - gPAC: Batch-first, multi-dimensional tensors
   - TensorPAC: Channel-wise processing
   - Impact: More efficient batch processing

## Recommendations

### For Direct Comparison
- Use `n_bins=18` for Tort MI method
- Account for ~3.4x filter power difference
- Expect ~10% MI scale difference
- Use `trainable=False` for fixed filters

### For Optimal Performance
- Use GPU when available
- Process multiple channels/epochs in batches
- Leverage memory management features
- Use appropriate data shapes for each library

## Example Output

```
================================================================================
gPAC vs TensorPAC Comprehensive Comparison
================================================================================
Timestamp: 2025-06-08 00:15:00
Device: CUDA
GPU: NVIDIA A100-SXM4-80GB
================================================================================

[1/4] Testing BandPass Filter...
--- BandPass Filter Comparison ---
Phase correlation: 0.8234
Amplitude correlation: 0.8567
Phase power ratio (gPAC/TensorPAC): 0.2941
Amp power ratio (gPAC/TensorPAC): 0.3012

[2/4] Testing Hilbert Transform...
--- Hilbert Transform Comparison ---
Phase correlation: 0.999998
Phase RMSE: 0.000002
Amplitude correlation: 0.999999
Amplitude RMSE: 0.000001

[3/4] Testing Modulation Index...
--- Modulation Index Comparison ---
gPAC MI: 0.324567
TensorPAC MI: 0.356789
Relative difference: 9.03%

[4/4] Testing Full PAC Pipeline...
--- Full PAC Pipeline Comparison ---
Correlation: 0.7845
Scale factor: 1.12x
Normalized MAE: 0.0234

================================================================================
COMPREHENSIVE COMPARISON SUMMARY
================================================================================
✅ FILTER: PASSED
✅ HILBERT: PASSED
✅ MI: PASSED
✅ PAC: PASSED
```