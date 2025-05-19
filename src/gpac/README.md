# gPAC Implementation Details

## Core Components

### Public API
- `__init__.py`: Exports the main classes and functions
- `_pac.py`: Contains the `calculate_pac()` function for high-level API
- `_PAC.py`: Defines the main `PAC` class (PyTorch Module) for core calculations
- `_SyntheticDataGenerator.py`: Generator for synthetic PAC signals

### Internal Modules
- `_BandPassFilter.py`: Standard bandpass filtering implementation
- `_DifferenciableBandPassFilter.py`: Gradient-enabled filtering for trainable models
- `_Hilbert.py`: FFT-based Hilbert transform implementation
- `_ModulationIndex.py`: Modulation Index calculation for quantifying PAC
- `_utils.py`: Utility functions for tensor manipulation
- `_decorators.py`: Function decorators for timing, etc.

## Calculation Pipeline

1. **Input Preparation**: Shape validation and normalization
2. **Bandpass Filtering**: Extract phase and amplitude bands
3. **Hilbert Transform**: Calculate analytic signal for phase and amplitude
4. **Modulation Index**: Quantify phase-amplitude coupling 
5. **Permutation Testing**: Generate distribution of surrogate values (optional)
6. **Return**: PAC values, frequency information, and optional surrogate distribution

## Advanced Features

### Return Full Surrogate Distributions

The `return_dist=True` parameter in `calculate_pac()` allows accessing the full distribution of surrogate values from permutation testing:

```python
# With return_dist=True and n_perm not None
pac_values, surrogate_dist, freqs_pha, freqs_amp = calculate_pac(
    signal=signal,
    fs=fs,
    n_perm=200,
    return_dist=True
)
```

### Trainable Frequency Bands

Enable gradient flow through frequency band selection with `trainable=True`:

```python
pac_values, freqs_pha, freqs_amp = calculate_pac(
    signal=signal,
    fs=fs,
    trainable=True
)
```

### Chunked Processing

For very large datasets, use `chunk_size` to process data in manageable blocks:

```python
pac_values, freqs_pha, freqs_amp = calculate_pac(
    signal=large_signal,
    fs=fs,
    chunk_size=16  # Process 16 traces at a time
)
```

### Synthetic Data Generation

Create signals with known PAC properties for testing and validation:

```python
data_generator = SyntheticDataGenerator(fs=1000.0)
datasets = data_generator.generate_and_split()
```