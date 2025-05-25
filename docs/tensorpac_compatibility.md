# TensorPAC Compatibility Mode for gPAC

## Overview

gPAC now includes a TensorPAC-compatible mode that matches TensorPAC's filter implementation and default parameters for fair comparisons between the libraries.

## Key Features

### 1. TensorPAC's Custom FIR1 Implementation
- Uses TensorPAC's custom `fir1` filter design instead of scipy's `firwin`
- Produces identical filter coefficients to TensorPAC
- Filter lengths match exactly between libraries

### 2. Matched Cycle Parameters
- **Phase filters**: 3 cycles (default)
- **Amplitude filters**: 6 cycles (default)
- This matches TensorPAC's default `cycle=(3, 6)` parameter

### 3. Filter Design Differences

| Feature | Standard gPAC | TensorPAC-Compatible gPAC | TensorPAC |
|---------|---------------|---------------------------|-----------|
| Filter Implementation | scipy.firwin | Custom fir1 | Custom fir1 |
| Default Cycles | 3 for both | (3, 6) | (3, 6) |
| Filter Length | Odd (to_odd) | Matches TensorPAC | Variable |
| Zero-padding | Yes (FFT efficiency) | Minimal | No (filtfilt) |
| GPU Optimization | Full | Reduced | N/A |

## Usage

### Basic Example

```python
import gpac
import torch

# Create TensorPAC-compatible PAC model
model = gpac.PAC_TensorPACCompatible(
    seq_len=1024,
    fs=512,
    pha_n_bands=10,
    amp_n_bands=10,
    n_perm=None,
    trainable=False
)

# Or use the convenience function
model = gpac.create_tensorpac_compatible_pac(
    seq_len=1024,
    fs=512
)

# Process signal (same API as standard gPAC)
signal = torch.randn(1, 1, 1, 1024)
pac_result = model(signal)
```

### Comparison with Standard gPAC

```python
# Standard gPAC
standard_model = gpac.PAC(
    seq_len=1024,
    fs=512,
    filter_cycle=3  # Single cycle parameter
)

# TensorPAC-compatible gPAC
compatible_model = gpac.PAC_TensorPACCompatible(
    seq_len=1024,
    fs=512,
    filter_cycle_pha=3,  # Phase cycle
    filter_cycle_amp=6   # Amplitude cycle
)
```

## Implementation Details

### Filter Resolution
Both implementations achieve similar frequency resolution:
- Resolution ≈ fs / (cycle × f_low) Hz
- Lower frequencies → longer filters → better frequency resolution
- Higher frequencies → shorter filters → better temporal resolution

### Example Filter Lengths (fs=512 Hz, seq_len=2048)

| Band | Frequency Range | Filter Length |
|------|----------------|---------------|
| Delta | 2-4 Hz | 683 samples |
| Theta | 6-10 Hz | 256 samples |
| Alpha | 8-12 Hz | 193 samples |
| Beta | 13-30 Hz | 118 samples |
| Gamma | 60-120 Hz | 25 samples |

### Trade-offs

**GPU Efficiency vs. Compatibility:**
- Standard gPAC uses zero-padding for FFT efficiency on GPU
- TensorPAC-compatible mode uses minimal padding to match TensorPAC
- This may slightly reduce GPU performance but ensures exact compatibility

**Edge Handling:**
- gPAC uses convolution with 'same' padding
- TensorPAC uses filtfilt with different edge handling
- Results may differ slightly at signal edges

## When to Use TensorPAC-Compatible Mode

Use this mode when:
1. **Comparing results** directly with TensorPAC
2. **Reproducing published results** that used TensorPAC
3. **Validating algorithms** across implementations
4. **Benchmarking performance** with matched parameters

Use standard gPAC when:
1. **Maximum GPU performance** is needed
2. **Training neural networks** with PAC (trainable mode)
3. **Processing large batches** of data
4. **Real-time applications** requiring speed

## Validation Results

Filter comparison shows:
- Identical filter lengths between TensorPAC and compatible mode
- Nearly identical frequency responses
- Slight differences in PAC results due to implementation details

The correlation between results varies depending on the signal, but the implementations are functionally equivalent for most practical purposes.

## References

- TensorPAC: https://github.com/EtienneCmb/tensorpac
- Original fir1 implementation adapted from TensorPAC's spectral module