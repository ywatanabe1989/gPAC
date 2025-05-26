# TensorPAC Compatibility Guide

## Overview

gPAC provides a compatibility module to match TensorPAC's behavior and output scale. This is important because:

1. **Scale Difference**: gPAC values are typically 4-12x smaller than TensorPAC values
2. **Frequency Ranges**: TensorPAC uses specific default frequency ranges
3. **Band Configuration**: Common configurations use many frequency bands (e.g., 50x30)

## Quick Start

```python
from gpac import calculate_pac_tensorpac_compat

# Calculate PAC with TensorPAC-compatible scaling
pac, f_pha, f_amp = calculate_pac_tensorpac_compat(signal, fs=1000)
```

## Available Configurations

### 'compatible' (Default, Recommended)
- **Phase bands**: 50
- **Amplitude bands**: 30
- **Phase range**: 1.5-25 Hz
- **Amplitude range**: 52.5-180 Hz
- **Scale factor**: 12x

### 'hres' (High Resolution)
- **Phase bands**: 50
- **Amplitude bands**: 50
- **Phase range**: 1.5-25 Hz
- **Amplitude range**: 52.5-180 Hz
- **Scale factor**: 8x

### 'medium'
- **Phase bands**: 30
- **Amplitude bands**: 30
- **Phase range**: 1.5-25 Hz
- **Amplitude range**: 52.5-180 Hz
- **Scale factor**: 6.5x

### 'standard'
- **Phase bands**: 10
- **Amplitude bands**: 10
- **Phase range**: 2-20 Hz
- **Amplitude range**: 60-160 Hz
- **Scale factor**: 5x

## Usage Examples

### Basic Usage

```python
import numpy as np
from gpac import calculate_pac_tensorpac_compat

# Create test signal
fs = 1000  # Sampling frequency
t = np.arange(2000) / fs
signal = (1 + 0.5 * np.sin(2*np.pi*5*t)) * np.sin(2*np.pi*70*t)

# Calculate PAC with default 50x30 configuration
pac, pha_freqs, amp_freqs = calculate_pac_tensorpac_compat(signal, fs)

print(f"PAC shape: {pac.shape}")  # (50, 30)
print(f"Max PAC: {pac.max():.3f}")  # Scaled to match TensorPAC
```

### Using Different Configurations

```python
# High-resolution 50x50
pac_hres, f_pha, f_amp = calculate_pac_tensorpac_compat(
    signal, fs, config='hres'
)

# Standard 10x10
pac_std, f_pha, f_amp = calculate_pac_tensorpac_compat(
    signal, fs, config='standard'
)
```

### Getting Unscaled Values

```python
# Get both scaled and unscaled values
pac_scaled, f_pha, f_amp, pac_raw = calculate_pac_tensorpac_compat(
    signal, fs, return_unscaled=True
)

print(f"Raw gPAC max: {pac_raw.max():.6f}")
print(f"Scaled max: {pac_scaled.max():.6f}")
print(f"Scale factor: {pac_scaled.max() / pac_raw.max():.1f}x")
```

### Custom Scaling

```python
# Use custom scale factor
pac, f_pha, f_amp = calculate_pac_tensorpac_compat(
    signal, fs, custom_scale=10.0
)
```

### Comparing with TensorPAC

```python
from gpac import compare_with_tensorpac

# Requires TensorPAC to be installed
results = compare_with_tensorpac(signal, fs, config='compatible')

if 'error' not in results:
    print(f"TensorPAC max: {results['tensorpac_max']:.3f}")
    print(f"gPAC scaled max: {results['gpac_max_scaled']:.3f}")
    print(f"Actual scale factor: {results['actual_scale_factor']:.1f}x")
```

## Important Notes

1. **Scale Factors**: The scale factors are empirically determined and may vary slightly depending on the signal characteristics.

2. **Frequency Ranges**: The default ranges (1.5-25 Hz for phase, 52.5-180 Hz for amplitude) match TensorPAC's 'hres' configuration.

3. **Filter Settings**: The compatibility module uses `filtfilt_mode=True` for better filter matching with TensorPAC.

4. **Performance**: Using many frequency bands (e.g., 50x30) requires more computation. Consider using fewer bands for faster processing if high resolution is not needed.

## Migration from TensorPAC

If you're migrating from TensorPAC:

```python
# TensorPAC code:
from tensorpac import Pac
pac_obj = Pac(idpac=(2,0,0), f_pha='hres', f_amp='hres')
pac_tp = pac_obj.filterfit(fs, signal)

# Equivalent gPAC code:
from gpac import calculate_pac_tensorpac_compat
pac_gp, f_pha, f_amp = calculate_pac_tensorpac_compat(
    signal, fs, config='hres'
)
```

## Technical Details

The compatibility module addresses several differences between gPAC and TensorPAC:

1. **Filter Implementation**: Different filter designs lead to slightly different frequency responses
2. **Amplitude Extraction**: Minor differences in how amplitude is computed from analytic signals
3. **Normalization**: Subtle differences in the modulation index calculation
4. **GPU Acceleration**: gPAC uses PyTorch for GPU support, while TensorPAC uses NumPy

Despite these differences, the compatibility module ensures that:
- PAC patterns are preserved
- Peak locations are consistent
- Values are on the same scale for comparison

## References

- TensorPAC: Combrisson et al. (2020). Tensorpac: An open-source Python toolbox for tensor-based phase-amplitude coupling measurement in electrophysiological brain signals. PLoS Comput Biol.
- Tort et al. (2010). Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. J Neurophysiol.