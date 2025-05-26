# TensorPAC Frequency Band Handling Documentation

## Overview

TensorPAC has a unique and sometimes confusing approach to frequency band specification that can lead to unexpected results when comparing with other PAC implementations like gPAC.

## Key Findings

### 1. String-Based Frequency Configuration

TensorPAC allows frequency bands to be specified using predefined string arguments that **completely override** any explicit frequency ranges:

```python
# String configurations available:
- 'lres' (low resolution): 10 bands
- 'mres' (medium resolution): 30 bands  
- 'hres' (high resolution): 50 bands
- 'demon': 70 bands
- 'hulk': 100 bands
```

**Critical Issue**: When using these strings, TensorPAC ignores standard frequency ranges and uses its own predefined ranges:

```python
# Expected when using strings:
f_pha = (2, 20)   # You might expect 2-20 Hz
f_amp = (60, 160) # You might expect 60-160 Hz

# What TensorPAC actually uses:
f_pha = (1.5, 25)   # Actually 1.5-25 Hz
f_amp = (52.5, 180) # Actually 52.5-180 Hz
```

### 2. Band Definition Formula

When using string configurations, TensorPAC creates overlapping frequency bands using specific formulas:

- **Phase bands**: `[f - f/4, f + f/4]` where f is the center frequency
- **Amplitude bands**: `[f - f/8, f + f/8]` where f is the center frequency

This creates overlapping bands, unlike the sequential non-overlapping bands typically used in other implementations.

### 3. Explicit Band Specification

To match other implementations, you must provide explicit frequency band pairs:

```python
# Create non-overlapping sequential bands (gPAC style)
pha_vec = np.linspace(2, 20, 11)  # 11 points = 10 bands
amp_vec = np.linspace(60, 160, 11)

# Convert to band pairs
f_pha = np.c_[pha_vec[:-1], pha_vec[1:]]  # [[2, 3.8], [3.8, 5.6], ...]
f_amp = np.c_[amp_vec[:-1], amp_vec[1:]]  # [[60, 70], [70, 80], ...]

# Initialize Pac with explicit bands
pac = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
```

## Comparison Results

### String-based vs Explicit Configuration

| Configuration | Phase Bands | Amplitude Bands | Frequency Range |
|--------------|-------------|-----------------|-----------------|
| 'mres' string | 30 overlapping | 30 overlapping | 1.5-25 Hz / 52.5-180 Hz |
| Explicit 10x10 | 10 sequential | 10 sequential | 2-20 Hz / 60-160 Hz |

### Impact on PAC Values

- String-based 'mres': Max PAC = 0.413 (30x30 matrix)
- Explicit 10x10: Max PAC = 0.388 (10x10 matrix)
- Different peak locations due to different band definitions

## Recommendations for gPAC-TensorPAC Comparison

1. **Always use explicit frequency bands** when comparing gPAC with TensorPAC
2. **Avoid string configurations** like 'mres' as they use different frequency ranges
3. **Create matching sequential bands** for both implementations:

```python
# For gPAC
pac_gpac, _, _ = calculate_pac(
    signal,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=10
)

# For TensorPAC (matching configuration)
pha_bands = [[2+(i*1.8), 2+(i+1)*1.8] for i in range(10)]
amp_bands = [[60+(i*10), 60+(i+1)*10] for i in range(10)]
pac_obj = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands)
```

## Known Issues

1. **Poor correlation between implementations**: Even with matching frequency bands, gPAC and TensorPAC show poor correlation (r=-0.149) due to:
   - Different filter implementations
   - Different normalization approaches
   - Different Modulation Index calculations

2. **Value scale differences**: gPAC values are typically 4-5x smaller than TensorPAC values
   - gPAC max: ~0.093
   - TensorPAC max: ~0.388

3. **Peak location mismatch**: The two implementations often identify different frequency pairs as having maximum coupling

## Best Practices

1. Document which frequency configuration you're using
2. Be explicit about frequency bands - don't rely on defaults
3. When publishing results, specify:
   - Exact frequency ranges used
   - Number of bands
   - Whether bands are overlapping or sequential
   - Implementation used (gPAC vs TensorPAC)

## Code Example: Proper Comparison

```python
import numpy as np
from tensorpac import Pac
from gpac import calculate_pac

# Define matching parameters
n_pha_bands = 10
n_amp_bands = 10
pha_range = (2, 20)
amp_range = (60, 160)

# Create frequency vectors
pha_edges = np.linspace(*pha_range, n_pha_bands + 1)
amp_edges = np.linspace(*amp_range, n_amp_bands + 1)

# For TensorPAC - explicit bands
pha_bands = np.c_[pha_edges[:-1], pha_edges[1:]]
amp_bands = np.c_[amp_edges[:-1], amp_edges[1:]]
pac_tp = Pac(idpac=(2, 0, 0), f_pha=pha_bands, f_amp=amp_bands)

# For gPAC - range specification
pac_gp, _, _ = calculate_pac(
    signal,
    fs=fs,
    pha_start_hz=pha_range[0],
    pha_end_hz=pha_range[1],
    pha_n_bands=n_pha_bands,
    amp_start_hz=amp_range[0],
    amp_end_hz=amp_range[1],
    amp_n_bands=n_amp_bands
)
```

## References

- TensorPAC source: `tensorpac/utils.py` - `pac_vec()` function
- Bahramisharif et al. 2013 - Original frequency band definitions
- idpac=(2,0,0) - Modulation Index (Tort method), no surrogates, no normalization