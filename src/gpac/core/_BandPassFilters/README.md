<!-- ---
!-- Timestamp: 2025-06-08 09:17:13
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/src/gpac/core/_BandPassFilters/README.md
!-- --- -->

# GPU Bandpass Filters

High-performance GPU-accelerated bandpass filters for Phase-Amplitude Coupling (PAC) analysis.

## Quick Start

```python
import torch
from gpac.core._BandPassFilters import StaticBandPassFilter

# Define frequency bands
pha_bands = [[4, 8], [8, 12], [12, 20]]    # Phase bands (Hz)  
amp_bands = [[60, 80], [80, 120], [120, 160]]  # Amplitude bands (Hz)

# Create filter
filter_obj = StaticBandPassFilter(
    pha_bands_hz=pha_bands,
    amp_bands_hz=amp_bands, 
    fs=500,  # Sampling frequency
    n_cycles=4  # Filter order
)

# Apply filtering
x = torch.randn(2, 3, 1000)  # (batch, channels, samples)
filtered = filter_obj(x)     # (batch, channels, n_filters, samples)
```

## Filter Classes

### StaticBandPassFilter
- Purpose: Fixed bandpass filtering optimized for GPU acceleration
- Features: Vectorized Conv1D, adaptive filter length, zero-phase filtering
- Use case: Standard PAC analysis with known frequency bands

### PooledBandPassFilter  
- Purpose: Learnable filter selection with Gumbel Softmax
- Features: Creates filter pool then learns optimal subset selection
- Use case: Machine learning applications requiring trainable filters

## Key Features

- Vectorized Processing: All frequency bands processed simultaneously
- GPU Optimized: Full CUDA support with FP16 precision
- Adaptive Filtering: Filter length based on lowest frequency for accuracy
- Zero-Phase: Forward-backward filtering preserves phase relationships
- Device Compatible: Seamless CPU/GPU transfers with `.to(device)`

## Architecture

```
Input Signal (batch, channels, samples)
        ↓
   Filter Bank Creation (adaptive length)
        ↓  
   Vectorized Conv1D (all bands parallel)
        ↓
   Zero-Phase Filtering (forward-backward)
        ↓
Output (batch, channels, n_filters, samples)
```

## Contact

Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)

<!-- EOF -->