# gPAC API Reference

## Overview

gPAC (GPU-accelerated Phase-Amplitude Coupling) is a PyTorch-based implementation for computing Phase-Amplitude Coupling (PAC) in neural signals with GPU acceleration support.

## Quick Start

```python
import torch
from gpac import calculate_pac

# Generate sample data (batch_size=1, channels=1, segments=1, time_points=1024)
signal = torch.randn(1, 1, 1, 1024)

# Calculate PAC
pac_values, pha_freqs, amp_freqs = calculate_pac(
    signal,
    fs=256,  # Sampling frequency in Hz
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=10
)
```

## Main Functions

### calculate_pac

The primary function for calculating Phase-Amplitude Coupling.

```python
calculate_pac(
    signal: torch.Tensor | np.ndarray,
    fs: float,
    pha_start_hz: float = 2.0,
    pha_end_hz: float = 20.0,
    pha_n_bands: int = 50,
    amp_start_hz: float = 60.0,
    amp_end_hz: float = 160.0,
    amp_n_bands: int = 30,
    n_perm: Optional[int] = None,
    trainable: bool = False,
    fp16: bool = False,
    amp_prob: bool = False,
    mi_n_bins: int = 18,
    filter_cycle: int = 3,
    device: Optional[str | torch.device] = None,
    chunk_size: Optional[int] = None,
    average_channels: bool = False,
    return_dist: bool = False,
    v01_mode: bool = False,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]
```

#### Parameters

- **signal** : torch.Tensor or np.ndarray
  - Input signal with shape (batch_size, channels, segments, time_points)
  - Can be 1D, 2D, 3D, or 4D (automatically reshaped to 4D)

- **fs** : float
  - Sampling frequency in Hz

- **pha_start_hz** : float, default=2.0
  - Start frequency for phase bands (Hz)

- **pha_end_hz** : float, default=20.0
  - End frequency for phase bands (Hz)

- **pha_n_bands** : int, default=50
  - Number of phase frequency bands

- **amp_start_hz** : float, default=60.0
  - Start frequency for amplitude bands (Hz)

- **amp_end_hz** : float, default=160.0
  - End frequency for amplitude bands (Hz)

- **amp_n_bands** : int, default=30
  - Number of amplitude frequency bands

- **n_perm** : int or None, default=None
  - Number of permutations for surrogate testing
  - If None, no statistical testing is performed

- **trainable** : bool, default=False
  - Use trainable filters for deep learning integration

- **fp16** : bool, default=False
  - Use half-precision for faster computation

- **amp_prob** : bool, default=False
  - Return amplitude probability distribution instead of MI

- **mi_n_bins** : int, default=18
  - Number of phase bins for Modulation Index calculation

- **filter_cycle** : int, default=3
  - Filter bandwidth in cycles

- **device** : str or torch.device or None, default=None
  - Computation device ('cuda', 'cpu', or torch.device)
  - If None, automatically selects GPU if available

- **chunk_size** : int or None, default=None
  - Process in chunks to save memory
  - If None, processes all data at once

- **average_channels** : bool, default=False
  - Average PAC values across channels

- **return_dist** : bool, default=False
  - Return surrogate distribution (requires n_perm > 0)

- **v01_mode** : bool, default=False
  - Use v01 depthwise convolution for better TensorPAC compatibility

#### Returns

Standard return (return_dist=False):
- **pac_values** : torch.Tensor
  - PAC values with shape (batch_size, channels, pha_n_bands, amp_n_bands)
- **pha_freqs** : np.ndarray
  - Phase frequency centers with shape (pha_n_bands,)
- **amp_freqs** : np.ndarray
  - Amplitude frequency centers with shape (amp_n_bands,)

With surrogate distribution (return_dist=True and n_perm > 0):
- **pac_values** : torch.Tensor
  - Z-scored PAC values
- **surrogate_dist** : torch.Tensor
  - Surrogate distribution with shape (n_perm, batch_size, channels, pha_n_bands, amp_n_bands)
- **pha_freqs** : np.ndarray
- **amp_freqs** : np.ndarray

## Core Classes

### PAC

Main class for Phase-Amplitude Coupling computation.

```python
class PAC(nn.Module):
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_start_hz: float = 2.0,
        pha_end_hz: float = 20.0,
        pha_n_bands: int = 50,
        amp_start_hz: float = 60.0,
        amp_end_hz: float = 160.0,
        amp_n_bands: int = 30,
        n_perm: Optional[int] = None,
        trainable: bool = False,
        fp16: bool = False,
        mi_n_bins: int = 18,
        filter_cycle_pha: int = 3,
        filter_cycle_amp: int = 6,
    )
```

### BandPassFilter

Bandpass filter implementation with TensorPAC compatibility.

```python
class BandPassFilter(nn.Module):
    def __init__(
        self,
        pha_bands: List[List[float]],
        amp_bands: List[List[float]],
        fs: float,
        seq_len: int,
        fp16: bool = False,
        cycle_pha: int = 3,
        cycle_amp: int = 6,
        filtfilt_mode: bool = False,
        edge_mode: Optional[str] = None,
        v01_mode: bool = False,
    )
```

### Hilbert

Hilbert transform for extracting instantaneous phase and amplitude.

```python
class Hilbert(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int = -1,
        fp16: bool = False,
    )
```

### ModulationIndex

Standard Modulation Index calculation (Tort et al., 2010).

```python
class ModulationIndex(nn.Module):
    def __init__(
        self,
        n_bins: int = 18,
        differentiable: bool = False,
    )
```

### DifferentiableModulationIndex

Differentiable version for deep learning applications.

```python
class DifferentiableModulationIndex(nn.Module):
    def __init__(
        self,
        n_bins: int = 18,
        sigma: float = 0.1,
    )
```

## TensorPAC Compatibility

For comparing with TensorPAC results, use the compatibility layer:

```python
from gpac._calculate_gpac_tensorpac_compat import calculate_pac_tensorpac_compat

# This applies scaling to match TensorPAC value ranges
pac_compat = calculate_pac_tensorpac_compat(
    signal, fs=fs,
    pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
    amp_start_hz=60, amp_end_hz=160, amp_n_bands=10
)
```

### Important Notes on TensorPAC Compatibility

1. **Frequency Band Specification**: When comparing with TensorPAC, use explicit frequency bands rather than string configurations ('mres', 'hres') which override parameters.

2. **Value Scale**: gPAC values are typically 4-5x smaller than TensorPAC values due to different normalization approaches.

3. **Correlation**: With the compatibility layer, correlation improves from ~0.336 to ~0.676.

## Examples

### Basic PAC Calculation

```python
import torch
import numpy as np
from gpac import calculate_pac
import matplotlib.pyplot as plt

# Generate synthetic signal with PAC
fs = 1000  # Hz
duration = 10  # seconds
t = np.arange(0, duration, 1/fs)

# Phase signal (5 Hz)
phase_freq = 5
phase_signal = np.sin(2 * np.pi * phase_freq * t)

# Amplitude signal (80 Hz) modulated by phase
amp_freq = 80
modulation_depth = 0.8
amplitude = 1 + modulation_depth * np.sin(2 * np.pi * phase_freq * t)
amp_signal = amplitude * np.sin(2 * np.pi * amp_freq * t)

# Combined signal
signal = phase_signal + 0.5 * amp_signal
signal_tensor = torch.from_numpy(signal).float().reshape(1, 1, 1, -1)

# Calculate PAC
pac_values, pha_freqs, amp_freqs = calculate_pac(
    signal_tensor,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=20,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=20
)

# Plot results
plt.figure(figsize=(10, 8))
plt.imshow(pac_values[0, 0].numpy(), aspect='auto', origin='lower',
           extent=[amp_freqs[0], amp_freqs[-1], pha_freqs[0], pha_freqs[-1]])
plt.xlabel('Amplitude Frequency (Hz)')
plt.ylabel('Phase Frequency (Hz)')
plt.title('Phase-Amplitude Coupling')
plt.colorbar(label='PAC Strength')
plt.show()
```

### Statistical Testing with Permutations

```python
# Calculate PAC with permutation testing
pac_values, pha_freqs, amp_freqs = calculate_pac(
    signal_tensor,
    fs=fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=120,
    amp_n_bands=10,
    n_perm=200  # 200 permutations for statistical testing
)

# pac_values now contains z-scores
significant = pac_values > 2.0  # z > 2 roughly corresponds to p < 0.05
```

### Batch Processing

```python
# Process multiple signals at once
batch_size = 16
n_channels = 4
signal_batch = torch.randn(batch_size, n_channels, 1, 10000)

# Process with chunking for memory efficiency
pac_values, pha_freqs, amp_freqs = calculate_pac(
    signal_batch,
    fs=500,
    chunk_size=4,  # Process 4 samples at a time
    average_channels=True  # Average across channels
)

# Result shape: (16, 1, 50, 30) due to channel averaging
```

### Deep Learning Integration

```python
# Use trainable filters for end-to-end learning
model = PAC(
    seq_len=1024,
    fs=256,
    trainable=True,  # Learnable frequency bands
    pha_n_bands=5,
    amp_n_bands=5
)

# Can be used in a neural network
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    pac_result = model(signal_batch)
    loss = some_loss_function(pac_result['mi'], target)
    loss.backward()
    optimizer.step()
```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for ~8x speedup
   ```python
   if torch.cuda.is_available():
       signal = signal.cuda()
   ```

2. **Batch Processing**: Process multiple signals together for ~13x efficiency
   ```python
   # Good: Process 32 signals at once
   batch_signal = torch.stack([signal1, signal2, ..., signal32])
   
   # Bad: Process one by one
   for sig in signals:
       pac = calculate_pac(sig, ...)
   ```

3. **Memory Management**: Use chunk_size for large datasets
   ```python
   # For 1000 signals, process in chunks of 100
   pac_values, _, _ = calculate_pac(
       large_batch,
       fs=fs,
       chunk_size=100
   )
   ```

4. **Half Precision**: Use fp16=True for 2x memory savings (with slight precision loss)
   ```python
   pac_values, _, _ = calculate_pac(
       signal,
       fs=fs,
       fp16=True
   )
   ```

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**
   - Ensure input is 4D: (batch, channels, segments, time)
   - Use `signal.reshape(1, 1, 1, -1)` for 1D signals

2. **Frequency Warnings**
   - Amplitude frequencies exceeding Nyquist limit are automatically adjusted
   - Check sampling rate vs requested frequencies

3. **Memory Errors**
   - Reduce batch size or use chunk_size parameter
   - Consider using fp16=True

4. **Poor Correlation with TensorPAC**
   - Use explicit frequency bands, not string configs
   - Apply compatibility layer for value scaling
   - Check filter parameters match (cycle=3 default)

## References

- Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010). Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. Journal of neurophysiology, 104(2), 1195-1210.

- Canolty, R. T., & Knight, R. T. (2010). The functional role of cross-frequency coupling. Trends in cognitive sciences, 14(11), 506-515.