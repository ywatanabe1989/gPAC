# Channel Handling Guide for gPAC

This guide explains how gPAC handles multi-channel signals and common patterns to avoid shape mismatches.

## Overview

gPAC supports both single-channel and multi-channel signal processing. Understanding how channels are handled throughout the pipeline is crucial for avoiding shape mismatch errors.

## Signal Input Format

### Expected Shape
gPAC expects input signals in the following format:
```
(batch_size, n_channels, n_trials, n_samples)
```

Where:
- `batch_size`: Number of signals to process simultaneously
- `n_channels`: Number of recording channels (e.g., EEG electrodes)
- `n_trials`: Number of trials (use 1 for continuous data)
- `n_samples`: Number of time samples

### Examples

```python
import numpy as np
import gpac

# Single-channel signal
signal_1ch = np.random.randn(1, 1, 1, 1024)  # (batch=1, channels=1, trials=1, samples=1024)

# Multi-channel signal (e.g., 64-channel EEG)
signal_64ch = np.random.randn(1, 64, 1, 1024)  # (batch=1, channels=64, trials=1, samples=1024)

# Batch processing
batch_signal = np.random.randn(32, 8, 1, 1024)  # (batch=32, channels=8, trials=1, samples=1024)
```

## PAC Module Output

The PAC module outputs a 4D tensor:
```
(batch_size, n_channels, n_pha_bands, n_amp_bands)
```

This preserves channel information throughout the computation.

### Example Output Shapes

```python
# Single-channel input → Single-channel output
# Input: (1, 1, 1, 1024)
# Output: (1, 1, 20, 15)  # 20 phase bands × 15 amplitude bands

# Multi-channel input → Multi-channel output  
# Input: (1, 64, 1, 1024)
# Output: (1, 64, 20, 15)  # Preserves all 64 channels
```

## Channel Reduction Strategies

When using PAC values for downstream tasks (e.g., classification), you often need to reduce the channel dimension:

### 1. Average Across Channels
Most common approach for getting a single PAC representation:

```python
import torch
import gpac

# Multi-channel signal
signal = torch.randn(1, 64, 1, 1024)

# Compute PAC
model = gpac.PAC(seq_len=1024, fs=512, pha_n_bands=20, amp_n_bands=15)
pac_values = model(signal)  # Shape: (1, 64, 20, 15)

# Average across channels
pac_avg = pac_values.mean(dim=1)  # Shape: (1, 20, 15)
```

### 2. Channel-wise Processing
Keep channels separate for channel-specific analysis:

```python
# Process each channel independently
for ch in range(pac_values.shape[1]):
    channel_pac = pac_values[:, ch, :, :]  # Shape: (1, 20, 15)
    # Analyze channel-specific PAC patterns
```

### 3. Concatenate Channels
For models that can handle high-dimensional input:

```python
# Flatten all channels
batch_size = pac_values.shape[0]
pac_flat = pac_values.view(batch_size, -1)  # Shape: (1, 64*20*15)
```

## Common Pitfalls and Solutions

### 1. Shape Mismatch in Classifier

**Problem**: PAC module outputs 4D tensor, but classifier expects 2D input.

**Solution**: Properly handle the channel dimension:

```python
class PACClassifier(nn.Module):
    def __init__(self, n_pha_bands, n_amp_bands, n_classes):
        super().__init__()
        self.n_pha_bands = n_pha_bands
        self.n_amp_bands = n_amp_bands
        self.classifier = nn.Linear(n_pha_bands * n_amp_bands, n_classes)
    
    def forward(self, pac_values):
        # Handle 4D input (B, C, F_pha, F_amp)
        if pac_values.dim() == 4:
            # Average across channels
            pac_values = pac_values.mean(dim=1)
        
        # Reshape for classifier
        batch_size = pac_values.shape[0]
        pac_flat = pac_values.view(batch_size, -1)
        
        return self.classifier(pac_flat)
```

### 2. Single vs Multi-Channel Confusion

**Problem**: Code assumes single-channel but receives multi-channel input.

**Solution**: Always check and handle both cases:

```python
def process_pac_values(pac_values):
    """Process PAC values regardless of channel count."""
    if pac_values.shape[1] == 1:
        # Single channel - remove channel dimension
        pac_values = pac_values.squeeze(1)
    else:
        # Multi-channel - average across channels
        pac_values = pac_values.mean(dim=1)
    
    return pac_values
```

### 3. Forgetting Batch Dimension

**Problem**: Assuming input is 3D when it's actually 4D with batch dimension.

**Solution**: Always preserve batch dimension:

```python
# Wrong - loses batch dimension
if len(signal.shape) == 4:
    signal = signal[0]  # Don't do this!

# Correct - preserves batch dimension
if signal.shape[0] == 1 and len(signal.shape) == 4:
    # Process single batch item while keeping dimension
    result = model(signal)  # Still shape (1, ...)
```

## Best Practices

1. **Always document expected shapes** in your functions:
   ```python
   def compute_pac_features(signal):
       """
       Args:
           signal: Input signal of shape (batch, channels, trials, samples)
       
       Returns:
           features: PAC features of shape (batch, n_features)
       """
   ```

2. **Add shape assertions** for debugging:
   ```python
   assert signal.ndim == 4, f"Expected 4D signal, got {signal.ndim}D"
   assert signal.shape[2] == 1, "Only single-trial signals supported"
   ```

3. **Use consistent channel reduction** across your pipeline:
   ```python
   # Define once and reuse
   def reduce_channels(pac_values):
       """Standard channel reduction for this project."""
       return pac_values.mean(dim=1)
   ```

4. **Handle edge cases** explicitly:
   ```python
   if n_channels == 0:
       raise ValueError("At least one channel required")
   ```

## Working with Different Data Sources

### EEG Data (Multi-channel)
```python
# Typical EEG: 64 channels, 1000 Hz, 5 second recording
eeg_signal = np.random.randn(1, 64, 1, 5000)
pac_values = gpac.calculate_pac(eeg_signal, fs=1000)
# Output shape: (1, 64, n_pha, n_amp)
```

### LFP Data (Often single-channel)
```python
# Single LFP channel
lfp_signal = np.random.randn(1, 1, 1, 2048)
pac_values = gpac.calculate_pac(lfp_signal, fs=512)
# Output shape: (1, 1, n_pha, n_amp)
```

### MEG Data (Very high channel count)
```python
# MEG: 306 channels
meg_signal = np.random.randn(1, 306, 1, 10000)
# Consider processing in chunks to manage memory
```

## Debugging Shape Issues

When encountering shape mismatches:

1. **Print shapes at each step**:
   ```python
   print(f"Input shape: {signal.shape}")
   print(f"PAC output shape: {pac_values.shape}")
   print(f"After channel reduction: {pac_reduced.shape}")
   ```

2. **Use shape validation**:
   ```python
   expected_shape = (batch_size, n_channels, n_pha_bands, n_amp_bands)
   assert pac_values.shape == expected_shape, \
       f"Expected {expected_shape}, got {pac_values.shape}"
   ```

3. **Check intermediate tensors** in complex pipelines.

## Summary

- gPAC preserves channel information throughout computation
- Always be explicit about channel handling
- Use consistent reduction strategies
- Document expected shapes
- Test with both single and multi-channel inputs

Following these guidelines will help prevent shape mismatch errors and make your code more robust.