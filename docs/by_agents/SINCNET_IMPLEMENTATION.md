# SincNet-Style Implementation in gPAC

## Overview

We have successfully replaced the previous differentiable bandpass filter implementation with a SincNet-style approach that allows learning optimal frequency bands through backpropagation.

## Key Features

### 1. **Learnable Frequency Boundaries**
- Each filter is parameterized by its low and high cutoff frequencies
- Parameters are initialized using mel-scale or linear spacing
- Frequencies are constrained to valid ranges using absolute values and clamping

### 2. **Sinc-based Filtering**
- Uses parametric sinc functions in the time domain
- Bandpass filters created as: `highpass_sinc - lowpass_sinc`
- Applies Hamming window for better frequency response
- Energy normalization to ensure consistent filter gains

### 3. **Frequency-dependent Normalization**
Three normalization modes available:
- `'std'`: Standard deviation normalization (default)
- `'freq_weighted'`: Compensates for frequency-dependent energy variations
- `'bandwidth'`: Normalizes by bandwidth to ensure equal contribution

### 4. **Regularization**
Built-in regularization losses to control filter behavior:
- **Overlap penalty**: Encourages non-overlapping adjacent bands
- **Bandwidth control**: Prevents too narrow or too wide bands
- **Frequency ordering**: Ensures bands are ordered by frequency

### 5. **Hard Constraints**
Method to apply hard constraints after optimization:
- Ensures positive frequencies
- Maintains minimum bandwidth
- Prevents exceeding Nyquist frequency

## Usage Example

```python
from gpac import PAC

# Create PAC model with trainable filters
pac_model = PAC(
    seq_len=1024,
    fs=256,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=4,
    amp_start_hz=30,
    amp_end_hz=100,
    amp_n_bands=4,
    trainable=True,  # Enables SincNet-style learnable filters
    n_perm=0
)

# Training loop
optimizer = torch.optim.Adam(pac_model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    # Forward pass
    pac_output = pac_model(signals)
    pac_values = pac_output['pac']
    
    # Compute task-specific loss
    loss = compute_loss(pac_values, targets)
    
    # Add filter regularization
    reg_losses = pac_model.bandpass.get_regularization_loss()
    loss = loss + reg_losses['total']
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Apply hard constraints
    pac_model.bandpass.constrain_parameters()
    
    # Inspect learned bands
    bands = pac_model.bandpass.get_filter_banks()
    print(f"Phase bands: {bands['pha_bands']}")
    print(f"Amplitude bands: {bands['amp_bands']}")
```

## Advantages over Previous Implementation

1. **More flexible**: Can learn arbitrary frequency boundaries, not just center frequencies
2. **Better interpretability**: Direct control over band edges
3. **Improved optimization**: Constraints and regularization prevent invalid configurations
4. **Frequency-aware**: Handles high-frequency fluctuations appropriately
5. **Neuroscience-aligned**: Matches how frequency bands are typically defined in PAC analysis

## Implementation Details

### Files Modified/Created:
- `/src/gpac/_Filters/_DifferentiableBandPassFilter.py`: Complete rewrite with SincNet approach
- `/src/gpac/_Filters/_StaticBandPassFilter.py`: Updated to use same sinc-based approach
- `/src/gpac/_BandPassFilter.py`: Updated wrapper to expose new functionality
- `/tests/test_pac_trainability.py`: Comprehensive trainability test
- `/tests/test_pac_trainability_simple.py`: Simple demonstration

### Key Classes:
- `DifferentiableBandPassFilter`: SincNet-style learnable filter
- `StaticBandPassFilter`: Non-learnable version using same sinc approach
- `BandPassFilter`: Unified interface switching between static/trainable modes

## Future Enhancements

1. Add more sophisticated initialization strategies (e.g., based on prior knowledge)
2. Implement adaptive learning rates for different frequency ranges
3. Add support for non-uniform frequency spacing
4. Explore alternative window functions (Kaiser, Blackman, etc.)
5. Add visualization tools for learned filter banks