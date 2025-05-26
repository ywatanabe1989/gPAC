# Differentiable Bucketize Functions

## Overview

The standard `torch.bucketize` function is non-differentiable because it performs hard assignment of values to bins. This breaks gradient flow in neural networks and optimization pipelines. The `gpac._differentiable_bucketize` module provides fully differentiable alternatives using soft binning techniques.

## Key Features

- **Gradient Flow**: Maintains complete gradient flow through discretization operations
- **Multiple Methods**: Supports softmax, sigmoid, and Gaussian soft binning
- **Circular Support**: Special handling for circular data (e.g., phase values)
- **Drop-in Replacement**: Can replace `torch.bucketize` with minimal code changes
- **Temperature Control**: Adjustable hardness/softness of binning

## Functions

### `differentiable_bucketize`

Returns soft bin assignments (weights) for each input value.

```python
import torch
from gpac import differentiable_bucketize

x = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
boundaries = torch.tensor([0., 1., 2., 3.])

# Returns shape (3, 3) - 3 inputs, 3 bins
soft_bins = differentiable_bucketize(x, boundaries, temperature=0.1)
```

### `differentiable_bucketize_indices`

Returns weighted bin indices (continuous values) instead of weights.

```python
# Returns shape (3,) - continuous indices
soft_indices = differentiable_bucketize_indices(x, boundaries, temperature=0.1)
# Result: approximately [0.5, 1.5, 2.5]
```

### `differentiable_phase_binning`

Specialized function for phase data with circular boundary handling.

```python
phases = torch.tensor([-3.14, 0., 3.14], requires_grad=True)
soft_bins = differentiable_phase_binning(phases, n_bins=18)
```

## Methods

### Softmax Method (default)
- Uses negative distance as logits
- Naturally handles multi-dimensional inputs
- Good for general use cases

### Sigmoid Method
- Based on cumulative probabilities
- Can be more intuitive for 1D data
- Exact at boundaries with `right` parameter

### Gaussian Method
- Uses Gaussian kernel weighting
- Smooth and continuous
- Good for data with natural variance

## Temperature Parameter

The `temperature` parameter controls the hardness of binning:

- **Low temperature (< 0.1)**: Near-hard binning, concentrated weights
- **Medium temperature (0.5-1.0)**: Balanced soft binning
- **High temperature (> 2.0)**: Very soft binning, distributed weights

## Integration with ModulationIndex

The differentiable bucketize can be integrated into the ModulationIndex calculation:

```python
class DifferentiableModulationIndex(nn.Module):
    def _phase_to_masks(self, pha, phase_bin_cutoffs):
        # Replace hard bucketize with soft version
        return differentiable_bucketize(
            pha, phase_bin_cutoffs, 
            temperature=0.1,
            circular=True
        )
```

## Example: Gradient-Based Optimization

```python
# Optimize phase distribution for maximum entropy
class PhaseOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phases = nn.Parameter(torch.randn(100))
        
    def forward(self):
        # Soft phase binning
        soft_bins = differentiable_phase_binning(self.phases, n_bins=18)
        
        # Compute entropy of phase distribution
        bin_probs = soft_bins.mean(dim=0)
        entropy = -(bin_probs * torch.log(bin_probs + 1e-9)).sum()
        
        return -entropy  # Minimize negative entropy

# Train with standard PyTorch
model = PhaseOptimizer()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(100):
    loss = model()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Performance Considerations

- Soft binning is more computationally expensive than hard binning
- Use appropriate temperature for your use case
- Consider using `@torch.jit.script` for production code
- The module version (`DifferentiableBucketize`) can be more efficient for repeated use

## Mathematical Details

### Softmax Method
```
weight[i] = exp(-distance_to_bin[i] / temperature) / sum(exp(-distance_to_bin / temperature))
```

### Sigmoid Method
```
P(bin_i) = sigmoid((x - boundary[i]) / T) - sigmoid((x - boundary[i+1]) / T)
```

### Gaussian Method
```
weight[i] = exp(-distance²_to_center[i] / (2 * temperature²)) / normalization
```

## Comparison with Standard Bucketize

| Feature | torch.bucketize | differentiable_bucketize |
|---------|-----------------|-------------------------|
| Differentiable | ❌ | ✅ |
| Output type | Integer indices | Float weights/indices |
| Gradient flow | Blocked | Maintained |
| Speed | Fast | Slower |
| Use case | Inference | Training/Optimization |