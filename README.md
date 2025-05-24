# gPAC: GPU-Accelerated Phase-Amplitude Coupling

`gPAC` is a PyTorch-based package for efficient computation of Phase-Amplitude Coupling (PAC) metrics with GPU acceleration.

## Key Features

- **GPU Acceleration**: 5-100x faster PAC computation via PyTorch/CUDA
- **Differentiable Filters**: Optional gradient flow for integration with deep learning models
- **Synthetic Data Generation**: Built-in tools for generating test signals with known PAC properties
- **Statistical Analysis**: Permutation testing and surrogate distributions for validation
- **Return Full Distributions**: Access complete surrogate data for custom statistical analyses

## Quick Start

```bash
# Installation
git clone https://github.com/[username]/gPAC.git
cd gPAC
pip install -e .
```

```python
# Basic usage
import torch
import gpac
import numpy as np

# Create example data (batch_size, channels, segments, time)
signal = torch.randn(2, 4, 1, 1024)

# Calculate PAC with GPU acceleration
pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
    signal=signal,
    fs=256.0,         # Sampling frequency
    pha_n_bands=10,   # Number of phase bands
    amp_n_bands=10,   # Number of amplitude bands
    device="cuda",    # Use GPU
    n_perm=200,       # Permutation testing
)
```

## Documentation

For detailed usage examples and API reference, see:
- `examples/` directory for sample scripts
- `src/gpac/README.md` for implementation details
- Docstrings in the source code for function parameters

## Contact

Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)