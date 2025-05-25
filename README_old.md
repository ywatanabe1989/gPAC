<!-- ---
!-- Timestamp: 2025-05-25 10:18:09
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/README.md
!-- --- -->

# gPAC: GPU-Accelerated Phase-Amplitude Coupling

`gPAC` is a PyTorch-based package for efficient computation of Phase-Amplitude Coupling (PAC) using Modulation Index (MI) with GPU acceleration.

## Key Features

- **GPU Acceleration**: 5-100x faster Modulation Index (MI) computation via PyTorch/CUDA
- **TensorPAC Compatibility**: Uses identical filter design (3 cycles for phase, 6 for amplitude) for easy comparison
- **Differentiable Filters**: Optional gradient flow for integration with deep learning models
- **Synthetic Data Generation**: Built-in tools for generating test signals with known PAC properties
- **Statistical Analysis**: Permutation testing and surrogate distributions for validation
- **Return Full Distributions**: Access complete surrogate data for custom statistical analyses

## Demo

![PAC Analysis Demo](docs/demo.gif)

The animation above shows gPAC computing Modulation Index (MI) with different frequency resolutions. Each frequency band combination is **independent**, making PAC computation perfectly suited for GPU parallelization.

## Why GPU Acceleration Works

![Parallelization Diagram](docs/parallelization_diagram.png)

Each frequency band combination requires no communication with other calculations, allowing thousands of GPU cores to compute different frequency pairs simultaneously.

## Comparison with tensorpac

![PAC Analysis Comparison](docs/pac_analysis.gif)

### Demo Image
- **Top**: Input Synthetic Signal with known PAC coupling
- **Bottom left**: PAC calculated by gPAC
- **Bottom center**: PAC calculated by Tensorpac  
- **Bottom right**: Difference (PAC calculated by gPAC - that of Tensorpac)

Both methods calculate Modulation Index (MI) using identical settings, demonstrating compatibility while gPAC provides significant speed improvements through GPU acceleration. Red dashed boxes show ground truth coupling ranges.

## Quick Start

```bash
# Installation
git clone https://github.com/ywatanabe1989/gPAC.git
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

<!-- EOF -->