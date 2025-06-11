# gpu-pac v0.1.0 Release Notes

🎉 **First Release of gpu-pac** - GPU-accelerated Phase-Amplitude Coupling

## 📦 Installation

```bash
pip install gpu-pac
```

## 🚀 Key Features

### Performance
- **341.8x speedup** over TensorPAC on real benchmarks
- Smart memory management with auto/chunked/sequential strategies
- Multi-GPU support with DataParallel
- Automatic mixed precision (fp16/fp32) support

### Core Functionality
- **Modulation Index (MI)** calculation with full GPU acceleration
- **Unbiased surrogate generation** using full range time shifts
- **Z-score normalization** with permutation testing (n_perm)
- **Trainable bandpass filters** with learnable frequency selection
- High correlation with TensorPAC (0.81 ± 0.04)

### API Features
- Full PyTorch integration with gradient support
- Access to frequency bands as tensors (`pha_bands_hz`, `amp_bands_hz`)
- Memory-efficient chunked processing for large datasets
- Comprehensive configuration options

## 📊 Benchmarks

| Parameter | Performance |
|-----------|-------------|
| Batch Size Scaling | Linear up to 1024 |
| Time Duration | Handles 60+ seconds efficiently |
| Frequency Bands | 30+ bands with minimal overhead |
| Memory Strategy | Auto-selects optimal approach |

## 🔧 Basic Usage

```python
import torch
from gpac import PAC

# Initialize PAC calculator
pac = PAC(
    seq_len=5120,
    fs=512,
    pha_range_hz=(4, 30),
    amp_range_hz=(60, 150),
    pha_n_bands=10,
    amp_n_bands=20,
    n_perm=200  # For z-score calculation
)

# Compute PAC
signal = torch.randn(32, 64, 5120)  # (batch, channels, time)
results = pac(signal)

# Access results
pac_values = results['pac']       # Phase-amplitude coupling values
pac_z_scores = results['pac_z']   # Z-normalized PAC values
```

## 📝 What's Included

- Core PAC computation with MI (Modulation Index)
- Bandpass filtering with FIR filters
- Hilbert transform implementation
- Synthetic PAC data generation utilities
- Comprehensive test suite
- Detailed examples and benchmarks

## 🔗 Links

- **PyPI**: https://pypi.org/project/gpu-pac/
- **GitHub**: https://github.com/ywatanabe1989/gPAC
- **Documentation**: See README.md for detailed usage

## 🙏 Acknowledgments

This package implements GPU-accelerated computation of Phase-Amplitude Coupling, 
providing significant speedups for neuroscience research applications.

---

**Note**: This is the first release with a new package name `gpu-pac` (previously 
planned as `gpac` which was already taken on PyPI).