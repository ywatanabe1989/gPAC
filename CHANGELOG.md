# Changelog

All notable changes to gpu-pac will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-06-12

### Fixed
- Changed default `fp16` parameter from `True` to `False` in PAC module
- Added automatic float32 conversion for fp16 outputs to ensure dtype consistency
- Fixed RuntimeError when using default parameters with mixed precision tensors

### Added
- Comprehensive fp16/float32 compatibility tests (25 new tests)
- Tests verify high correlation (>0.99) between fp16 and float32 results

## [0.2.0] - 2025-06-11

### Fixed
- Fixed biased surrogate generation (now uses full time range)
- Enhanced frequency band access as tensor properties

### Added
- Production-ready codebase after cleanup
- Maintained 99.6% test coverage

## [0.1.0] - 2025-06-10

### Added
- Initial release of gpu-pac (GPU-accelerated Phase-Amplitude Coupling)
- Ultra-fast PAC computation with 341.8x speedup over TensorPAC
- Smart memory management with auto/chunked/sequential strategies
- Full PyTorch integration with gradient support
- Multi-GPU support with DataParallel
- Trainable bandpass filters with pooling
- Comprehensive test suite
- TensorPAC compatibility layer
- Detailed benchmarking tools
- `pha_bands_hz` and `amp_bands_hz` properties for accessing exact frequency bands

### Features
- Unbiased surrogate generation using full range time shifts
- Z-score normalization with permutation testing
- Support for batch processing
- Memory-efficient chunked processing for large datasets
- Automatic precision management (fp16/fp32)
- Rich examples and documentation

[0.2.1]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.2.1
[0.2.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.2.0
[0.1.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.1.0