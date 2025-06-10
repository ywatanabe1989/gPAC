# Changelog

All notable changes to gPAC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-06-10

### Added
- New `pha_bands_hz` and `amp_bands_hz` properties in PAC class for accessing exact frequency band definitions as tensors
- Improved TensorPAC compatibility with exact band matching
- Script for re-running all permutation-based analyses

### Changed
- **BREAKING**: Fixed biased surrogate generation in ModulationIndex
  - Changed from restricted 25-75% time shifts to full range (1 to time-1) shifts
  - Provides unbiased null distribution for more accurate statistical testing
  - Z-scores calculated with v0.2.0 will differ from v0.1.x results
- Improved correlation with TensorPAC z-scores

### Fixed
- Biased surrogate generation that was limiting time shifts to middle 50% of signal
- TensorPAC z-score visualization scaling issues

## [0.1.0] - 2025-06-01

### Added
- Initial release of gPAC
- GPU-accelerated Phase-Amplitude Coupling computation
- Support for multiple GPUs
- Trainable bandpass filters
- Comprehensive benchmarking suite
- TensorPAC compatibility layer

[0.2.0]: https://github.com/ywatanabe1989/gPAC/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.1.0