# Changelog

All notable changes to gpu-pac will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — scitex-grade tests

- Unit tests under `tests/gpac/test__PAC_scitex.py`,
  `tests/gpac/core/test__Hilbert_scitex.py`,
  `tests/gpac/core/test__BandPassFilter_scitex.py`, and
  `tests/gpac/core/test__ModulationIndex_scitex.py` covering
  filter-bank construction, the Hilbert wrapper, modulation-index
  computation, and seeded surrogate generation. Each test follows the
  scitex Arrange / Act / Assert convention with a single assertion
  per test (STX-TQ001-007).
- Integration tests under `tests/integration/`:
  `test_synthetic_pac_detection.py` (peak MI lands on the known
  6 Hz / 80 Hz pair), `test_reproducibility.py` (same inputs +
  `random_seed=42` → identical outputs across invocations),
  `test_gpac_vs_tensorpac_within_5pct.py` (raw PAC agrees with
  TensorPAC within the 5 % tolerance from neurovista PR #52).
- Pytest markers `unit`, `integration`, `requires_data`, `slow`, and
  `gpu` registered in `pyproject.toml`.

### Changed — CI

- `ci.yml` and `tests.yml` matrix updated to Python 3.10 / 3.11 / 3.12
  (dropped EOL 3.8 / 3.9). Slow + GPU tests are now skipped on the
  CPU runner via `-m "not slow and not gpu"`. Codecov upload runs with
  `if: always()` so coverage uploads even when a row fails.
- New `tests-gpu.yml` workflow stub guarded by `if: false`; the
  operator wires it up to a self-hosted GPU runner later.
- New `codecov.yml` pinning the unbranched badge to `develop` and
  ignoring `tests/`, `examples/`, `benchmark/`, `paper/`, `docs/`,
  and `src/gpac/dataset/` (data-generation helpers, exercised only
  through integration tests).

## [0.4.0] - 2025-09-01

### Added
- **Random Seed Functionality**: Reproducible permutation testing for scientific reproducibility
  - Added `random_seed` parameter to PAC class (default: 42, following ML conventions)
  - PyTorch generator-based seeding for deterministic surrogate generation
  - Automatic GPU/CPU device handling for generators
  - Backward compatibility mode with `random_seed=None` for legacy behavior

### Features
- **Reproducible Results**: Same seed produces identical z-scores across runs and systems
- **Scientific Integrity**: Maintains excellent compatibility with existing analyses
  - Z-score correlation: **0.9908** (>99% consistency with unseeded version)
  - Mean absolute difference: **0.0911** (negligible impact on statistical power)
  - Raw PAC values remain perfectly identical (correlation = 1.0000)
- **User Experience**: Intuitive API design with sensible defaults
  - Default seed of 42 follows scikit-learn and ML conventions
  - Graceful degradation with `random_seed=None` preserves legacy workflows
  - No breaking changes to existing PAC API

### Technical Implementation
- **Thread-Safe**: PyTorch generator-based implementation ensures thread safety
- **Device-Aware**: Automatic generator device placement for CUDA compatibility  
- **Validation**: Comprehensive parameter validation with clear error messages
- **Performance**: Minimal computational overhead (~0.1% impact)

### Testing
- **Comprehensive Test Suite**: 6 new test functions covering all aspects
  - `test_random_seed_reproducibility()` - Identical results with same seed
  - `test_random_seed_different_results()` - Different surrogates with different seeds
  - `test_random_seed_default_value()` - Default seed verification
  - `test_random_seed_none_non_deterministic()` - Legacy behavior preservation  
  - `test_random_seed_parameter_validation()` - Input validation
  - `test_backward_compatibility_z_scores()` - Production validation with realistic signals
- **Production Validation**: Tested with n_perm=200, 2-second signals, multiple coupling strengths
- **Backward Compatibility**: Verified across different noise levels and signal characteristics

### Usage Examples
```python
# New default behavior (reproducible)
pac = PAC(..., random_seed=42, n_perm=200)
result1 = pac(signal)  # Deterministic z-scores

# Custom seed for different random sequences  
pac = PAC(..., random_seed=999, n_perm=200)
result2 = pac(signal)  # Different but reproducible z-scores

# Legacy mode (non-deterministic, backward compatible)
pac = PAC(..., random_seed=None, n_perm=200) 
result3 = pac(signal)  # Non-deterministic z-scores (original behavior)
```

### Scientific Impact
This implementation enables reproducible PAC research while maintaining full backward compatibility. Users with existing computational results (2+ months of computation) can continue using their analyses with complete confidence, while gaining reproducibility for future work.

## [0.3.0] - 2025-06-12

### Changed
- **BREAKING**: Changed default frequency band spacing from `log` to `linear`
- **BREAKING**: Changed default phase frequency range from `(4, 30)` Hz to `(2, 20)` Hz
- **BREAKING**: Changed default amplitude frequency range from `(60, 150)` Hz to `(60, 160)` Hz

### Improved
- Default parameters now better align with PAC literature recommendations
- Linear spacing provides more standard neuroscience frequency band distribution
- Phase range now covers key neural rhythms (delta through beta)
- Reduced issues with excessively wide bandwidths at higher frequencies

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

[0.4.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.4.0
[0.3.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.3.0
[0.2.1]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.2.1
[0.2.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.2.0
[0.1.0]: https://github.com/ywatanabe1989/gPAC/releases/tag/v0.1.0