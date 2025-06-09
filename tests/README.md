# gPAC Test Suite

This directory contains all tests for the gPAC package, organized into three main categories:

## Directory Structure

### 1. `gpac/` - Core Module Tests
Basic unit tests that mirror the source code structure:
- `test__BandPassFilter.py` - Tests for the unified bandpass filter interface
- `test__Hilbert.py` - Tests for Hilbert transform implementation
- `test__ModulationIndex.py` - Tests for modulation index calculation
- `test__PAC.py` - Tests for the main PAC module
- `test___init__.py` - Tests for package initialization
- `_Filters/` - Subdirectory for filter-specific tests:
  - `test__BaseFilter1D.py` - Base filter class tests
  - `test__StaticBandPassFilter.py` - Static (non-trainable) filter tests
  - `test__DifferentiableBandPassFilter.py` - SincNet-style trainable filter tests

### 2. `trainability/` - Trainability Tests
Tests for verifying gradient flow and parameter optimization:
- `test_pac_trainability.py` - Comprehensive trainability test with synthetic PAC signals
- `test_pac_trainability_simple.py` - Simplified demonstration of filter learning

### 3. `comparison_with_tensorpac/` - Comparison Tests
Tests comparing gPAC with TensorPAC implementation:
- `test_bandpass_filter.py` - Bandpass filter comparison (FIR vs Butterworth)
- `test_hilbert.py` - Hilbert transform comparison
- `test_modulation_index.py` - Modulation index calculation comparison
- `test_pac.py` - Full PAC pipeline comparison
- `README.md` - Detailed notes on TensorPAC differences and compatibility

## Running Tests

Run all tests:
```bash
pytest
```

Run specific category:
```bash
pytest tests/gpac/              # Core tests only
pytest tests/trainability/       # Trainability tests only
pytest tests/comparison_with_tensorpac/  # Comparison tests only
```

Run specific test file:
```bash
pytest tests/gpac/test__PAC.py
```

## Configuration

- `conftest.py` - Shared pytest fixtures and configuration
- `pytest.ini` - Pytest configuration settings