# Release Notes - v0.3.1

## New Features

### Amplitude Distribution Analysis
- Added `compute_distributions` parameter to `PAC.forward()` method
- When enabled, returns amplitude distributions across phase bins for clinical analysis
- Useful for seizure onset detection and phase preference analysis
- Returns phase bin centers and edges for proper visualization

## API Improvements
- Consistent return dictionary structure - all keys are always present
- Keys return `None` when feature is disabled instead of being omitted
- Better support for downstream applications expecting consistent output format

## Bug Fixes
- Fixed deprecated property references throughout codebase
- Updated examples to use new `pha_bands_hz` and `amp_bands_hz` properties
- Fixed parameter naming inconsistencies in examples

## Documentation
- Added clinical analysis examples in README
- New example demonstrating amplitude distribution visualization
- Updated all code examples to reflect current API

## Performance Notes
- `compute_distributions=False` (default) maintains same performance as v0.3.0
- When enabled, additional memory allocation and computation for distributions
- Recommended to enable only when amplitude distributions are needed

## Compatibility
- Backward compatible with v0.3.0
- Python 3.8+ required
- PyTorch 1.9+ required