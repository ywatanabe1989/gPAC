# Release Notes - v0.2.1

## Bug Fixes

### FP16 Default Parameter Fix
- **Changed default `fp16` parameter from `True` to `False`** in PAC module
  - Prevents unexpected RuntimeError with mixed precision tensors
  - Ensures stable operation by default
  - Users can still enable fp16 mode explicitly for memory efficiency

### FP16 Output Consistency
- **Added automatic float32 conversion for fp16 outputs**
  - All PAC outputs now consistently return float32 dtype
  - Prevents downstream dtype mismatches
  - Maintains numerical precision for scientific applications

## New Features

### Comprehensive FP16/Float32 Testing
- **Added 25 new tests** for fp16/float32 compatibility across all modules:
  - `test__PAC_fp16.py`: 5 tests for main PAC functionality
  - `test__BandPassFilter_fp16.py`: 6 tests for filtering operations
  - `test__Hilbert_fp16.py`: 7 tests for Hilbert transform
  - `test__ModulationIndex_fp16.py`: 7 tests for MI calculation
- Tests verify:
  - Proper dtype handling for mixed precision inputs
  - High correlation (>0.99) between fp16 and float32 results
  - Correct default parameters
  - Edge case handling

## Compatibility

- Fully backward compatible
- No API changes
- Existing code will work without modifications

## Upgrade Instructions

```bash
pip install --upgrade gpu-pac
```

## Contributors

- Yusuke Watanabe (@ywatanabe1989)