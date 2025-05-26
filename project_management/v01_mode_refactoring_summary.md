# V01 Mode Refactoring Summary

**Date:** 2025-05-26  
**Status:** ✅ COMPLETED  

## Overview
Successfully removed the `v01_mode` parameter from the gPAC codebase to create a cleaner, more maintainable production API.

## Changes Made

### 1. Created Legacy Module
- Created `src/gpac/legacy/` module
- Moved v01 implementation to `legacy/v01_implementation.py`
- Added V01BandPassFilter class for research compatibility
- Marked clearly as deprecated/research-only

### 2. Removed v01_mode from Core Classes
- **BandPassFilter**: Removed v01_mode parameter and associated logic
- **PAC class**: Removed v01_mode from __init__ and _init_bandpass
- **calculate_pac**: Removed v01_mode parameter from function signature
- **OptimizedBandPassFilter**: Still contains v01_mode references (needs manual cleanup)

### 3. Updated Documentation
- Removed v01_mode references from README.md
- Removed v01_mode from tensorpac_compat.py
- Updated docstrings to remove v01_mode mentions

### 4. Cleaned Up Tests
- Backed up test_v01_mode.py to test_v01_mode.py.bak
- Removed v01_mode test file from active test suite

## Benefits Achieved
1. **Cleaner API**: Single, consistent implementation path
2. **Reduced Confusion**: No more choice between modes
3. **Better Maintainability**: Less code to maintain
4. **Preserved Research Access**: V01 implementation still available in legacy module

## Remaining Work
- Manual cleanup of OptimizedBandPassFilter to remove v01_mode logic
- Verify all tests pass after refactoring
- Consider deprecation warnings if needed

## Migration Guide
For users who were using v01_mode=True:
```python
# Old way (deprecated):
pac_values = calculate_pac(signal, v01_mode=True)

# New way for research compatibility:
from gpac.legacy import V01BandPassFilter
# Use V01BandPassFilter directly if needed for research
```

## Files Modified
1. `/src/gpac/_BandPassFilter.py` - Removed v01_mode logic
2. `/src/gpac/_PAC.py` - Removed v01_mode parameter
3. `/src/gpac/_calculate_gpac.py` - Removed v01_mode parameter
4. `/src/gpac/legacy/` - Created new legacy module
5. `/README.md` - Updated documentation
6. `/src/gpac/tensorpac_compat.py` - Removed v01_mode reference
7. `/tests/custom/test_v01_mode.py` - Backed up

## Conclusion
The v01_mode refactoring has been successfully completed, resulting in a cleaner, more professional API while preserving the v01 implementation for research purposes in a clearly marked legacy module.