# Feature Request: Refactor v01_mode Legacy Code

**Date:** 2025-05-26  
**Priority:** Medium  
**Status:** Pending  

## Overview
Remove the `v01_mode` parameter and associated legacy code paths from the production codebase. This parameter represents an older implementation approach that should not be exposed in the public API.

## Current Issue
- `v01_mode` parameter exists in multiple classes (PAC, BandPassFilter, calculate_pac)
- Creates confusion about which mode to use
- Maintains legacy code paths that complicate maintenance
- Not appropriate for a clean production API

## Affected Components
1. **PAC class** - Has `v01_mode` parameter in __init__
2. **BandPassFilter class** - Switches between two filtering implementations
3. **calculate_pac function** - Passes through v01_mode parameter
4. **Tests** - Some tests specifically for v01_mode functionality

## Proposed Solution
1. **Remove v01_mode parameter** from all public APIs
2. **Consolidate on the scipy-compatible implementation** (current default)
3. **Move v01 code to legacy module** if needed for research/comparison
4. **Update all tests** to use the standard implementation
5. **Create migration guide** if any users depend on v01_mode

## Benefits
- Cleaner, more maintainable codebase
- Single, well-tested implementation path
- Reduced confusion for users
- Easier to optimize and maintain

## Migration Strategy
1. Deprecate v01_mode in next minor release (0.3.0) with warnings
2. Remove completely in next major release (1.0.0)
3. Provide clear documentation on any behavioral changes

## Alternative Approach
If v01 behavior is needed for specific research compatibility:
- Move to a separate `gpac.legacy` module
- Not part of main API
- Clearly documented as legacy/research-only

## Success Criteria
- [ ] All v01_mode parameters removed from public API
- [ ] Single, consistent filtering implementation
- [ ] All tests pass with standard implementation
- [ ] Documentation updated
- [ ] No performance regression