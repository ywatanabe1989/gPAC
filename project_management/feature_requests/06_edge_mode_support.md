# Feature Request: Edge Mode Support for Filtering

**Date:** 2025-05-26  
**Priority:** Low  
**Status:** Planned  

## Overview
Add comprehensive edge mode support for bandpass filtering to handle signal boundaries properly.

## Current Limitation
- Only 'reflect' mode supported
- TensorPAC supports multiple modes
- Edge artifacts possible with current implementation

## Required Edge Modes
1. **reflect** (current) - Mirror padding
2. **constant** - Zero padding
3. **edge** - Replicate edge values
4. **wrap** - Circular padding
5. **symmetric** - Symmetric padding

## Implementation Details
- Modify `_BandPassFilter.py`
- Add edge_mode parameter
- Ensure differentiability maintained
- Match scipy.signal behavior

## Use Cases
- **reflect**: General purpose (default)
- **constant**: Known baseline signals
- **edge**: Continuous recordings
- **wrap**: Periodic signals
- **symmetric**: Avoid discontinuities

## Success Criteria
- [ ] All 5 edge modes implemented
- [ ] Behavior matches scipy.signal
- [ ] Differentiable variants supported
- [ ] Tests for each mode

## Test Plan
- Edge case signals (short, discontinuous)
- Compare with scipy outputs
- Validate gradient flow
- Performance impact assessment