# Feature Request: Differentiable Module Implementation

**Date:** 2025-05-26  
**Priority:** High  
**Status:** Requested  

## Overview

Ensure all core modules are fully differentiable to enable end-to-end gradient-based optimization.

## Requirements

### 1. Differentiable Modules
All modules must be differentiable unless explicitly using static components:

- **BandPassFilter**: ✅ Implemented with trainable and static variants
- **Hilbert Transform**: ❓ Need to verify differentiability 
- **Modulation Index**: ❓ Need to verify differentiability
- **PAC Calculator**: ❓ Need to verify differentiability

### 2. Filter Variants
- **Static BandPassFilter**: Non-differentiable, fixed parameters
- **Trainable BandPassFilter**: Fully differentiable with learnable frequency parameters
- **Main BandPassFilter**: Supports both modes

## Implementation Strategy

1. **Audit existing modules** for gradient compatibility
2. **Replace non-differentiable operations** with differentiable alternatives
3. **Maintain backward compatibility** with static versions
4. **Add gradient tests** to ensure differentiability

## Success Criteria

- [ ] All modules support `requires_grad=True`
- [ ] Gradients flow through entire pipeline
- [ ] Performance comparable to static versions
- [ ] Tests validate gradient computation

## Related Files

- `src/gpac/_BandPassFilter.py`
- `src/gpac/_Filters/_TrainableBandPassFilter.py`  
- `src/gpac/_Hilbert.py`
- `src/gpac/_ModulationIndex.py`
- `src/gpac/_PAC.py`