# Feature Request: Surrogate Data Generation Methods

**Date:** 2025-05-26  
**Priority:** Low  
**Status:** Planned  

## Overview
Implement surrogate data generation methods for statistical validation of PAC results.

## Methods Needed
1. **Phase shuffling**
   - Randomize phase while preserving spectrum
   - Test null hypothesis of no PAC

2. **Amplitude shuffling**
   - Shuffle amplitude time series
   - Preserve spectral properties

3. **Time-shifted surrogates**
   - Circular shift of signals
   - Break temporal relationships

4. **Block resampling**
   - Bootstrap-like approach
   - Preserve local structure

## Implementation
- Add to `src/gpac/surrogate.py`
- Integrate with PAC statistical testing
- Support batch generation
- Maintain differentiability where possible

## Use Cases
- Statistical significance testing
- Multiple comparisons correction
- Confidence interval estimation
- Null distribution generation

## Success Criteria
- [ ] 4+ surrogate methods implemented
- [ ] Integration with PAC pipeline
- [ ] Statistical validation tools
- [ ] Documentation and examples

## Related Work
- TensorPAC surrogate methods
- Existing neuroscience standards
- Performance considerations for large datasets