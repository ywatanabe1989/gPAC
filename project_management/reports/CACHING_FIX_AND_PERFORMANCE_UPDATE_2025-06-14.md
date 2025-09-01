# Caching Fix and Performance Update
Date: 2025-06-14
Author: Agent 6bde3d14-f0b0-42bd-a37e-89b71c4201f7

## Summary

Fixed critical caching bug that was using memory addresses instead of tensor values for cache keys. Re-evaluated performance with honest benchmarks.

## Caching Bug Fix

### Issue
- Cache key used `x.data_ptr()` (memory address) instead of tensor values
- Caused incorrect cache hits/misses

### Solution  
- Implemented `_create_cache_key()` using SHA256 hash of tensor values
- Added comprehensive test suite verifying correct behavior

## Performance Results

### Configuration Impact on Performance

| Configuration | Frequency Bands | Permutations | gPAC vs TensorPAC |
|--------------|-----------------|--------------|-------------------|
| Simple | 10×10 | 20-50 | ~2x faster |
| Production (tested) | 50×50 | 200 | ~6x slower |
| Parameter sweep | up to 64×64 | 0-64 | Varies |

### Key Findings

1. **Performance is configuration-dependent**
   - Small configurations: gPAC has GPU advantage
   - Large configurations: Algorithmic complexity dominates

2. **Fair comparison confirmed**
   - Both implementations use identical frequency bands
   - Parameter sweep benchmark ensures equal conditions

3. **Data reshaping overhead**
   - gPAC: Native support for `(batch, channels, segments, time)` format
   - TensorPAC: Requires reshape to `(batch*channels*segments, time)`
   - Reshaping adds overhead but doesn't explain 6x slowdown with large configs

4. **Caching provides limited real-world benefit**
   - Useful for repeated identical computations
   - Not applicable to typical neuroscience workflows

## Next Steps

1. Investigate optimization opportunities for large frequency band configurations
2. Update documentation with configuration-dependent performance characteristics
3. Consider making caching optional/disabled by default

## Status
- Caching bug: Fixed ✓
- Performance evaluation: Complete ✓
- Branch: develop (ahead by 1 commit)