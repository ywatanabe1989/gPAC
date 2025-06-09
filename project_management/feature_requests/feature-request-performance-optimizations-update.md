# Performance Optimization Update - Critical Regression Found

**Date**: 2025-06-06  
**Updated By**: auto-CLAUDE-20250606-speed  
**Priority**: CRITICAL

## Major Performance Regression Discovered

### Current Status vs Expected Performance
- **Previous achievement**: 100x+ speedup (documented in IMPORTANT-FAIR-COMPARISON-WITH-TENSORPAC.md)
- **Current performance**: 0.5-0.6x speed (2x SLOWER than TensorPAC)
- **Regression factor**: ~200x performance loss

### Root Causes Identified

1. **Memory Operations (42% of CUDA time)**
   - Excessive device-to-device copies (`aten::copy_`: 42.34%)
   - Memory bandwidth saturation from poor access patterns
   - Unnecessary tensor cloning in StaticBandPassFilter

2. **ModulationIndex Implementation Issues**
   - Creates massive one_hot tensors (23GB for medium-size data)
   - Over-vectorization causing OOM errors
   - Should use scatter/gather operations instead

3. **Inefficient Vectorization Strategy**
   - Trying to process all phase-amplitude pairs simultaneously
   - Creates intermediate tensors too large for GPU memory
   - Missing chunked processing for memory efficiency

4. **torch.compile Issues**
   - Long initialization times (2+ minutes)
   - Should be disabled by default until optimized

### Immediate Actions Required

1. **Revert to Previous Fast Implementation**
   - Check git history for when 100x speedup was achieved
   - Review implementations mentioned in CLAUDE.md

2. **Fix Current Implementation**
   - Remove unnecessary clone() operations
   - Rewrite ModulationIndex without one_hot encoding
   - Implement proper chunked processing
   - Disable torch.compile by default

3. **Optimize Memory Access**
   - Use in-place operations where possible
   - Reduce intermediate tensor creation
   - Implement memory pooling

### Updated Success Metrics

- [ ] ❌ 20-30% reduction in memory usage (FAILED - using MORE memory)
- [ ] ❌ 15-25% speedup on typical workloads (FAILED - 2x SLOWER)
- [ ] ❌ Better multi-GPU scaling efficiency (Not tested due to OOM)
- [ ] ✅ No regression in accuracy (Only metric achieved)

### Revised Implementation Priority

#### Phase 0 (URGENT - Performance Recovery)
1. Identify and fix performance regression
2. Remove memory-intensive operations
3. Restore baseline 100x speedup

#### Phase 1 (Previous Quick Wins - Re-evaluate)
1. ⚠️ torch.compile support (causing issues, disable by default)
2. ✅ Reduce obvious reshape operations (completed but insufficient)
3. ⚠️ Lazy surrogate statistics (needs memory optimization first)

#### Phase 2+ (Postponed until baseline restored)
- Further optimizations only after recovering lost performance

### Testing Evidence

```
Small (1ch, 1s): gPAC 0.021s vs TensorPAC 0.013s = 0.6x
Medium (8ch, 4s): gPAC 0.632s vs TensorPAC 0.300s = 0.5x
Large (16ch, 60s): OOM error (23GB allocation attempted)
```

### Conclusion

The performance optimizations in Phase 1 have not addressed the fundamental regression. The current implementation is significantly slower than both TensorPAC and previous gPAC versions. Priority must shift to recovering the documented 100x speedup before pursuing further optimizations.