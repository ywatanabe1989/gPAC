# Feature Request: Performance Optimizations for PAC Module

**Date**: 2025-06-02
**Priority**: High
**Status**: Phase 1 Implemented

## Summary
Implement various performance optimizations for the PAC module based on comprehensive analysis.

## Current Performance Strengths
- ✅ Adaptive memory strategy selection
- ✅ Multi-GPU support with batch parallelization
- ✅ FP16 support
- ✅ Vectorized surrogate generation
- ✅ Frequency caching

## Proposed Optimizations

### 1. Critical: Reduce Tensor Reshaping Operations
**Problem**: Multiple reshape/permute operations create memory copies
```python
# Current flow has many reshapes:
x = x.reshape(batch_size * n_chs, n_segments, seq_len)
x = x.reshape(batch_size, n_chs, n_segments, -1, seq_len, 2)
pha = pha.permute(0, 1, 3, 2, 4)
amp = amp.permute(0, 1, 3, 2, 4)
```

**Solution**: 
- Redesign component interfaces to maintain consistent tensor layouts
- Use `view()` instead of `reshape()` where possible
- Minimize permute operations by designing better data flow

### 2. Add PyTorch Compilation Support
**Problem**: Missing modern PyTorch optimization features
**Solution**:
```python
# Add torch.compile for PyTorch 2.0+
if torch.__version__ >= "2.0":
    self.modulation_index = torch.compile(self.modulation_index, mode="max-autotune")
    
# Or use JIT for critical paths
@torch.jit.script
def _compute_modulation_index_core(pha, amp):
    # Core computation
```

### 3. Optimize Surrogate Generation Memory Usage
**Problem**: All surrogates stored in memory even when only statistics needed
**Solution**:
- Add option to compute only mean/std without storing all surrogates
- Implement streaming surrogate computation for large n_perm
- Use memory pooling for frequent allocations

### 4. FFT-based Filtering for Long Sequences
**Problem**: Time-domain filtering inefficient for long sequences
**Solution**:
- Implement FFT-based filtering option
- Auto-switch based on sequence length (crossover ~1000-2000 samples)
- Benchmark to find optimal crossover point

### 5. Optimize Multi-GPU Communication
**Problem**: Frequent device transfers in multi-GPU processing
**Solution**:
- Use NCCL for efficient collective operations
- Minimize cross-device transfers
- Keep results on original devices longer

## Implementation Priority

### Phase 1 (Quick Wins) ✅ COMPLETED
1. ✅ Add torch.compile support
2. ✅ Reduce obvious reshape operations  
3. ✅ Add lazy surrogate statistics option

### Phase 2 (Major Improvements)
1. Redesign data flow to minimize permutations
2. Implement FFT-based filtering
3. Optimize multi-GPU communication

### Phase 3 (Advanced Optimizations)
1. Custom CUDA kernels for critical paths
2. Memory pooling system
3. Streaming processing support

## Success Metrics
- [ ] 20-30% reduction in memory usage
- [ ] 15-25% speedup on typical workloads
- [ ] Better multi-GPU scaling efficiency
- [ ] No regression in accuracy

## Testing Requirements
- Comprehensive benchmarks across data sizes
- Memory profiling before/after
- Multi-GPU scaling tests
- Accuracy validation against current implementation

## Implementation Summary (Phase 1)

### Completed Optimizations (2025-06-02)

#### 1. Torch.compile Support
- Added `enable_torch_compile` parameter to PAC constructor
- Automatically detects PyTorch 2.0+ availability
- Compiles core computation methods (`_compute_pac_core`)
- Graceful fallback for older PyTorch versions
- **Usage**: `PAC(..., enable_torch_compile=True)`

#### 2. Tensor Reshape Optimizations
- Replaced `reshape()` with `view()` where possible for better performance
- Combined slice and permute operations to reduce intermediate tensors
- Optimized data flow in `_compute_pac_core` method
- Reduced memory allocation overhead

#### 3. Lazy Surrogate Statistics
- Added `lazy_surrogate_stats` parameter to PAC constructor
- Computes only mean/std without storing all surrogates
- Implements chunked processing for memory efficiency
- Significant memory savings for large `n_perm` values
- **Usage**: `PAC(..., lazy_surrogate_stats=True)`

### Benchmark Results
- Created comprehensive benchmark script: `examples/performance/benchmark_optimizations.py`
- Tests all optimization combinations
- Measures execution time, memory usage, and speedup ratios
- Provides visualization of performance improvements

### API Changes
All optimizations are **backward compatible** and optional:
- Default behavior unchanged (all optimizations disabled by default)
- New parameters are opt-in with sensible defaults
- Existing code continues to work without modification

### Next Steps
- Phase 2: Major improvements (data flow redesign, FFT filtering)
- Phase 3: Advanced optimizations (custom kernels, memory pooling)

## Notes
- Maintain backward compatibility where possible
- Document any breaking changes
- Consider feature flags for experimental optimizations