# Known Limitations of gPAC

## Current Implementation Status (December 2025)

This document provides an honest assessment of gPAC's current limitations and implementation status.

## 1. Memory Optimization Not Implemented

### The Reality
- **Claim**: "89x memory reduction through smart chunking"
- **Truth**: Current implementation uses full memory expansion without chunking

### Technical Details
```python
# Current implementation in src/gpac/_PAC.py:
phase_expanded = phase.expand(batch, channels, n_pha, n_amp, seq_len)
amplitude_expanded = amplitude.expand(batch, channels, n_pha, n_amp, seq_len)
# This creates full tensors for ALL frequency combinations at once
```

### What This Means
- Memory usage scales with `n_phase_bands × n_amp_bands × seq_len`
- No chunking or sequential processing implemented
- MemoryManager class exists but is NOT integrated

## 2. Speed vs Memory Trade-off

### Current Trade-off
- **Speed**: ✅ 166-180x faster than TensorPAC (verified)
- **Memory**: ❌ Similar to naive GPU implementation
- **Choice**: We prioritized speed through full vectorization

### Why This Trade-off Exists
1. Full vectorization maximizes GPU parallelism → fast
2. Full vectorization requires all data in memory → memory-hungry
3. You cannot have both maximum speed AND minimal memory

## 3. GPU Memory Requirements

### Current Requirements
- For typical neuroscience datasets: ~24GB GPU memory
- For small test cases: Works on consumer GPUs
- For large-scale analysis: Requires datacenter GPUs (A100, V100)

### Who This Affects
- ❌ Researchers with consumer GPUs (RTX 3060, etc.)
- ✅ Researchers with datacenter access
- ✅ Those prioritizing speed over memory

## 4. Missing Features

### Not Yet Implemented
1. **Adaptive memory management**: MemoryManager exists but not integrated
2. **Chunked processing**: Code supports it but not enabled
3. **Multi-GPU support**: Single GPU only currently
4. **CPU fallback**: GPU required

### What Exists But Isn't Used
- `_MemoryManager.py`: Sophisticated memory management (343 lines, not integrated)
- `ModulationIndexMemoryOptimized`: Memory-efficient MI calculation (not fully utilized)
- Chunking infrastructure: Present but disabled for speed

## 5. Documentation vs Reality

### Previous Documentation Issues
- Documentation claimed memory optimization that doesn't exist
- Mixed theoretical capabilities with actual implementation
- Overclaimed achievements

### Current Status
- Documentation updated to reflect reality
- Claims now match implementation
- Future improvements clearly marked as "planned"

## 6. Comparison with TensorPAC

### What gPAC Provides
- ✅ 166-180x speedup on GPU
- ✅ PyTorch integration
- ✅ Full differentiability
- ✅ Comparable accuracy

### What TensorPAC Provides
- ✅ CPU efficiency
- ✅ Lower memory usage
- ✅ Mature, stable codebase
- ✅ No GPU required

## 7. Use Case Recommendations

### Use gPAC When
- You have sufficient GPU memory
- Speed is critical
- You need PyTorch integration
- You're building ML pipelines

### Use TensorPAC When
- GPU memory is limited
- CPU-only environment
- Memory efficiency is critical
- Stability is paramount

## 8. Future Roadmap

### Planned Improvements
1. **Memory optimization integration** (v2.0)
   - Enable MemoryManager
   - Implement adaptive chunking
   - Maintain speed where possible

2. **Multi-GPU support** (v2.1)
   - Distribute frequency bands across GPUs
   - Further memory distribution

3. **CPU fallback** (v2.2)
   - Automatic GPU/CPU selection
   - Graceful degradation

### Timeline
- These are aspirational goals
- No committed timeline
- Contributions welcome

## 9. Contributing

If you want to help address these limitations:
1. Memory optimization is top priority
2. See `src/gpac/_MemoryManager.py` for existing code
3. Integration points are marked in `_PAC.py`
4. Tests must maintain accuracy

## 10. Honest Summary

**gPAC is**:
- A fast GPU-accelerated PAC implementation
- Great for users with sufficient GPU memory
- Well-suited for ML integration

**gPAC is not**:
- Memory-optimized (yet)
- Suitable for memory-constrained environments
- A drop-in replacement for all TensorPAC use cases

**The Bottom Line**:
We built a Ferrari (fast) not a Prius (efficient). Choose accordingly.

---

*Last updated: December 2025*
*Version: 1.0.0*