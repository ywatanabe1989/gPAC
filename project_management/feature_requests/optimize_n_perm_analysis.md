<!-- ---
!-- Timestamp: 2025-06-01 15:00:00
!-- Author: ywatanabe  
!-- File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC/project_management/feature_requests/optimize_n_perm_analysis.md
!-- --- -->

# gPAC n_perm Optimization Analysis

## Current Performance Issue
- Without permutations: gPAC is 5.1x faster than TensorPAC
- With 200 permutations: gPAC becomes SLOWER than TensorPAC (0.0x speedup)
- Cost per permutation: ~1.8ms (356ms for 200 perms)

## Root Cause
The current implementation in `src/gpac/_PAC.py` processes permutations sequentially:
```python
for perm_idx, shift in enumerate(shift_points):
    # Process each permutation one by one
```

## Optimization Opportunities

### 1. Batch Permutation Processing
Instead of looping through permutations, process multiple permutations in parallel:
```python
# Current: Sequential processing
for perm in range(n_perm):
    shifted = torch.roll(pha, shift[perm])
    pac = compute_mi(shifted, amp)

# Optimized: Batch processing  
all_shifts = torch.stack([torch.roll(pha, s) for s in shifts])
all_pacs = compute_mi_batch(all_shifts, amp)  # Process all at once
```

### 2. GPU Memory Optimization
- Pre-allocate buffers for all permutations
- Use in-place operations where possible
- Process in optimal batch sizes based on available VRAM

### 3. Algorithmic Improvements
- Use FFT-based circular shifting for efficiency
- Implement permutation caching for repeated analyses
- Consider approximate methods (e.g., analytical p-values for large n_perm)

### 4. Parallelization Strategy
```python
def generate_surrogates_optimized(self, pha, amp):
    # Reshape to add permutation dimension
    batch, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
    
    # Generate all shifts at once
    shifts = torch.randint(seq_len, (self.n_perm,), device=pha.device)
    
    # Expand phase for all permutations at once
    pha_expanded = pha.unsqueeze(2).expand(-1, -1, self.n_perm, -1, -1, -1)
    
    # Apply all shifts in parallel using advanced indexing
    indices = (torch.arange(seq_len, device=pha.device).unsqueeze(0) - 
               shifts.unsqueeze(1)) % seq_len
    pha_shifted = torch.gather(pha_expanded, -1, indices.expand_as(pha_expanded))
    
    # Compute MI for all permutations at once
    return self.modulation_index_batch(pha_shifted, amp.unsqueeze(2))
```

### 5. Implementation Priority
1. **High Priority**: Batch processing of permutations
2. **Medium Priority**: Memory optimization and pre-allocation
3. **Low Priority**: FFT-based methods and approximations

## Expected Performance Gains
- Target: Maintain >5x speedup even with 200+ permutations
- Estimated improvement: 10-50x faster permutation processing
- Memory usage: May increase but controllable via batching

## Testing Strategy
1. Benchmark with various n_perm values: [10, 50, 100, 200, 500, 1000]
2. Monitor GPU memory usage
3. Validate statistical accuracy of z-scores
4. Compare with TensorPAC's permutation implementation

<!-- EOF -->