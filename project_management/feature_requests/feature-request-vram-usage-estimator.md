<!-- ---
!-- Timestamp: 2025-06-01 15:25:00
!-- Author: ywatanabe
!-- File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC/project_management/feature_requests/feature-request-vram-usage-estimator.md
!-- --- -->

# Feature Request: VRAM Usage Estimator

## Description
Add a VRAM usage estimator that calculates expected GPU memory requirements based on input parameters before running PAC analysis. This would improve user experience by preventing out-of-memory errors.

## Motivation
- Users often encounter CUDA out-of-memory errors with large datasets
- No easy way to predict memory requirements before running
- Would allow users to adjust parameters proactively

## Proposed Implementation

### 1. Estimation Function
```python
def estimate_vram_usage(
    batch_size: int,
    n_chs: int, 
    seq_len: int,
    fs: float,
    pha_n_bands: int,
    amp_n_bands: int,
    n_perm: Optional[int] = None,
    fp16: bool = False
) -> Dict[str, float]:
    """
    Estimate VRAM usage in MB.
    
    Returns:
        Dictionary with:
        - 'signal_mb': Input signal memory
        - 'filters_mb': Filter bank memory
        - 'filtered_mb': Filtered signals memory
        - 'pac_mb': PAC matrix memory
        - 'perm_mb': Permutation testing memory
        - 'total_mb': Total estimated memory
        - 'recommended_gpu': Suggested GPU memory size
    """
```

### 2. Integration Points
- Add as method to PAC class: `pac.estimate_memory()`
- Add warning if estimated > available VRAM
- Suggest parameter adjustments if memory too high

### 3. Memory Calculation Formula
Based on the architecture:
- Input signal: batch × n_chs × seq_len × dtype_size
- Filter banks: 2 × n_bands × filter_len × dtype_size
- Filtered signals: batch × n_chs × n_bands × seq_len × dtype_size × 2 (phase + amp)
- PAC matrix: batch × n_chs × pha_n_bands × amp_n_bands × dtype_size
- Permutations: n_perm × pac_matrix_size (if using batch processing)

### 4. Example Usage
```python
# Check memory before running
mem_estimate = gpac.estimate_memory(
    batch_size=32,
    n_chs=64,
    seq_len=10000,
    fs=1000,
    pha_n_bands=50,
    amp_n_bands=50
)

print(f"Estimated VRAM: {mem_estimate['total_mb']:.1f} MB")
if mem_estimate['total_mb'] > torch.cuda.get_device_properties(0).total_memory / 1024**2:
    print("WARNING: May exceed available VRAM!")
```

## Benefits
1. Prevents OOM errors
2. Helps users optimize parameters
3. Better resource planning
4. Improved user experience

## Priority
Medium - Would significantly improve usability

## Status
✅ **COMPLETED** (2025-06-01)

### Implementation Summary
- Created `memory_estimator.py` module with full functionality
- Integrated `estimate_memory()` method into PAC class
- Added parameter suggestion functionality
- Created comprehensive demo and test suite
- Memory estimation accurately predicts VRAM usage based on:
  - Signal dimensions (batch, channels, time)
  - Frequency resolution (phase/amplitude bands)
  - Permutation testing requirements
  - FP16 optimization
- Provides warnings for high memory usage
- Suggests parameter adjustments to fit memory constraints

<!-- EOF -->