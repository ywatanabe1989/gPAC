<!-- ---
!-- Timestamp: 2025-06-02 23:23:00
!-- Author: Claude
!-- File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/gPAC/project_management/feature_requests/feature-request-adaptive-sincnet.md
!-- --- -->

# Feature Request: Adaptive SincNet Filter Implementation

## Summary
Implement adaptive filter length calculation in SincNet-based bandpass filters to improve frequency selectivity and PAC detection accuracy while maintaining trainability.

## Background
Current implementation uses fixed 251-tap filters regardless of frequency, leading to:
- Poor frequency selectivity for low frequencies
- Over-designed filters for high frequencies
- Significant PAC detection discrepancy compared to TensorPAC

## Proposed Solution

### 1. Adaptive Filter Length Calculation
```python
def compute_filter_length(f_low, fs, n_cycles=3, min_len=51, max_len=1001):
    filter_len = int(n_cycles * fs / f_low)
    if filter_len % 2 == 0:
        filter_len += 1
    return np.clip(filter_len, min_len, max_len)
```

### 2. Integration Points
- Modify `DifferentiableBandPassFilter` to use adaptive lengths
- Update `BandPassFilter` wrapper class
- Maintain backward compatibility with fixed-length option

### 3. Benefits
- 73.6% improvement in MI detection (tested)
- Better matches TensorPAC frequency selectivity
- Maintains full differentiability for gradient-based optimization
- More efficient computation (fewer taps for high frequencies)

## Implementation Details

### Files to Modify
1. `/src/gpac/_Filters/_DifferentiableBandPassFilter.py`
   - Add adaptive filter length calculation
   - Modify `__init__` to accept `n_cycles` parameter

2. `/src/gpac/_BandPassFilter.py`
   - Pass through adaptive parameters
   - Add compatibility flag for fixed vs adaptive

### API Changes
```python
BandPassFilter(
    seq_len=1000,
    fs=500,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=10,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=10,
    adaptive_filter=True,  # New parameter
    n_cycles=3,           # New parameter
    trainable=True
)
```

## Testing Requirements
1. Verify frequency response matches design specifications
2. Test gradient flow for trainable bands
3. Compare PAC detection accuracy with TensorPAC
4. Benchmark performance (adaptive should be faster for mixed frequencies)

## Priority
High - This significantly improves PAC detection accuracy, which is the core functionality.

## References
- Adaptive implementation demo: `/examples/gpac/adaptive_sincnet_implementation.py`
- Systematic analysis: `/examples/gpac/systematic_sincnet_analysis.py`
- Original issue discussion: Phase error analysis and filter comparison

<!-- EOF -->