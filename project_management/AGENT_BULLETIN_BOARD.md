<!-- ---
!-- Timestamp: 2025-05-26 09:42:30
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/project_management/AGENT_BULLETIN_BOARD.md
!-- --- -->

# Project Agent Bulletin Board

## Agent Status
| Agent ID | Module | Status | Progress | Last Update |
|----------|--------|--------|----------|-------------|
| auto-CLAUDE-001-20250526 | PAC correlation investigation | ✅ | 100% | 10:12 |
| auto-CLAUDE-002-20250526 | MI formula fix implementation | ✅ | 100% | 10:22 |
| auto-CLAUDE-003-20250526 | Amplitude extraction investigation | ✅ | 100% | 10:35 |
| auto-CLAUDE-004-20250526 | Root cause identification | ✅ | 100% | 10:50 |
| auto-CLAUDE-005-20250526 | Compatibility layer implementation | ✅ | 100% | 11:05 |

## Current Work

### ✅ COMPLETED 
- **PAC correlation investigation** (auto-CLAUDE-001)
  - Root cause identified: Modulation Index calculation differs by ~22x
  - Filter implementation is correct (r=0.998 for filtfilt)
  - Issue is in MI normalization or calculation

## Key Findings

### 🚨 CRITICAL DISCOVERY
**The issue is NOT in the filters but in the Modulation Index calculation!**

### Detailed Analysis Results
1. **Filtfilt implementations**: ✅ Nearly identical (r=0.998)
2. **Filter coefficients**: ✅ Perfect match (r=1.000)
3. **Filter outputs**: ✅ Good correlation (r=0.993-0.998)
4. **PAC values**: ❌ Negative correlation (r=-0.413)
5. **Value magnitude**: ❌ gPAC is 22x smaller than TensorPAC

### Test Results (10 Hz phase, 80 Hz amplitude)
- **gPAC**: max=0.026, peak at 10.1 Hz / 158.3 Hz
- **TensorPAC**: max=0.575, peak at 3.3 Hz / 61.7 Hz
- **Ratio**: 0.0456 (gPAC/TensorPAC)

### Root Cause
The Modulation Index calculation in gPAC produces values ~22x smaller than TensorPAC:
- Different normalization approach
- Possible missing scaling factor
- Different entropy/KL divergence calculation

### ✅ MI Formula Fixed
- Changed from normalized MI [0,1] to TensorPAC-compatible [0,2] range
- Formula updated: `MI = 1.0 + entropy_part / log_n_bins`
- Matches TensorPAC's inverted scale (2=no coupling, 0=perfect coupling)

### ✅ Hilbert Transform Verified
- Hilbert amplitude extraction is perfect (r=1.000 with scipy)
- No scaling issue in Hilbert transform
- Issue must be elsewhere in the pipeline

### 🚨 FINAL ROOT CAUSE IDENTIFIED

The issue is **NOT** a 22x scaling problem, but a fundamental difference in approach:

1. **TensorPAC's filter() method**:
   - Returns processed phase angles [-π, π] for 'phase' mode
   - Returns amplitude envelope [0, max] for 'amplitude' mode
   - Combines filtering + Hilbert in one step

2. **gPAC's approach**:
   - Separates filtering and Hilbert into distinct steps
   - May be processing all frequency bands together differently

3. **MI Scale Difference**:
   - gPAC MI values: ~0.001-0.05 range
   - TensorPAC MI values: ~0.5-1.0 range
   - This is NOT just a scaling factor but a different calculation approach

### ✅ SOLUTION IMPLEMENTED
1. **Created compatibility layer**: `_calculate_gpac_tensorpac_compat.py`
2. **Improved correlation**: From 0.336 to 0.676 (2x improvement)
3. **Scaling factor**: ~2.86x brings values into similar range

### 📊 RESULTS
- Original correlation: r=0.336
- With compatibility layer: r=0.676
- Values now in similar range (max ~0.165)
- Still not perfect due to fundamental algorithmic differences

### 🎯 RECOMMENDATIONS FOR FUTURE
1. **Use compatibility layer** when comparing with TensorPAC
2. **Consider reimplementing** to match TensorPAC's filter() approach exactly
3. **Document differences** in user guide
4. **The v01 working version** likely had different implementation that matched better

### 🔍 V01 ANALYSIS (auto-CLAUDE-006)
**Why v01 had better correlation**:
1. **Simpler filtfilt**: Used depthwise convolution with `groups=len(kernels)`
2. **Batched processing**: All filters processed together, not individually
3. **Simple padding**: Just `padding="same"`, no manual odd extension
4. **Less overhead**: Direct approach inadvertently matched TensorPAC better

**Recommendation**: Consider restoring v01's depthwise convolution approach for better TensorPAC compatibility

## Dependencies
- All PAC calculations depend on correct MI normalization
- TensorPAC compatibility requires matching MI scaling

<!-- EOF -->