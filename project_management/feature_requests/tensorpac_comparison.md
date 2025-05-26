# Feature Request: TensorPAC Comparison and Validation

**Date:** 2025-05-26  
**Priority:** High  
**Status:** ✅ COMPLETED - Analysis tools implemented, root cause identified  

## Overview

Compare each gPAC module with corresponding TensorPAC implementation to ensure compatibility and correctness.

## Requirements

### Target TensorPAC Configuration
- **Method**: Tort (Torr)  
- **idpac**: (2, 0, 0)
- **Source**: `tensorpac_source/` directory

### Modules to Compare

1. **BandPass Filtering**
   - Compare filter design and application
   - Validate frequency response matching
   - Test edge handling and padding

2. **Hilbert Transform**  
   - Compare analytical signal computation
   - Validate phase extraction accuracy
   - Test amplitude extraction

3. **Modulation Index Calculation**
   - Compare MI computation methods
   - Validate statistical measures
   - Test binning and circular statistics

4. **PAC Pipeline**
   - End-to-end comparison
   - Validate complete workflow
   - Compare performance metrics

## Implementation Plan

### Phase 1: Analysis ✅ COMPLETED
- [x] Study TensorPAC implementation in `tensorpac_source/`
- [x] Identify key differences in algorithms
- [x] Document compatibility requirements

### Phase 2: Testing ✅ COMPLETED
- [x] Create comparison test suite (`scripts/vs_tensorpac/compare_tensorpac_gpac.py`)
- [x] Generate synthetic test data
- [x] Run parallel computations (gPAC vs TensorPAC)
- [x] Validate numerical accuracy - FOUND MAJOR DISCREPANCY

### Phase 3: Analysis Results ✅ COMPLETED
- [x] Root cause identified: Bandpass filter implementation differences
- [x] Created detailed debugging tools (`scripts/vs_tensorpac/debug_comparison.py`)
- [x] Performance analysis: TensorPAC 8x faster than gPAC

## 🔍 KEY FINDINGS

### Comparison Results
- **Overall correlation**: 41% (target: ≥99%)
- **Root issue**: Bandpass filtering stage correlation = 0.001
- **Frequency agreement**: Perfect match
- **Performance**: gPAC 8x slower than TensorPAC

### Technical Analysis
1. **Filtering Stage**: Complete disagreement (r=0.001) - different filter designs
2. **Hilbert Transform**: Amplitude correlation good (r=0.99), phase poor (r=-0.86)
3. **Modulation Index**: Cannot validate due to upstream errors

## Success Criteria Evaluation

- ❌ ≥99% numerical agreement on synthetic data (ACHIEVED: 41%)
- ❌ Performance within 10% of TensorPAC (ACHIEVED: 12% - 8x slower)
- ❌ All tests pass with idpac=(2,0,0) configuration (FAILED: filter mismatch)
- ✅ Documented compatibility report (COMPLETED)

## 🎯 DELIVERABLES COMPLETED

1. **Comprehensive comparison suite**: `scripts/vs_tensorpac/compare_tensorpac_gpac.py`
2. **Step-by-step debugging tool**: `scripts/vs_tensorpac/debug_comparison.py`  
3. **Root cause analysis**: Bandpass filter implementation discrepancy
4. **Performance benchmarks**: Detailed runtime and accuracy metrics
5. **Integration ready**: Tools can be run by future agents

## 📋 NEXT STEPS FOR FUTURE AGENTS

**Immediate priority**: Fix bandpass filter compatibility
- **Issue**: gPAC and TensorPAC filters produce completely different outputs
- **Impact**: All downstream PAC calculations are invalid
- **Files to investigate**: `src/gpac/_BandPassFilter.py` vs TensorPAC spectral module
- **Goal**: Achieve >95% filter output correlation before proceeding

## Test Data Requirements

- **Synthetic signals** with known PAC characteristics
- **Real neural data** for validation
- **Edge cases** (noise, artifacts, different sampling rates)

## Related Files

- `tensorpac_source/` (reference implementation)
- `scripts/vs_tensorpac/` (comparison scripts)
- `tests/custom/test_values_comparison.py`