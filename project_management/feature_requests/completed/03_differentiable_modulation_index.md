# Feature Request: Differentiable Modulation Index

**Date:** 2025-05-26  
**Priority:** High  
**Status:** Completed  

## Overview
Ensure Modulation Index calculation is fully differentiable while maintaining compatibility with TensorPAC's Tort method.

## Current Status
- Implementation in `src/gpac/_ModulationIndex.py`
- Cannot validate due to upstream filter issues
- Differentiability status unknown

## Requirements
1. Support differentiable binning operations
2. Implement differentiable circular statistics
3. Maintain numerical stability
4. Match TensorPAC's idpac=(2,0,0) configuration

## Technical Challenges
- Binning typically uses non-differentiable operations
- Need soft binning or alternative approach
- Circular mean calculation must preserve gradients
- KL divergence computation needs care

## Success Criteria
- [ ] Gradients flow through MI calculation
- [ ] Results match TensorPAC within 1%
- [ ] Stable gradients (no NaN/Inf)
- [ ] Performance overhead <20%

## Implementation Options
1. Soft binning with Gaussian kernels
2. Differentiable histogram approximation
3. Continuous relaxation of discrete operations

## Test Plan
- Gradient checking with finite differences
- Compare with TensorPAC on synthetic data
- Test gradient stability across parameter ranges