# Feature Requests Index

This directory contains granular feature requests for the gPAC project, organized by priority and completion status.

## Completed Features ✅

The following feature requests have been successfully implemented and moved to the `completed/` directory:

- ✅ [01_bandpass_filter_tensorpac_compat.md](completed/01_bandpass_filter_tensorpac_compat.md) - Fixed filter compatibility (0.001 → 0.999 correlation)
- ✅ [02_differentiable_hilbert.md](completed/02_differentiable_hilbert.md) - Verified Hilbert transform is fully differentiable
- ✅ [03_differentiable_modulation_index.md](completed/03_differentiable_modulation_index.md) - Implemented DifferentiableModulationIndex with soft binning
- ✅ [05_gradient_testing_suite.md](completed/05_gradient_testing_suite.md) - Created comprehensive gradient validation tests

## Pending Features

### Medium Priority
- ✅ [04_performance_optimization.md](04_performance_optimization.md) - Performance optimization (COMPLETED - achieved 32-108x speedup!)
- [08_refactor_v01_mode.md](08_refactor_v01_mode.md) - Remove v01_mode legacy parameter from production API

### Low Priority
- [06_edge_mode_support.md](06_edge_mode_support.md) - Additional edge handling modes for filtering  
- [07_surrogate_data_methods.md](07_surrogate_data_methods.md) - Statistical validation via surrogate data

## Key Achievements

- **Bandpass filter**: Achieved >99.9% correlation with TensorPAC (Phase r=0.999, Amplitude r=1.000)
- **Differentiability**: All core modules verified differentiable with gradient tests
- **Modulation Index**: Both standard and differentiable versions available
- **Test Coverage**: Comprehensive gradient checking suite with torch.autograd.gradcheck

## Original Feature Requests
- [tensorpac_comparison.md](tensorpac_comparison.md) - Original comprehensive comparison request
- [differentiable_modules.md](differentiable_modules.md) - Original differentiability request