# gPAC Manuscript Outline

## Title Options
1. "gPAC: GPU-Accelerated Differentiable Phase-Amplitude Coupling Enabling Large-Scale Neuroscience"
2. "158-Fold Acceleration of Phase-Amplitude Coupling Analysis Through GPU Optimization"
3. "Differentiable Phase-Amplitude Coupling: Bridging Neuroscience and Deep Learning"

## Abstract (250 words)
- **Problem**: PAC analysis is computationally prohibitive for large datasets
- **Solution**: GPU-accelerated, fully differentiable implementation
- **Results**: 158x speedup, enables NeuroVista-scale analysis (69 days → 10.5 hours)
- **Impact**: First differentiable PAC, enables ML integration, real-time analysis

## 1. Introduction
### 1.1 Background
- PAC as biomarker in neuroscience
- Computational challenges with current tools
- Need for statistical validation (surrogates)
- Emerging need for ML integration

### 1.2 Current Limitations
- TensorPAC: CPU-bound, sequential processing
- PACTools: Limited functionality
- No differentiable implementations
- Impractical for large datasets (NeuroVista example)

### 1.3 Our Contribution
- First GPU-accelerated PAC
- Fully differentiable for ML
- 158x faster than state-of-the-art
- Enables new scientific applications

## 2. Methods
### 2.1 Algorithm Design
- Vectorized bandpass filtering
- Parallel Hilbert transform  
- Broadcasting for modulation index
- Memory-efficient returns

### 2.2 GPU Optimization
- Eliminated dictionary overhead
- Reduced loops from O(n⁴) to O(n²)
- Utilized PyTorch operations
- Mixed precision support

### 2.3 Differentiability
- Maintained gradient flow
- Soft binning for phase
- Compatible with autograd
- End-to-end trainable

### 2.4 Implementation
- Pure PyTorch (no custom CUDA)
- Cross-platform compatible
- Open source (MIT license)

## 3. Results
### 3.1 Performance Benchmarks
- **Figure 1**: Speed comparison (1 GPU vs 64 CPUs)
- **Table 1**: Scaling with data size
- **Figure 2**: Memory efficiency

### 3.2 Accuracy Validation
- **Figure 3**: Synthetic data recovery
- **Figure 4**: Comparison with TensorPAC
- Correlation analysis: r > 0.95

### 3.3 Statistical Testing
- **Figure 5**: Surrogate performance
- 200 surrogates: 11s vs 32 min
- Enables proper multiple comparisons

### 3.4 Real-World Application
- **Figure 6**: NeuroVista processing time
- 15 patients, months of data
- Seizure prediction pipeline
- **Figure 7**: PAC patterns in epilepsy

## 4. Discussion
### 4.1 Technical Innovations
- Dictionary overhead discovery
- Broadcasting strategy
- Differentiability preservation

### 4.2 Scientific Impact
- Enables population studies
- Real-time BCI applications
- ML/DL integration possible
- Reproducible science

### 4.3 Limitations
- Requires GPU for full benefit
- Memory limited by GPU VRAM
- Single GPU implementation (multi-GPU future)

### 4.4 Future Directions
- Multi-GPU support
- Custom CUDA kernels
- Additional PAC methods
- Integration with fMRI

## 5. Conclusion
- Transformed PAC from impractical to routine
- Enables new neuroscience at scale
- Bridge between neuroscience and AI
- Open source for community

## Supplementary Material
### S1. Detailed Benchmarks
- Extended performance tests
- Hardware configurations
- Profiling results

### S2. Code Examples
- Basic usage
- ML integration
- Seizure detection pipeline

### S3. Mathematical Details
- Soft binning formulation
- Gradient derivations
- Numerical stability

## Key Figures (7 main + supplementary)
1. **Performance comparison** (bar chart + speedup heatmap)
2. **Accuracy validation** (PAC matrices side-by-side)
3. **Surrogate statistics** (time scaling)
4. **NeuroVista timeline** (69 days vs 10.5 hours)
5. **Seizure PAC patterns** (real application)
6. **Differentiability demo** (gradient flow)
7. **Architecture diagram** (method overview)

## Target Length
- Main: 4,000 words
- Methods: 1,500 words
- Results: 1,500 words
- Discussion: 1,000 words

## Key Messages
1. **Speed**: 158x faster, enables impossible → routine
2. **Accuracy**: Validated against gold standard
3. **Innovation**: First differentiable PAC
4. **Impact**: Real neuroscience applications
5. **Open**: Available for community use

## Reviewer Concerns to Address
1. "Why not custom CUDA?" → PyTorch is sufficient, more maintainable
2. "Cherry-picked benchmark?" → Show multiple scenarios
3. "Accuracy vs speed tradeoff?" → No tradeoff, identical algorithm
4. "Limited to NVIDIA?" → PyTorch supports AMD ROCm
5. "Memory limitations?" → Address with chunking strategy

## Writing Timeline (2 weeks)
- Days 1-3: Introduction & Methods
- Days 4-6: Results with all figures
- Days 7-9: Discussion & Conclusion  
- Days 10-12: Revision & polish
- Days 13-14: Supplementary & final check

---
Ready to start writing! 📝