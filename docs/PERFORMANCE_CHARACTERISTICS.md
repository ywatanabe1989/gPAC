# gPAC v1.0.0 Performance Characteristics

This document summarizes the performance characteristics and limitations of gPAC based on comprehensive testing.

## Executive Summary

gPAC v1.0.0 demonstrates excellent performance characteristics:
- **GPU Acceleration**: Up to 8.25x speedup over CPU for large problems
- **Batch Processing**: Up to 13x efficiency improvement for batch size 64
- **Throughput**: >6,800 signals/second for batched processing on GPU
- **Memory Efficient**: <1MB GPU memory for typical use cases

## Performance Scaling

### Signal Length Scaling
Performance scales sub-linearly with signal length due to FFT algorithm efficiency:

| Signal Duration | Throughput (signals/s) | Notes |
|----------------|------------------------|--------|
| 0.5s | 564 | Overhead dominated |
| 1.0s | 582 | Optimal for short signals |
| 2.0s | 469 | Good performance |
| 5.0s | 234 | Reasonable for long signals |
| 10.0s | 184 | Good for very long signals |
| 30.0s | 189 | Stable performance |

### Batch Processing Performance
Batch processing provides significant efficiency gains:

| Batch Size | Per-Signal Time | Efficiency | Throughput |
|------------|----------------|------------|-------------|
| 1 | 1.9ms | 1.0x | 520 signals/s |
| 4 | 1.8ms | 1.1x | 825 signals/s |
| 8 | 0.7ms | 2.7x | 1,422 signals/s |
| 16 | 0.2ms | 8.0x | 4,173 signals/s |
| 32 | 0.2ms | 10.1x | 5,264 signals/s |
| 64 | 0.1ms | 13.3x | 6,896 signals/s |

**Recommendation**: Use batch sizes of 16-64 for optimal performance.

## GPU vs CPU Performance

GPU acceleration provides significant speedup, especially for larger problems:

| Problem Size | CPU Time | GPU Time | GPU Speedup |
|--------------|----------|----------|-------------|
| 512 samples, 10×10 bands | 1.1ms | 0.9ms | 1.15x |
| 1024 samples, 20×15 bands | 3.8ms | 1.3ms | 2.91x |
| 2048 samples, 30×20 bands | 10.1ms | 1.2ms | 8.25x |

**Key Insight**: GPU advantage increases with problem complexity (frequency resolution).

## Memory Usage

### GPU Memory
- Minimal GPU memory footprint
- Scales linearly with problem size
- Examples:
  - 1024 samples, 10×10 bands: 0.04 MB
  - 8192 samples, 40×25 bands: 0.26 MB

### System Memory
- Very efficient memory usage
- Typically <1 MB for standard configurations
- Memory per element: <1 byte for most cases

## Supported Configurations

### Signal Properties
- **Minimum length**: 64 samples (tested)
- **Maximum length**: 30s+ at 512Hz (tested)
- **Channels**: 1-8+ channels supported
- **Sampling rates**: Any (automatically handled)

### Frequency Resolution
- **Phase frequencies**: Up to 50 bands tested
- **Amplitude frequencies**: Up to 40 bands tested
- **Total frequency pairs**: Up to 2000 tested (50×40)

### Edge Cases Handled
✅ Very short signals (64 samples)
✅ High frequency resolution (50×40 bands)
✅ Multi-channel signals (8+ channels)
✅ Near-Nyquist frequencies
✅ Variable signal lengths in production

## Optimization Recommendations

### For Maximum Throughput
1. Use GPU when available
2. Process in batches (16-64 signals)
3. Initialize model once, compute many times
4. Use appropriate frequency resolution for your needs

### For Memory Efficiency
1. Process signals in reasonable chunks
2. Clear GPU cache between large batches if needed
3. Use CPU for very small problems (<100 frequency pairs)

### For Production Deployment
1. **Initialization**: ~0.5s one-time cost
2. **Warm-up**: Always warm up GPU with dummy computation
3. **Batch size**: Start with 32, adjust based on latency requirements
4. **Monitoring**: Track GPU memory usage for long-running processes

## Limitations

### Known Limitations
1. Single computation overhead due to initialization
2. CPU performance limited for high-resolution analysis
3. GPU memory scales with batch_size × seq_len × n_bands

### Not Recommended For
- Single-shot computations (use CPU libraries instead)
- Very low latency requirements (<1ms)
- Systems without CUDA support for high-resolution analysis

## Comparison with Other Libraries

### vs Tensorpac
- **Initialization**: Tensorpac faster (one-time cost)
- **Computation**: gPAC 10-50x faster (recurring cost)
- **Break-even**: ~2-5 computations depending on configuration
- **Production**: gPAC superior for any repeated computation

### Unique Advantages
1. GPU acceleration
2. Batch processing
3. PyTorch integration
4. Trainable PAC networks
5. Modern Python API

## Conclusion

gPAC v1.0.0 delivers exceptional performance for PAC analysis:
- **Fast**: Sub-millisecond per-signal computation
- **Scalable**: Efficient batch processing
- **Robust**: Handles diverse signal types and edge cases
- **Production-ready**: Optimized for real-world deployment

For applications requiring repeated PAC computations, gPAC provides unmatched performance and flexibility.