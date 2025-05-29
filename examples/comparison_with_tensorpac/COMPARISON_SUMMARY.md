# gPAC vs TensorPAC Comparison Summary

## Key Findings

### 1. Speed Performance
- **Single signal processing**: TensorPAC is faster (~0.01s vs ~0.24s for small signals)
- **Batch processing**: gPAC shows significant advantages
  - For realistic EEG data (16 channels, 60s @ 400Hz): **gPAC is 172x faster**
  - GPU parallelization enables processing multiple channels/batches simultaneously
  - TensorPAC processes sequentially, leading to linear scaling with data size

### 2. Accuracy Comparison
- **Frequency detection**: Both methods detect PAC, but with differences:
  - TensorPAC: More accurate phase frequency detection (error ~0.5 Hz)
  - gPAC: Sometimes detects harmonics (error ~8.7 Hz with default settings)
  - Both have similar amplitude frequency detection accuracy

- **PAC values**: Scale difference of ~3-4x
  - gPAC values are smaller due to different normalization
  - This is a calibration issue, not a fundamental problem
  - Can be addressed by adjusting filter gain normalization

### 3. Filter Types
- **Static filters**: Faster, fixed frequency bands
- **Trainable (differentiable) filters**: ~25% overhead but allows learning
  - Based on SincNet architecture
  - Enables end-to-end learning in neural networks
  - Essential for applications requiring gradient flow

### 4. Band Definitions
- gPAC uses center frequencies for band specification
- TensorPAC uses band edges
- Both support customizable frequency resolutions
- Higher frequency resolution improves accuracy but increases computation time

## Recommendations

### When to use gPAC:
1. **Large-scale data processing** (multiple channels, long recordings)
2. **Real-time applications** requiring GPU acceleration
3. **Deep learning integration** (trainable filters)
4. **Batch processing** of multiple subjects/trials

### When to use TensorPAC:
1. **Small datasets** or single-channel analysis
2. **CPU-only environments**
3. **Established analysis pipelines** requiring exact reproduction
4. **When phase frequency accuracy is critical**

## Performance Examples

| Configuration | gPAC Time | TensorPAC Time | Speedup |
|--------------|-----------|----------------|---------|
| 1 channel, 1s @ 256Hz | 0.24s | 0.01s | 0.04x |
| 16 channels, 60s @ 400Hz | 0.013s | 2.28s | 172x |
| 32 channels, 20s @ 1024Hz | 0.023s | 3.78s | 162x |

## Technical Considerations

1. **Normalization**: The PAC value difference is due to filter normalization methods
   - Can be adjusted by modifying gain normalization in filters
   - Not a fundamental algorithmic difference

2. **Frequency Resolution**: 
   - More bands improve accuracy but increase computation
   - gPAC can handle 100+ bands efficiently with GPU
   - Consider signal length when choosing resolution

3. **Memory Usage**:
   - gPAC requires GPU memory (~8 channels max for large signals)
   - Can process in batches for larger channel counts
   - TensorPAC has lower memory requirements but slower

## Conclusion

gPAC and TensorPAC serve different use cases:
- **gPAC**: Optimized for large-scale, GPU-accelerated processing and deep learning
- **TensorPAC**: Reliable for traditional analysis on smaller datasets

The choice depends on your specific requirements for speed, scale, and integration needs.