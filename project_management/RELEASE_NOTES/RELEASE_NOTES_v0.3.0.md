# Release Notes - v0.3.0

## Major Changes

### 🔄 Breaking Changes: Default Parameter Updates

Based on PAC literature recommendations and best practices, we've updated the default parameters:

1. **Frequency Band Spacing**: Changed from `log` to `linear`
   - Linear spacing is more standard in neuroscience applications
   - Provides more intuitive band distribution

2. **Phase Frequency Range**: Changed from `(4, 30)` Hz to `(2, 20)` Hz
   - Better captures key neural rhythms: delta (2-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-20 Hz)
   - Aligns with standard neuroscience frequency bands
   - Reduces issues with overly wide bandwidths at higher frequencies

3. **Amplitude Frequency Range**: Changed from `(60, 150)` Hz to `(60, 160)` Hz
   - Extended upper bound to better capture high gamma activity
   - Consistent with common PAC analysis practices

### 📚 Literature Alignment

These changes address feedback that the previous default bandwidth formula (f/2 for phase) could create excessively wide bands at higher frequencies. For example:
- Old: 25 Hz center → 12.5 Hz bandwidth (too wide for accurate phase estimation)
- New: Linear spacing with more appropriate bandwidths

## Migration Guide

If you need to maintain the previous behavior, explicitly specify the old defaults:

```python
# Old behavior (v0.2.x)
pac = PAC(
    seq_len=seq_len,
    fs=fs,
    pha_range_hz=(4, 30),
    amp_range_hz=(60, 150),
    spacing='log'  # Note: spacing parameter not directly available in PAC
)

# New behavior (v0.3.0) - automatic with new defaults
pac = PAC(seq_len=seq_len, fs=fs)
```

## Bug Fixes

- None in this release

## Other Changes

- Updated documentation to reflect new default parameters
- All affected modules updated consistently:
  - `_StaticBandPassFilter.py`
  - `_PooledBandPassFilter.py`
  - `_BandPassFilter.py`
  - `_PAC.py`

## Compatibility

- **Python**: 3.8+
- **PyTorch**: 1.8.0+
- **CUDA**: 11.0+ (for GPU support)

## Acknowledgments

Thanks to user feedback regarding PAC literature recommendations and bandwidth considerations.

---

**Full Changelog**: https://github.com/ywatanabe1989/gPAC/compare/v0.2.1...v0.3.0