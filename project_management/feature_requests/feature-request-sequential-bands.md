# Feature Request: Sequential (Non-Overlapping) Frequency Bands

## Date: 2025-06-03
## Priority: High
## Status: Open

## Summary
Add support for sequential (non-overlapping) frequency bands in gPAC to improve frequency localization accuracy, particularly for low frequencies like 6 Hz.

## Current Issue
gPAC currently uses overlapping frequency bands with the formula:
- Phase bands: `[center - center/4, center + center/4]`
- Amplitude bands: `[center - center/8, center + center/8]`

This causes:
1. Poor frequency localization (6 Hz signal spreads across 5 overlapping bands)
2. Misalignment between band definitions and reported center frequencies
3. Non-gradual comodulogram appearance
4. Difficulty in comparing results with TensorPAC sequential bands

## Proposed Solution

### 1. Add Band Mode Parameter
```python
pac = PAC(
    ...,
    band_mode='sequential',  # 'overlapping' (default) or 'sequential'
    ...
)
```

### 2. Implement Sequential Band Calculation
```python
def _calc_sequential_bands(start_hz, end_hz, n_bands):
    """Calculate sequential non-overlapping bands."""
    edges = torch.linspace(start_hz, end_hz, n_bands + 1)
    bands = torch.stack([edges[:-1], edges[1:]], dim=1)
    return bands
```

### 3. Update Band Center Calculation
Ensure band centers accurately reflect the actual passband:
```python
if band_mode == 'sequential':
    self.pha_mids = (pha_bands[:, 0] + pha_bands[:, 1]) / 2
else:
    # Current overlapping logic
```

## Benefits
1. **Better frequency precision**: Especially for low frequencies (2-10 Hz)
2. **TensorPAC compatibility**: Easier comparison with sequential band implementations
3. **Clearer interpretation**: Each frequency uniquely belongs to one band
4. **Improved visualization**: Smoother, more gradual comodulogram

## Implementation Steps
1. Add `band_mode` parameter to PAC and BandPassFilter classes
2. Implement sequential band calculation methods
3. Update filter initialization to support both modes
4. Ensure band centers match actual filter passbands
5. Update documentation and examples
6. Add tests comparing overlapping vs sequential modes

## Backward Compatibility
- Default to 'overlapping' mode to maintain current behavior
- No breaking changes to existing API

## Testing
- Verify 6 Hz x 80 Hz synthetic signal detection improves
- Compare with TensorPAC sequential band results
- Ensure gradual comodulogram appearance

## References
- `/docs/IMPORTANT-Tensorpac-Band-Definitions.md`
- Issue identified: 2025-06-03 (6 Hz detection problem)
- Related: Adaptive filtering implementation