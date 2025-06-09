# TensorPAC Band Comparison Notes

## Critical Finding

TensorPAC's frequency band handling is **significantly different** from gPAC and requires careful configuration for fair comparison.

## Key Differences

### 1. String Configurations Override Everything

When using string configurations ('lres', 'mres', 'hres', 'demon', 'hulk'), TensorPAC:
- **Ignores** any frequency ranges you specify
- Uses hardcoded ranges: phase (2-20 Hz), amplitude (60-160 Hz)
- Creates overlapping bands using formulas: [f - f/4, f + f/4] for phase, [f - f/8, f + f/8] for amplitude

### 2. Band Count Mapping

| Config | Band Count | TensorPAC Name |
|--------|------------|----------------|
| Low    | 10         | 'lres'         |
| Medium | 30         | 'mres'         |
| High   | 50         | 'hres'         |
| Demon  | 70         | 'demon'        |
| Hulk   | 100        | 'hulk'         |

### 3. Correct Usage for Comparison

```python
# WRONG - Uses TensorPAC's internal frequency ranges
pac = Pac(idpac=(2, 0, 0), f_pha='mres', f_amp='mres')

# WRONG - Single frequencies get converted incorrectly  
f_pha = np.linspace(2, 20, 10)  # Creates 9 bands, not 10!
pac = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)

# CORRECT - Explicit band pairs for exact control
pha_edges = np.linspace(2, 20, 11)  # 11 edges = 10 bands
amp_edges = np.linspace(30, 100, 11)
f_pha = np.c_[pha_edges[:-1], pha_edges[1:]]  # [[2, 3.8], [3.8, 5.6], ...]
f_amp = np.c_[amp_edges[:-1], amp_edges[1:]]  # [[30, 37], [37, 44], ...]
pac = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
```

## Benchmark Implications

1. **Always use explicit band pairs** when comparing gPAC with TensorPAC
2. **Match band counts** to TensorPAC's standard configurations (10, 30, 50, 70, 100)
3. **Document frequency ranges** explicitly - don't assume defaults match

## Performance Comparison Notes

- With corrected band definitions, fair performance comparisons are possible
- gPAC shows better scaling with increased frequency resolution
- Both implementations produce different absolute PAC values due to normalization differences
- Correlation between implementations remains moderate even with matched bands

## References

- Original documentation: `/home/ywatanabe/proj/gPAC/docs/IMPORATANT-Band-Definitions-in-Tensorpac.md`
- TensorPAC source: `tensorpac/utils.py` - `pac_vec()` function
- Corrected benchmark: `examples/performance/comprehensive_benchmark_refined.py`