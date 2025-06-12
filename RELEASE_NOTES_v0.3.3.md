# Release Notes v0.3.3

## Improvements

### Simplified PAC output structure
- **Change**: Streamlined the output dictionary to include only essential fields
- **Removed fields**: 
  - `n_perm`, `fs`, `trainable` (these are instance attributes, not per-computation results)
  - `surrogate_mean`, `surrogate_std` (redundant with z-score information)
  - `phase_bin_edges` (redundant when `phase_bin_centers` is provided)
- **Rationale**: Cleaner, more focused output containing only the core computational results
- **Impact**: More intuitive API with less clutter in the returned dictionary

### Current output structure
```python
{
    "pac": pac_values,                    # Core PAC values
    "phase_bands_hz": phase_bands,         # Phase frequency bands used
    "amplitude_bands_hz": amplitude_bands, # Amplitude frequency bands used
    "pac_z": z_scores,                    # Z-scores (when n_perm is used)
    "amplitude_distributions": amp_dists,  # (when compute_distributions=True)
    "phase_bin_centers": phase_centers,    # (when compute_distributions=True)
}
```

## Technical Details

### Changed Files
- `src/gpac/_PAC.py`: Simplified return dictionary in forward method

## Upgrade Instructions
```bash
pip install --upgrade gpu-pac==0.3.3
```

## Compatibility Note
This change simplifies the output structure. If you were using the removed fields, you can still access them as attributes of the PAC instance (e.g., `pac_instance.n_perm`, `pac_instance.fs`).