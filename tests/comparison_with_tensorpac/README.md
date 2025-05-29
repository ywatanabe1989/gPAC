# TensorPAC vs gPAC Comparison Notes

This directory contains tests comparing gPAC implementations with TensorPAC to ensure compatibility and understand differences.

## TensorPAC Frequency Band Definitions

TensorPAC uses string specifications for frequency bands:

- **'lres'** (Low resolution): 10 frequency bands
  - Phase frequencies: [2, 3.4, 4.8, 6.3, 7.9, 9.6, 11.4, 13.3, 15.3, 17.5] Hz
  - Amplitude frequencies: [10.5, 19.4, 30.6, 44.2, 60.6, 80.0, 102.8, 129.3, 159.9, 195.0] Hz

- **'mres'** (Medium resolution): 30 frequency bands  
  - Phase frequencies: [2.0, 2.5, 3.0, ..., 17.5] Hz (30 bands)
  - Amplitude frequencies: [10.5, 13.5, 16.9, ..., 195.0] Hz (30 bands)

- **'hres'** (High resolution): 50 frequency bands
  - Phase frequencies: [2.0, 2.3, 2.6, ..., 17.5] Hz (50 bands)
  - Amplitude frequencies: [10.5, 12.4, 14.4, ..., 195.0] Hz (50 bands)

## Key Differences Found

### 1. Filter Implementations
- **gPAC**: Uses FIR filters (scipy.signal.firwin) with Kaiser window
- **TensorPAC**: 
  - Butterworth filters (default)
  - Morlet wavelet filters (when using wavelet method)
  
**Impact**: Butterworth filters produce ~3.4x higher power in filtered signals compared to FIR filters for phase bands.

### 2. Hilbert Transform
- **gPAC**: 
  - Standard mode: scipy.signal.hilbert
  - Differentiable mode: custom sigmoid-based approximation
- **TensorPAC**: scipy.signal.hilbert

**Impact**: Minimal differences when using standard mode.

### 3. Modulation Index Calculation
- **gPAC**: Supports multiple n_bins configurations
- **TensorPAC**: Uses idpac parameter
  - idpac=(2,0,0) corresponds to Tort MI method

**Impact**: When properly configured, MI values differ by only ~10% (scale factor ~1.1x).

### 4. Output Shape Convention
- **gPAC**: Returns shape (n_epochs, n_pha, n_amp)
- **TensorPAC**: Returns shape (n_pha, n_amp, n_epochs)

## Test Files

1. **test_bandpass_filter.py**: Compares bandpass filter implementations
   - Tests FIR (gPAC) vs Butterworth (TensorPAC) filters
   - Measures power ratio differences (~3.4x)
   - Validates filter frequency responses

2. **test_hilbert.py**: Compares Hilbert transform implementations
   - Tests amplitude and phase extraction
   - Compares gPAC's scipy.signal.hilbert with TensorPAC's implementation
   - Validates differentiable vs standard modes

3. **test_modulation_index.py**: Compares MI calculations
   - Module-level comparison of MI computation
   - Tests with synthetic PAC signals
   - Validates different binning strategies and idpac settings

4. **test_pac.py**: Full pipeline comparison
   - End-to-end PAC computation comparison
   - Tests complete workflow from raw signal to PAC values
   - Performance benchmarking and compatibility validation

## Usage Notes

When comparing with TensorPAC:
- Use `idpac=(2,0,0)` for Tort MI method
- Use `f_pha="hres", f_amp="mres"` to match default gPAC settings
- Account for the ~3.4x scale difference when comparing raw filter outputs
- Transpose outputs when comparing shapes

## Configuration for Compatibility

To achieve maximum compatibility with TensorPAC:

```python
# gPAC configuration
from gpac import PAC

pac = PAC(
    f_pha=[(2, 4), (4, 8), (8, 12)],  # or use string like "hres"
    f_amp=[(20, 40), (40, 80), (80, 160)],
    differentiable=False,  # Use standard scipy implementations
    n_bins=18  # Matches TensorPAC's default for Tort method
)

# TensorPAC configuration
from tensorpac import Pac

pac_tp = Pac(
    f_pha='hres',
    f_amp='mres', 
    idpac=(2, 0, 0),  # Tort MI method
    dcomplex='hilbert'  # Use Hilbert instead of wavelet
)
```