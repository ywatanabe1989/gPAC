# gPAC Examples Run Status

## Successfully Running Examples

1. **example_pac_trainability_simple.py**
   - Status: Runs successfully
   - Output: Saved to `./scripts/example_pac_trainability_simple/pac_trainability_simple_demo.png`
   - Note: Demonstrates SincNet filter trainability

2. **example_pac_trainability_working.py** 
   - Status: Runs successfully
   - Output: Saved to `./scripts/example_pac_trainability_working/pac_demo.png`
   - Note: Simple PAC detection demo

## Examples with Issues

### API Mismatches
- `example_PAC.py` - ImportError: cannot import name 'calculate_pac'
- `example_basic_usage.py` - AttributeError: 'dict' object has no attribute 'ndim'
- `example_pac_trainability.py` - AttributeError: 'dict' object has no attribute 'mean'

### Shape Mismatches
- `example_BandPassFilter.py` - ValueError: Input tensor has too many dimensions
- `example_bandpass_filter_comparison.py` - TypeError when TensorPAC not available

## Key Issues Found

1. **PAC returns dict not tensor**: The PAC class returns a dictionary with keys like 'pac', 'phase_frequencies', etc., but many examples expect a tensor.

2. **Filter shape expectations**: BandPassFilter expects 3D input but examples provide 4D.

3. **Missing TensorPAC**: Comparison examples fail gracefully when TensorPAC is not installed but don't create proper fallback visualizations.

4. **MNGS colorbar issue**: matplotlib colorbar doesn't work properly with mngs AxisWrapper.

## Recommendations

1. Update examples to match current API
2. Add input shape validation and helpful error messages
3. Make TensorPAC optional with proper fallbacks
4. Document expected input/output shapes clearly