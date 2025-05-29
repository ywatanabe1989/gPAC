# Next Steps for gPAC Project

## Completed Today (2025-05-29)

1. ✅ **Fixed examples to use mngs framework properly**
   - Updated visualization methods (set_xyt → set_xlabel/ylabel/title)
   - Fixed output directory paths
   - All examples now create `_out` directories

2. ✅ **Created example_Profiler.py**
   - Demonstrates performance monitoring with gPAC
   - Shows CPU, RAM, and GPU usage tracking
   - Generates performance reports

3. ✅ **Implemented README demo with GIF**
   - Created animated GIF showing PAC analysis
   - Shows ground truth PAC (θ=6 Hz → γ=80 Hz)
   - Displays GPU computation time (~1.8s)
   - Progressive visualization of signal and PAC matrix

4. ✅ **Downloaded Zen of Python to guidelines**
   - Added to `/home/ywatanabe/.claude/to_claude/guidelines/python/zen_of_python.md`

## Immediate Next Steps

### 1. Real-world EEG Demo Implementation
- **File**: `./examples/readme_demo_realworld.py`
- **Tasks**:
  - Research and download open-source EEG dataset
  - Good candidates:
    - MNE-Python sample datasets
    - OpenNeuro datasets
    - PhysioNet EEG data
  - Implement PAC analysis on cognitive task data
  - Show clinical relevance of PAC findings

### 2. TensorPAC Comparison Enhancement
- Fix TensorPAC integration in readme_demo.py
- Create side-by-side comparison showing:
  - Speed improvement (expected >10x)
  - Accuracy comparison
  - Memory usage differences

### 3. Documentation Updates
- Update main README.md with:
  - GIF animation from readme_demo
  - Performance benchmarks
  - Installation instructions
  - Example outputs

## Technical Considerations

1. **TensorPAC Compatibility**
   - High resolution (100 bands) causes issues
   - May need to reduce to 20-50 bands for comparison
   - Consider using mres (70 bands) instead of hres

2. **Real-world Data Challenges**
   - Need preprocessing pipeline (artifact removal, filtering)
   - Consider epoching around cognitive events
   - Handle multiple channels/subjects

3. **Performance Optimization**
   - Current GPU time: ~1.8s for 5s signal
   - Can optimize with batch processing
   - Consider fp16 mode for larger datasets

## Success Metrics
- ✅ GIF animation created
- ✅ Ground truth PAC correctly identified
- ✅ GPU acceleration demonstrated
- ⏳ TensorPAC comparison pending
- ⏳ Real-world demo pending

## References
- USER_PLAN specifies: p.idpac = (2,0,0) for MI
- hres → n_bands = 100
- mres → n_bands = 70
- See: `/home/ywatanabe/proj/mngs_repo/src/mngs/dsp/utils/pac.py`