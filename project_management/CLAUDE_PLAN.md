# CLAUDE_PLAN for gPAC Project

## Overview
The gPAC (GPU-accelerated Phase-Amplitude Coupling) project provides high-performance PAC analysis tools with PyTorch backend support. The main goals are to create comprehensive examples and demos that showcase the package's capabilities compared to existing solutions like TensorPAC.

## Key Components Completed
1. **Core Module Examples** (All using mngs framework):
   - BandPassFilter: Frequency filtering with TensorPAC compatibility
   - Hilbert: Transform for phase/amplitude extraction
   - ModulationIndex: KL divergence-based PAC measurement
   - PAC: Main PAC analysis class
   - SyntheticDataGenerator: Test signal generation
   - Profiler: Performance monitoring tools

2. **Infrastructure**:
   - All examples create `_out` directories via mngs framework
   - Visualization fixes applied (matplotlib compatibility)
   - Simplified versions created for quick testing

## Outstanding Tasks (from USER_PLAN.md)

### 1. README Demo Enhancement (`./examples/readme_demo.py`)
**Requirements**:
- Use synthetic data from gPAC package
- Create GIF showing:
  - Top: Raw synthetic signal
  - Bottom left: PAC by gPAC
  - Bottom center: PAC by TensorPAC  
  - Bottom right: Difference (gPAC - TensorPAC)
- X/Y axis labels must be in Hz
- Display calculation speed comparison
- Show ground truth PAC target range

**References**:
- `/home/ywatanabe/proj/mngs_repo/src/mngs/dsp/utils/pac.py`
  - Use p.idpac = (2,0,0) for MI
  - hres → n_bands = 100
  - mres → n_bands = 70

### 2. Real-world Demo (`./examples/readme_demo_realworld.py`)
**Requirements**:
- Use real EEG data during cognitive task
- Find easily downloadable dataset
- Demonstrate PAC analysis on real neural data

## Implementation Strategy

### For readme_demo.py:
1. Import both gPAC and TensorPAC
2. Generate synthetic PAC signal with known coupling
3. Run PAC analysis with both libraries
4. Create 4-panel visualization
5. Time both implementations
6. Save as animated GIF

### For readme_demo_realworld.py:
1. Research open EEG datasets (MNE-Python datasets, OpenNeuro)
2. Download cognitive task EEG data
3. Preprocess (bandpass filter, artifact removal)
4. Run PAC analysis
5. Visualize results with clinical interpretation

## Technical Considerations
- Ensure TensorPAC compatibility mode for fair comparison
- Use consistent parameters (fs, n_bands, etc.)
- Profile memory usage for large datasets
- Consider batch processing for real-world data

## Success Metrics
- Visual clarity of PAC differences
- Speed improvement demonstrated (>10x expected)
- Real-world applicability shown
- Documentation completeness