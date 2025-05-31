# Bug Report: Examples Missing MNGS Framework Compliance

**Date**: 2025-05-31  
**Reporter**: auto-CLAUDE-mngs-20250531  
**Severity**: Medium  
**Category**: Framework Compliance  
**Status**: PARTIALLY RESOLVED

## Summary

Multiple example files in the gPAC project do not follow the MNGS framework, resulting in missing or incorrectly placed output directories. This violates the project guidelines that require all examples to use the MNGS framework.

## Root Causes

1. **Missing MNGS framework components** (mngs.gen.start/close)
2. **Incorrect `__file__` definition** - Examples that do have `__file__` use lowercase and filename only (e.g., `__file__ = "simple_pac_demo.py"`) instead of the full relative path that MNGS expects

## Affected Files

### Completely Missing MNGS Framework (6 files):
1. `./examples/ComparisonAdapters/quick_comparison_demo.py`
   - Missing: mngs.gen.start(), mngs.gen.close()
   - Has: mngs.io.save() for outputs

2. `./examples/gpac/example_bandpass_filter.py`
   - Missing: mngs.gen.start(), mngs.gen.close()
   - Has: mngs.io.save() for outputs

3. `./examples/performance_comparison.py`
   - Missing: ALL MNGS components
   - No mngs imports at all

4. `./examples/comprehensive_benchmark.py`
   - Missing: ALL MNGS components
   - No mngs imports at all

5. `./examples/performance_test.py`
   - Missing: ALL MNGS components
   - No mngs imports at all

6. `./examples/gpac/_Filters/simple_differentiable_filter_demo.py`
   - Missing: mngs.gen.start(), mngs.gen.close()
   - Has: mngs.io import for saving only

7. `./examples/gpac/_Filters/simple_static_filter_demo.py`
   - Missing: mngs.gen.start(), mngs.gen.close()
   - Has: mngs.io import for saving only

### Fully Compliant Examples (for reference):
- `./examples/gpac/simple_pac_demo.py` ✓
- `./examples/readme/readme_demo.py` ✓
- `./examples/epilepsty/epilepsy_classification_demo.py` ✓
- `./examples/handgrasping/hand_grasping_demo.py` ✓
- `./examples/cognitive_workload/cognitive_workload_demo.py` ✓
- `./examples/cognitive_workload/cognitive_workload_demo_zscore.py` ✓

## Impact

1. **Missing Output Directories**: Examples without MNGS framework don't create `_out` directories
2. **Inconsistent Structure**: Violates project guidelines requiring MNGS framework
3. **Poor Organization**: Outputs (if any) may be saved to incorrect locations
4. **No Logging**: Missing MNGS logging and configuration management

## Root Cause

Examples were either:
1. Created before MNGS framework adoption
2. Created as quick demos without following guidelines
3. Performance benchmarks that prioritize speed over framework compliance

## Recommended Fix

All non-compliant examples should be updated to follow the MNGS framework pattern:

```python
#!/usr/bin/env python3
import sys
import argparse
import matplotlib.pyplot as plt
import mngs

# Parameters
CONFIG = mngs.gen.load_configs()

def parse_args():
    parser = argparse.ArgumentParser(description="Example description")
    parser.add_argument("--param", type=str, default="default")
    return parser.parse_args()

def run_main(args):
    # Start
    CONFIG, CC, sdir = mngs.gen.start(__file__, args=args, pyplot_backend="Agg")
    
    # Main logic here
    # ...
    
    # Save outputs
    mngs.io.save(figure, sdir / "output.png")
    
    # Close
    mngs.gen.close()

if __name__ == "__main__":
    args = parse_args()
    run_main(args)
```

## Priority

Medium - While not critical for functionality, this affects project organization and violates guidelines.

## Status

**OPEN** - Requires fixing 7 example files