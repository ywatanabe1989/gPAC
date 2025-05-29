# gPAC Examples

This directory contains example scripts demonstrating various features of the gPAC package. The structure mirrors the source code organization for easy reference.

## Directory Structure

### 1. `gpac/` - Core Module Examples
Examples for each core module, named with `example_` prefix:
- `example_PAC.py` - Basic PAC analysis workflow
- `example_BandPassFilter.py` - Bandpass filtering and performance demonstration
- `example_Hilbert.py` - Hilbert transform usage
- `example_ModulationIndex.py` - Modulation index calculation
- `example_SyntheticDataGenerator.py` - Generating synthetic PAC signals
- `example_basic_usage.py` - General usage examples (from README)
- `example_visualization.py` - Creating visualizations and diagrams
- `_Filters/` - Subdirectory for filter-specific examples (if needed)

### 2. `trainability/` - Trainable Features Examples
Examples demonstrating learnable/differentiable aspects:
- `example_differentiable_bucketize.py` - Differentiable histogram binning

### 3. `comparison_with_tensorpac/` - Comparison Examples
Examples comparing gPAC with other implementations:
- `example_mngs_comparison.py` - Comparison with mngs package
- `example_v01_comparison.py` - Comparison of different implementation modes

## Running Examples

Each example is self-contained and can be run directly:

```bash
# Core module examples
python examples/gpac/example_PAC.py
python examples/gpac/example_BandPassFilter.py
python examples/gpac/example_Hilbert.py

# Trainability example
python examples/trainability/example_differentiable_bucketize.py

# Comparison examples
python examples/comparison_with_tensorpac/example_mngs_comparison.py
```

## Example Structure

Each example follows a consistent pattern:
1. Import necessary modules
2. Create or load sample data
3. Demonstrate the module's functionality
4. Show results or save outputs
5. Include comments explaining key concepts

## Adding New Examples

When adding new examples:
1. Use the `example_` prefix for consistency
2. Place in the appropriate directory matching the module structure
3. Include docstring explaining what the example demonstrates
4. Add clear comments throughout the code
5. Keep examples focused on one main concept