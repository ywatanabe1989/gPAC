# gPAC vs Tensorpac Comparison Experiments

This directory contains rigorous scientific benchmarking experiments comparing gPAC and Tensorpac libraries for Phase-Amplitude Coupling (PAC) analysis.

## Experimental Design Principles

### 1. Fair Comparison Framework
- **Equivalent parameter mapping** between libraries
- **Separate measurement** of initialization vs computation costs
- **Realistic usage patterns** reflecting actual research workflows
- **Statistical rigor** with multiple repetitions and significance testing

### 2. Benchmarking Categories

#### A. Initialization Benchmarks (`01_initialization/`)
- Cold start performance (API-style usage)
- Warm instance performance (class-based usage)
- Memory allocation overhead
- Device transfer costs

#### B. Computational Benchmarks (`02_computation/`)
- Signal length scaling
- Frequency resolution scaling
- Channel count scaling
- Permutation testing scaling

#### C. Workflow Benchmarks (`03_workflows/`)
- Exploratory analysis patterns
- Production pipeline efficiency
- Real-time/streaming processing
- Memory-constrained scenarios

#### D. Statistical Analysis (`04_analysis/`)
- Performance distribution analysis
- Significance testing
- Effect size calculations
- Confidence intervals

## Directory Structure

```
./scripts/vs_tensorpac/
├── README.md                    # This file
├── config/                      # Experiment configurations
│   ├── base_config.yaml        # Default parameters
│   ├── scaling_configs.yaml    # Parameter scaling ranges
│   └── hardware_configs.yaml   # Device specifications
├── data/                        # Generated test data
│   ├── synthetic/              # Controlled synthetic signals
│   └── real/                   # Real EEG/MEG data samples
├── experiments/                 # Core experiment implementations
│   ├── __init__.py
│   ├── base_experiment.py      # Abstract base class
│   ├── initialization_exp.py   # Initialization benchmarks
│   ├── computation_exp.py      # Computational benchmarks
│   ├── workflow_exp.py         # Workflow benchmarks
│   └── utils.py                # Shared utilities
├── results/                     # Experimental results
│   ├── raw/                    # Raw timing/memory data
│   ├── processed/              # Analyzed results
│   └── figures/                # Generated plots
├── analysis/                    # Statistical analysis scripts
│   ├── statistical_tests.py    # Significance testing
│   ├── visualization.py        # Result plotting
│   └── report_generator.py     # Automated reporting
└── run_experiments.py           # Main experiment runner
```

## Usage

### Quick Start
```bash
# Install dependencies
pip install tensorpac

# Run all experiments
python run_experiments.py --config config/base_config.yaml

# Run specific experiment category
python run_experiments.py --category initialization
python run_experiments.py --category computation
python run_experiments.py --category workflows

# Generate analysis report
python analysis/report_generator.py --results results/raw/
```

### Custom Experiments
```bash
# Run with custom parameters
python run_experiments.py --config config/custom_config.yaml

# Run scaling experiments
python run_experiments.py --config config/scaling_configs.yaml --category computation

# CPU-only comparison
python run_experiments.py --device cpu

# GPU comparison (gPAC advantage)
python run_experiments.py --device cuda
```

## Experimental Controls

### 1. Hardware Standardization
- Consistent hardware specifications
- Temperature monitoring
- CPU/GPU load isolation
- Memory state reset between experiments

### 2. Software Standardization
- Fixed library versions
- Identical random seeds
- Consistent data types (float32)
- Standardized error handling

### 3. Statistical Controls
- Multiple repetitions (n≥30)
- Outlier detection and handling
- Confidence interval reporting
- Effect size calculations

## Key Research Questions

1. **Initialization Efficiency**: How do setup costs compare between libraries?
2. **Computational Scaling**: How does performance scale with data complexity?
3. **Memory Efficiency**: What are the memory usage patterns under different scenarios?
4. **Workflow Optimization**: Which library is more efficient for different research patterns?
5. **GPU Acceleration Benefits**: What are the practical benefits of GPU acceleration?

## Reproducibility

All experiments are designed for full reproducibility:
- Fixed random seeds
- Version-locked dependencies
- Hardware specification logging
- Complete parameter logging
- Automated result validation

## Citation

If you use these benchmarks in your research, please cite:
```
[Citation to be added upon publication]
```