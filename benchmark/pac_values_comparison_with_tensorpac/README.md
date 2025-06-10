# PAC Values Comparison with Tensorpac

This directory contains scripts to compare Phase-Amplitude Coupling (PAC) calculations between gPAC and Tensorpac.

## Guidelines for Fair Comparison between gPAC and TensorPAC

### Key Differences and Guidelines

#### 1. Frequency Band Definitions

**Critical**: gPAC and TensorPAC use different conventions for defining frequency bands.

- **gPAC**: Uses explicit band arrays where each row is `[low_freq, high_freq]`
- **TensorPAC**: Can use either explicit bands or `f_pha`/`f_amp` with `n_bands`

For fair comparison, always use **explicit band definitions**:

```python
# Define explicit frequency bands
pha_n_bands = 20  # Number of phase bands
amp_n_bands = 30  # Number of amplitude bands

# Create band edges
pha_edges = np.linspace(2, 30, pha_n_bands + 1)
amp_edges = np.linspace(30, 180, amp_n_bands + 1)

# Convert to band arrays
pha_bands_hz = np.c_[pha_edges[:-1], pha_edges[1:]]  # Shape: (20, 2)
amp_bands_hz = np.c_[amp_edges[:-1], amp_edges[1:]]  # Shape: (30, 2)
```

#### 2. PAC Method Selection (idpac for MI)

Both libraries support multiple PAC methods. For comparison:

- **gPAC**: Set `method='mi'` for Modulation Index
- **TensorPAC**: Use `idpac=(2, 0, 0)` for the equivalent MI method

```python
# gPAC
pac_calc = PAC(
    pha_bands_hz=pha_bands_hz,
    amp_bands_hz=amp_bands_hz,
    fs=fs,
    method='mi'
)

# TensorPAC
pac_obj = Pac(
    f_pha=pha_bands_hz,
    f_amp=amp_bands_hz,
    idpac=(2, 0, 0),  # MI method
    dcomplex='hilbert'
)
```

#### 3. Input Shape Requirements

The libraries expect different input shapes:

- **gPAC**: `(batch, n_channels, n_timepoints)`
- **TensorPAC**: `(n_channels, n_timepoints)` or `(n_epochs, n_channels, n_timepoints)`

```python
# Example: Prepare data for both libraries
n_channels = 64
n_timepoints = 10000
n_epochs = 10

# For gPAC (requires batch dimension)
data_gpac = data.reshape(n_epochs, n_channels, n_timepoints)

# For TensorPAC (can work with 2D or 3D)
data_tensorpac = data  # Already in correct shape
```

#### 4. Output Shape Differences

**CRITICAL**: The libraries return PAC values in transposed formats!

- **gPAC**: Returns shape `(batch, n_channels, n_phase_bands, n_amplitude_bands)`
  - Convention: `(phase, amplitude)`
- **TensorPAC**: Returns shape `(n_amplitude_bands, n_phase_bands, n_epochs)`
  - Convention: `(amplitude, phase)`

**To compare results, you must transpose TensorPAC output**:

```python
# Compute PAC
gpac_pac = pac_calc(data_gpac)  # Shape: (10, 64, 20, 30)
tensorpac_pac = pac_obj.fit(data_tensorpac)  # Shape: (30, 20, 10)

# Extract single epoch/channel for comparison
gpac_result = gpac_pac[0, 0]  # Shape: (20, 30) - (phase, amp)
tensorpac_result = tensorpac_pac[:, :, 0]  # Shape: (30, 20) - (amp, phase)

# IMPORTANT: Transpose TensorPAC output to match gPAC convention
tensorpac_result_transposed = tensorpac_result.T  # Now (20, 30) - (phase, amp)

# Now both are comparable
correlation = np.corrcoef(gpac_result.flatten(), 
                         tensorpac_result_transposed.flatten())[0, 1]
```

#### 5. Initialization and Random Seeds

For reproducible comparisons:

```python
# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

#### 6. Permutation Testing and Z-score Calculation

Both libraries support surrogate/permutation testing with z-score normalization:

- **gPAC**: Use `n_perm` parameter - automatically computes z-scores using time-shifted surrogates
- **TensorPAC**: Use appropriate `idpac` tuple for z-scores

**Important**: For equivalent z-score calculations between gPAC and TensorPAC:

```python
# gPAC with permutations and z-scores
pac_calc = PAC(..., n_perm=200)
results = pac_calc(data)
pac_values = results['pac']
z_scores = results['pac_z']  # Automatically computed

# TensorPAC equivalent z-score calculation
# Use idpac=(2, 2, 1) for MI with amplitude time-shift surrogates and z-score normalization
pac_obj = Pac(
    f_pha=pha_bands_hz,
    f_amp=amp_bands_hz,
    idpac=(2, 2, 1),  # MI method, swap amplitude time blocks, z-score normalization
)
z_scores_tensorpac = pac_obj.filterfit(fs, signal, n_perm=200)
```

**Note**: gPAC uses time-shifting of the amplitude signal for surrogate generation (similar to TensorPAC's method 2: "Swap amplitude time blocks"), so use `idpac=(2, 2, 1)` for equivalent z-score calculations, NOT `(2, 1, 1)`.

### Complete Working Example

```python
import numpy as np
import torch
from gpac import PAC
from tensorpac import Pac

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Parameters
fs = 1000
duration = 10
n_channels = 64
n_epochs = 10

# Generate data
time = np.arange(0, duration, 1/fs)
data = np.random.randn(n_epochs, n_channels, len(time))

# Define frequency bands explicitly
pha_edges = np.linspace(2, 30, 21)  # 20 bands
amp_edges = np.linspace(30, 180, 31)  # 30 bands
pha_bands_hz = np.c_[pha_edges[:-1], pha_edges[1:]]
amp_bands_hz = np.c_[amp_edges[:-1], amp_edges[1:]]

# Initialize both calculators
pac_gpac = PAC(
    pha_bands_hz=pha_bands_hz,
    amp_bands_hz=amp_bands_hz,
    fs=fs,
    method='mi'
)

pac_tensorpac = Pac(
    f_pha=pha_bands_hz,
    f_amp=amp_bands_hz,
    idpac=(2, 0, 0),
    dcomplex='hilbert'
)

# Compute PAC
gpac_result = pac_gpac(data)  # (10, 64, 20, 30)
tensorpac_result = pac_tensorpac.fit(data[0])  # (30, 20)

# Compare single channel results
gpac_single = gpac_result[0, 0]  # (20, 30)
tensorpac_single = tensorpac_result.T  # Transpose! Now (20, 30)

# Calculate correlation
correlation = np.corrcoef(gpac_single.flatten(), 
                         tensorpac_single.flatten())[0, 1]
print(f"Correlation: {correlation:.4f}")
```

### Common Pitfalls to Avoid

1. **Not transposing TensorPAC output** - This will result in negative or low correlations
2. **Using different frequency band definitions** - Always use explicit bands
3. **Forgetting the batch dimension for gPAC** - gPAC requires 3D input
4. **Using different PAC methods** - Ensure `method='mi'` for gPAC and `idpac=(2,0,0)` for TensorPAC
5. **Comparing different channels or epochs** - Be explicit about which slice you're comparing

## Scripts

### 1. `compare_comodulograms.py`
Main comparison script that:
- Generates synthetic PAC signals
- Computes PAC using both gPAC and Tensorpac
- Creates comodulogram visualizations
- Calculates correlation and RMSE between methods
- Optionally computes z-scores and p-values with permutation testing

### 2. `run_all_comparisons.py`
Batch comparison script that:
- Tests multiple parameter configurations
- Runs comparisons for different noise levels, frequencies, and PAC strengths
- Generates a comprehensive summary report
- Saves results in CSV format for further analysis

### 3. `generate_16_comparison_pairs.py`
Comprehensive comparison script that:
- Generates 16 diverse PAC signal pairs covering full frequency spectrum
- Includes classic theta-gamma, alpha-gamma, and beta-high gamma couplings
- Creates 2x2 subplot figures showing PAC values and z-scores
- Marks ground truth PAC locations on all plots
- Calculates and displays correlations for both PAC values and z-scores
- Tests edge cases with weak coupling and high noise

## Usage

### Quick comparison (without permutation testing):
```bash
python compare_comodulograms.py
```

### Full comparison with statistics:
```bash
python compare_comodulograms.py --n_perm 100
```

### Run all test configurations:
```bash
python run_all_comparisons.py
```

### Quick test run:
```bash
python run_all_comparisons.py --quick
```

### Generate 16 comparison pairs:
```bash
python generate_16_comparison_pairs.py
```

### Generate 16 pairs without permutation testing (faster):
```bash
python generate_16_comparison_pairs.py --n_perm 0
```

## Parameters

- `--batch_size`: Number of samples (default: 2)
- `--n_channels`: Number of channels (default: 4)
- `--duration`: Signal duration in seconds (default: 10)
- `--fs`: Sampling frequency (default: 512)
- `--phase_freq`: Phase frequency for synthetic PAC (default: 10)
- `--amp_freq`: Amplitude frequency for synthetic PAC (default: 80)
- `--pac_strength`: PAC coupling strength 0-1 (default: 0.5)
- `--noise_level`: Noise level (default: 0.5)
- `--n_perm`: Number of permutations for surrogate testing (default: 100)

## Output Files

### From `compare_comodulograms.py`:
- `comodulogram_comparison_gpac_tensorpac.gif`: Visual comparison of comodulograms
- `comparison_results.yaml`: Detailed results including correlations and statistics
- `comparison_statistics.csv`: Summary statistics in CSV format

### From `run_all_comparisons.py`:
- `comparison_summary_report.gif`: Summary visualization from batch comparisons
- `all_comparisons_results.csv`: All test results in tabular format

### From `generate_16_comparison_pairs.py`:
- `comparison_pair_01.gif` to `comparison_pair_16.gif`: Individual comparison figures with ground truth markers
- `correlation_summary.csv`: Correlation values for all 16 pairs
- `correlation_summary_visualization.gif`: Summary plot of correlation distribution

## Requirements

- gPAC (this package)
- tensorpac
- torch
- numpy
- matplotlib
- mngs
- pandas
- scipy