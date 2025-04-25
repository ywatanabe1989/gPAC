<!-- ---
!-- Timestamp: 2025-04-10 12:41:05
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/README.md
!-- --- -->

# gPAC: GPU-Accelerated Phase-Amplitude Coupling

`gPAC` is a Python package designed for efficient calculation of Phase-Amplitude Coupling (PAC) using PyTorch, enabling significant speedups through GPU acceleration especially for large datasets.

## Rationale for GPU Acceleration

The Modulation Index (MI) method for PAC calculation, implemented in `gPAC`, involves several computationally intensive steps:

1.  **Bandpass Filtering:** Extracting signals within specific low-frequency (phase) and high-frequency (amplitude) bands.
2.  **Hilbert Transform:** Calculating the analytic signal to obtain instantaneous phase and amplitude.
3.  **Modulation Index Calculation:** Quantifying the relationship between phase and amplitude, often involving binning and entropy measures (like Kullback-Leibler divergence).
4.  **Permutation Testing (Optional):** Repeating the calculation on surrogate data (e.g., shuffled time series) for statistical validation (Z-scoring).

Howerver, each of these steps possesses characteristics suitable for parallelization, especially on GPUs:

-   Filtering for different frequency bands can be performed independently.
-   The Hilbert transform (via FFT) can be applied independently to multiple channels or segments.
-   The MI calculation, involving binning and aggregation, can often be vectorized or parallelized across frequency band pairs.
-   Surrogate computations are inherently parallel, as each permutation is independent.

`gPAC` leverages PyTorch's CUDA capabilities to execute these steps in parallel on the GPU, significantly reducing computation time compared to sequential CPU processing.

## Differentiable Filters and Trainable Band Ranges

`gPAC` optionally supports differentiable bandpass filters, enabling end-to-end training when incorporated into larger deep learning models. This differentiability is achieved by leveraging sinc-based convolutional filters, as demonstrated by M. Ravanelli and Y. Bengio [1].

While phase calculation in the Hilbert transform involve analitically non-differentiable points (e.g., phase wraps at +/- pi), they are still "computationally differentiable" by defining a subgradient (or practically setting the gradient to zero or one) at these points. This is in a similar manner to how the ReLU activation function at x=0 is managed in deep learning frameworks.

This allows a model to potentially learn optimal frequency bands for PAC calculation relevant to a specific task directly from data, which might offer insights for underlying biological mechanisms.

## Installation

```bash
git clone https://github.com/[your_username]/gPAC.git # Replace [your_username]
cd gPAC
python -m venv .env
source .env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Basic Usage

```python
import torch
import gpac
import numpy as np

# --- Parameters ---
fs = 1024  # Sampling frequency
seq_len = fs * 5  # 5 seconds of data
batch_size = 4
n_chs = 16
n_segments = 1 # Example: treating each channel trace as one segment

# --- Generate Example Signal ---
# Replace with your actual data loading
# Shape: (batch_size, n_channels, n_segments, sequence_length)
signal_np = np.random.randn(batch_size, n_chs, n_segments, seq_len).astype(np.float32)
signal_gpu = torch.from_numpy(signal_np).cuda() # Move to GPU

# --- Calculate PAC ---
try:
    pac_values, freqs_pha, freqs_amp = gpac.calculate_pac(
        signal=signal_gpu,
        fs=fs,
        pha_n_bands=50,       # Number of phase frequency bands
        amp_n_bands=30,       # Number of amplitude frequency bands
        pha_start_hz=2.0,     # Min frequency for phase
        pha_end_hz=20.0,      # Max frequency for phase
        amp_start_hz=60.0,    # Min frequency for amplitude
        amp_end_hz=160.0,     # Max frequency for amplitude
        device='cuda',        # Specify computation device
        fp16=True,            # Use mixed-precision (optional)
        n_perm=200,           # Calculate Z-score with 200 permutations (optional)
        # trainable=False,    # Use static filters (default)
        # chunk_size=16       # Process in chunks (optional, if memory is limited)
    )

    print("PAC calculation successful.")
    # Output shape depends on n_perm and amp_prob flags
    print(f"Output PAC Tensor Shape: {pac_values.shape}")
    print(f"Phase Frequencies (Num): {len(freqs_pha)}")
    print(f"Amplitude Frequencies (Num): {len(freqs_amp)}")

    # pac_values tensor will be on the specified device ('cuda' in this case)

except Exception as e:
    print(f"An error occurred during PAC calculation: {e}")
```

## References

[1] M. Ravanelli and Y. Bengio, "Speaker Recognition from Raw Waveform with SincNet," *2018 IEEE Spoken Language Technology Workshop (SLT)*, Athens, Greece, 2018, pp. 1021-1028, doi: 10.1109/SLT.2018.8639585.

## Contact
Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)

<!-- EOF -->