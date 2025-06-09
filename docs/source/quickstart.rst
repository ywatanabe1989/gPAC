Quick Start
===========

This guide will help you get started with gPAC in just a few minutes.

Basic Usage
-----------

The simplest way to compute PAC:

.. code-block:: python

   import gpac
   import torch
   
   # Generate or load your signal
   signal = torch.randn(1, 1, 2048)  # (batch, channel, time)
   fs = 512  # Sampling frequency in Hz
   
   # Initialize PAC module
   pac_model = gpac.PAC(
       seq_len=2048,
       fs=fs,
       pha_start_hz=4.0,    # Phase: 4-8 Hz (theta)
       pha_end_hz=8.0,
       pha_n_bands=1,       # Single theta band
       amp_start_hz=30.0,   # Amplitude: 30-100 Hz (gamma)
       amp_end_hz=100.0,
       amp_n_bands=10,      # 10 gamma sub-bands
   )
   
   # Compute PAC
   pac_values = pac_model(signal)
   print(f"PAC matrix shape: {pac_values.shape}")
   # Output: PAC matrix shape: torch.Size([1, 1, 1, 10])

TensorPAC-Compatible Mode
-------------------------

For direct comparison with TensorPAC:

.. code-block:: python

   # Using 'hres' and 'mres' equivalent settings
   pac_values = gpac.calculate_pac(
       signal,
       fs=fs,
       pha_n_bands=50,      # 'hres' - high resolution phase
       amp_n_bands=30,      # 'mres' - medium resolution amplitude
       filtfilt_mode=True,  # Sequential filtering (recommended)
       edge_mode='reflect'  # Edge padding strategy
   )

Using GPU Acceleration
----------------------

.. code-block:: python

   # Check GPU availability
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # Move model and data to GPU
   pac_model = pac_model.to(device)
   signal = signal.to(device)
   
   # Compute PAC on GPU
   with torch.no_grad():  # Disable gradients for faster computation
       pac_values = pac_model(signal)

Batch Processing
----------------

Process multiple signals simultaneously:

.. code-block:: python

   # Batch of 32 signals, 4 channels each
   batch_signals = torch.randn(32, 4, 2048)
   
   # PAC computation is fully parallelized
   pac_values = pac_model(batch_signals)
   print(f"Batch PAC shape: {pac_values.shape}")
   # Output: Batch PAC shape: torch.Size([32, 4, 1, 10])

Statistical Testing
-------------------

Add permutation testing for statistical significance:

.. code-block:: python

   # Initialize with permutation testing
   pac_model = gpac.PAC(
       seq_len=2048,
       fs=fs,
       pha_start_hz=4.0,
       pha_end_hz=8.0,
       pha_n_bands=1,
       amp_start_hz=30.0,
       amp_end_hz=100.0,
       amp_n_bands=10,
       n_perm=200,         # 200 permutations
       return_dist=True    # Return surrogate distribution
   )
   
   # Compute PAC with z-scores
   pac_zscore, surrogate_dist = pac_model(signal)
   
   # Significant coupling where |z| > 2
   significant = torch.abs(pac_zscore) > 2
   print(f"Significant couplings: {significant.sum().item()}")

Modular Components
------------------

Use individual components for custom pipelines:

.. code-block:: python

   from gpac import BandPassFilter, Hilbert, ModulationIndex
   
   # 1. Bandpass filtering
   filter_module = BandPassFilter(
       seq_len=2048,
       fs=fs,
       pha_start_hz=4, pha_end_hz=8, pha_n_bands=1,
       amp_start_hz=30, amp_end_hz=100, amp_n_bands=10
   )
   filtered = filter_module(signal)
   
   # 2. Hilbert transform
   hilbert = Hilbert(seq_len=2048)
   phase, amplitude = hilbert(filtered)
   
   # 3. Modulation Index
   mi = ModulationIndex(n_bins=18)
   pac_values = mi(phase, amplitude)

Visualization Example
---------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Compute PAC
   pac_model = gpac.PAC(
       seq_len=2048, fs=512,
       pha_start_hz=2, pha_end_hz=20, pha_n_bands=10,
       amp_start_hz=30, amp_end_hz=150, amp_n_bands=20
   )
   pac_values = pac_model(signal).squeeze().cpu().numpy()
   
   # Plot PAC matrix
   plt.figure(figsize=(10, 6))
   plt.imshow(pac_values, aspect='auto', origin='lower',
              extent=[30, 150, 2, 20], cmap='hot')
   plt.xlabel('Amplitude Frequency (Hz)')
   plt.ylabel('Phase Frequency (Hz)')
   plt.title('Phase-Amplitude Coupling')
   plt.colorbar(label='PAC Strength')
   plt.show()

Next Steps
----------

* See :doc:`user_guide` for detailed usage instructions
* Check :doc:`examples` for real-world applications
* Read :doc:`api_reference` for complete API documentation
* Learn about :doc:`theory` for PAC background