.. gPAC documentation master file

Welcome to gPAC's documentation!
================================

**gPAC** (GPU-Accelerated Phase-Amplitude Coupling) is a PyTorch-based package for efficient computation of Phase-Amplitude Coupling (PAC) using Modulation Index (MI) with GPU acceleration.

.. note::
   gPAC achieves **158-172x speedup** over TensorPAC while maintaining high accuracy (r=0.898)

Key Features
------------

* **GPU Acceleration**: 28-63x faster than TensorPAC through PyTorch/CUDA optimization
* **Sequential Filtfilt**: Novel implementation that's 1.2x faster than averaging while matching scipy.signal.filtfilt
* **TensorPAC Compatibility**: Supports 'hres'/'mres' frequency specifications for direct comparison
* **Modular Design**: Use components independently (filtering, Hilbert, MI calculation)
* **Statistical Analysis**: Built-in permutation testing and surrogate distributions
* **Differentiable**: Optional gradient flow for deep learning integration

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   theory
   performance
   contributing
   changelog

Quick Example
-------------

.. code-block:: python

   import gpac
   import torch

   # Generate sample data
   signal = torch.randn(1, 1, 2048)  # (batch, channel, time)
   fs = 512  # Sampling frequency

   # Calculate PAC
   pac_values = gpac.calculate_pac(
       signal, 
       fs=fs,
       pha_n_bands=50,  # 'hres' equivalent
       amp_n_bands=30,  # 'mres' equivalent
       filtfilt_mode=True,
       edge_mode='reflect'
   )

Performance
-----------

.. list-table:: Performance Comparison
   :widths: 40 20 20 20
   :header-rows: 1

   * - Method
     - Time (ms)
     - Speedup
     - Correlation
   * - TensorPAC (wavelet)
     - 76
     - 1x
     - \-
   * - TensorPAC (hilbert)
     - 169
     - 1x
     - \-
   * - **gPAC (hilbert+filtfilt)**
     - **3**
     - **28-63x**
     - **0.898**

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`