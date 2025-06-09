Installation
============

Requirements
------------

* Python 3.8 or higher
* PyTorch 1.9.0 or higher
* CUDA-capable GPU (recommended for optimal performance)

Install from PyPI
-----------------

The easiest way to install gPAC is via pip:

.. code-block:: bash

   pip install gpac

To install with all optional dependencies:

.. code-block:: bash

   pip install gpac[all]

Install from Source
-------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/ywatanabe1989/gPAC.git
   cd gPAC
   pip install -e .

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[dev,testing,all]"

Verify Installation
-------------------

To verify that gPAC is installed correctly:

.. code-block:: python

   import gpac
   import torch
   
   # Check if GPU is available
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Quick test
   signal = torch.randn(1, 1, 1024)
   pac = gpac.PAC(seq_len=1024, fs=256)
   result = pac(signal)
   print(f"PAC shape: {result.shape}")

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

* **torch** >= 1.9.0: PyTorch for GPU acceleration
* **numpy** >= 1.19.0: Numerical operations
* **scipy** >= 1.7.0: Signal processing utilities
* **torchaudio** >= 0.9.0: Audio processing utilities

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For visualization and examples:

* **matplotlib** >= 3.0: Plotting and visualization
* **mne** >= 0.23: EEG/MEG data handling
* **scikit-learn** >= 0.24: Machine learning utilities

For development:

* **black** >= 22.0: Code formatting
* **flake8** >= 4.0: Code linting
* **isort** >= 5.0: Import sorting
* **pytest** >= 6.0: Testing framework
* **sphinx** >= 4.0: Documentation generation

Troubleshooting
---------------

CUDA Issues
~~~~~~~~~~~

If you encounter CUDA-related errors:

1. Ensure your PyTorch installation matches your CUDA version:

   .. code-block:: bash

      python -c "import torch; print(torch.version.cuda)"

2. Reinstall PyTorch with the correct CUDA version from https://pytorch.org/

Memory Issues
~~~~~~~~~~~~~

For large datasets, you may need to:

1. Reduce batch size
2. Use mixed precision (fp16)
3. Process data in chunks

.. code-block:: python

   # Enable mixed precision
   pac = gpac.PAC(seq_len=1024, fs=256, fp16=True)