User Guide
==========

This guide provides in-depth information about using gPAC effectively.

Understanding PAC
-----------------

Phase-Amplitude Coupling (PAC) quantifies the relationship between the phase of low-frequency oscillations and the amplitude of high-frequency oscillations. This coupling is thought to play a crucial role in neural communication and computation.

Key Concepts
~~~~~~~~~~~~

* **Phase signal**: Low-frequency oscillation (e.g., theta: 4-8 Hz)
* **Amplitude signal**: High-frequency oscillation (e.g., gamma: 30-100 Hz)
* **Modulation Index (MI)**: Measure of coupling strength
* **Comodulogram**: 2D visualization of PAC across frequency pairs

Choosing Parameters
-------------------

Frequency Ranges
~~~~~~~~~~~~~~~~

Common frequency band definitions in neuroscience:

.. code-block:: python

   # Standard frequency bands
   bands = {
       'delta': (0.5, 4),
       'theta': (4, 8),
       'alpha': (8, 13),
       'beta': (13, 30),
       'gamma': (30, 100),
       'high_gamma': (100, 200)
   }
   
   # Typical PAC combinations
   pac_model = gpac.PAC(
       seq_len=seq_len,
       fs=fs,
       pha_start_hz=bands['theta'][0],    # Theta phase
       pha_end_hz=bands['theta'][1],
       amp_start_hz=bands['gamma'][0],     # Gamma amplitude
       amp_end_hz=bands['gamma'][1],
   )

Number of Bands
~~~~~~~~~~~~~~~

More bands provide higher frequency resolution but increase computation:

.. code-block:: python

   # Low resolution (fast)
   pac_low_res = gpac.PAC(
       seq_len=seq_len, fs=fs,
       pha_n_bands=5,   # 5 phase bands
       amp_n_bands=10   # 10 amplitude bands
   )
   
   # High resolution (slower but more detailed)
   pac_high_res = gpac.PAC(
       seq_len=seq_len, fs=fs,
       pha_n_bands=50,  # TensorPAC 'hres' equivalent
       amp_n_bands=30   # TensorPAC 'mres' equivalent
   )

Advanced Features
-----------------

Edge Padding Strategies
~~~~~~~~~~~~~~~~~~~~~~~

Different padding modes affect edge artifacts:

.. code-block:: python

   # Available padding modes
   padding_modes = ['reflect', 'replicate', 'circular', 'zero', 'mirror']
   
   for mode in padding_modes:
       pac = gpac.PAC(
           seq_len=seq_len, fs=fs,
           padding_mode=mode  # Choose padding strategy
       )

Memory Optimization
~~~~~~~~~~~~~~~~~~~

For large datasets or limited GPU memory:

.. code-block:: python

   # Use half precision
   pac_fp16 = gpac.PAC(
       seq_len=seq_len, fs=fs,
       fp16=True  # Uses float16 instead of float32
   )
   
   # Process in chunks
   def process_long_signal(signal, chunk_size=4096, overlap=512):
       chunks = []
       for i in range(0, len(signal) - chunk_size, chunk_size - overlap):
           chunk = signal[..., i:i+chunk_size]
           pac_chunk = pac_model(chunk)
           chunks.append(pac_chunk)
       return torch.cat(chunks, dim=-1)

Differentiable Mode
~~~~~~~~~~~~~~~~~~~

For integration with deep learning:

.. code-block:: python

   # Enable gradient flow
   pac_trainable = gpac.PAC(
       seq_len=seq_len, fs=fs,
       trainable=True  # Makes filters trainable
   )
   
   # Use in neural network
   class PACNetwork(nn.Module):
       def __init__(self):
           super().__init__()
           self.pac = gpac.PAC(seq_len=1024, fs=256, trainable=True)
           self.classifier = nn.Linear(50, 2)  # Binary classification
       
       def forward(self, x):
           pac_features = self.pac(x)
           pac_flat = pac_features.flatten(start_dim=1)
           return self.classifier(pac_flat)

Performance Tips
----------------

GPU Utilization
~~~~~~~~~~~~~~~

Maximize GPU efficiency:

.. code-block:: python

   # Batch processing
   batch_size = 32  # Adjust based on GPU memory
   signals = torch.randn(batch_size, n_channels, seq_len).cuda()
   
   # Disable gradients if not training
   with torch.no_grad():
       pac_values = pac_model(signals)
   
   # Use torch.cuda.amp for mixed precision
   from torch.cuda.amp import autocast
   
   with autocast():
       pac_values = pac_model(signals)

Profiling
~~~~~~~~~

Monitor performance:

.. code-block:: python

   from gpac._Profiler import Profiler
   
   profiler = Profiler()
   
   with profiler:
       pac_values = pac_model(signal)
   
   print(profiler.get_summary())

Common Use Cases
----------------

EEG/MEG Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Typical EEG parameters
   pac_eeg = gpac.PAC(
       seq_len=2048,      # ~8 seconds at 256 Hz
       fs=256,            # Common EEG sampling rate
       pha_start_hz=4,    # Theta
       pha_end_hz=8,
       pha_n_bands=2,
       amp_start_hz=30,   # Gamma
       amp_end_hz=80,
       amp_n_bands=10,
       n_perm=200         # Statistical testing
   )

Multi-Channel Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process all EEG channels simultaneously
   n_channels = 64
   signals = torch.randn(1, n_channels, seq_len)
   
   # PAC for each channel
   pac_all_channels = pac_model(signals)
   
   # Average across channels
   pac_mean = pac_all_channels.mean(dim=1, keepdim=True)
   
   # Channel-specific analysis
   frontal_channels = [0, 1, 2, 3]  # Fp1, Fp2, F3, F4
   pac_frontal = pac_all_channels[:, frontal_channels]

Real-Time Processing
~~~~~~~~~~~~~~~~~~~~

For online applications:

.. code-block:: python

   class RealTimePAC:
       def __init__(self, window_size=1024, fs=256):
           self.pac = gpac.PAC(seq_len=window_size, fs=fs)
           self.buffer = torch.zeros(1, 1, window_size)
           
       def update(self, new_samples):
           # Shift buffer
           self.buffer = torch.roll(self.buffer, -len(new_samples), dims=-1)
           self.buffer[..., -len(new_samples):] = new_samples
           
           # Compute PAC
           with torch.no_grad():
               pac = self.pac(self.buffer)
           return pac

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Out of Memory**: Reduce batch size or use fp16
2. **Slow Performance**: Ensure CUDA is available and being used
3. **NaN Values**: Check for DC offset in signals, use detrending
4. **Poor Results**: Verify frequency ranges match your data

Best Practices
~~~~~~~~~~~~~~

1. Always normalize/standardize input signals
2. Use sufficient signal length (>= 1 second recommended)
3. Choose frequency ranges based on your specific application
4. Validate results with permutation testing
5. Compare with baseline/control conditions