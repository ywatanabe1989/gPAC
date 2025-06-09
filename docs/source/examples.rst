Examples
========

This section provides practical examples of using gPAC for various neuroscience applications.

Basic Examples
--------------

Simple PAC Analysis
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/gpac/simple_pac_demo.py
   :language: python
   :caption: Basic PAC computation example
   :lines: 35-65

Cognitive Workload Classification
---------------------------------

This example demonstrates using PAC features for classifying cognitive workload states.

Key Features:
* Multiple classifier comparison (including baselines)
* Proper cross-validation without data leakage
* Statistical testing between conditions

.. code-block:: python

   # Extract PAC features for classification
   pac_model = gpac.PAC(
       seq_len=n_times,
       fs=sfreq,
       pha_start_hz=4.0,    # Theta band
       pha_end_hz=8.0,
       pha_n_bands=2,
       amp_start_hz=30.0,   # Gamma band
       amp_end_hz=45.0,
       amp_n_bands=3,
   )
   
   # Compute PAC for each epoch
   pac_features = []
   for epoch in epochs:
       pac = pac_model(epoch)
       pac_features.append(pac.max())  # Use max PAC as feature
   
   # Classification with proper pipeline
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('svm', SVC(kernel='rbf'))
   ])
   
   scores = cross_val_score(pipeline, pac_features, labels, cv=5)

See full example: ``examples/cognitive_workload/``

Epilepsy State Detection
------------------------

Multi-class classification of epileptic states using PAC features.

.. code-block:: python

   # Define epilepsy-relevant frequency bands
   pac_epilepsy = gpac.PAC(
       seq_len=seq_len,
       fs=fs,
       pha_start_hz=0.5,    # Delta/theta for epilepsy
       pha_end_hz=8.0,
       pha_n_bands=4,
       amp_start_hz=30.0,   # High-frequency oscillations
       amp_end_hz=200.0,
       amp_n_bands=20,
   )
   
   # Extract features from multiple frequency combinations
   features = []
   for segment in eeg_segments:
       pac_matrix = pac_epilepsy(segment)
       
       # Feature vector includes:
       # - Mean PAC per frequency band
       # - Max PAC location
       # - PAC variance
       feature_vec = [
           pac_matrix.mean(dim=(-2, -1)),
           pac_matrix.max(dim=(-2, -1))[0],
           pac_matrix.var(dim=(-2, -1))
       ]
       features.append(torch.cat(feature_vec))

See full example: ``examples/epilepsy/``

Motor Cortex Analysis
---------------------

Analyzing PAC during hand grasping movements.

.. code-block:: python

   # Motor cortex specific bands
   pac_motor = gpac.PAC(
       seq_len=seq_len,
       fs=fs,
       pha_start_hz=13.0,   # Beta phase
       pha_end_hz=30.0,
       pha_n_bands=3,
       amp_start_hz=60.0,   # High gamma amplitude
       amp_end_hz=150.0,
       amp_n_bands=15,
   )
   
   # Time-resolved PAC
   window_size = 512  # 2 seconds at 256 Hz
   step_size = 128    # 0.5 second steps
   
   pac_over_time = []
   for t in range(0, signal.shape[-1] - window_size, step_size):
       window = signal[..., t:t+window_size]
       pac_t = pac_motor(window)
       pac_over_time.append(pac_t)
   
   # Stack time windows
   pac_dynamics = torch.stack(pac_over_time, dim=0)

Real-World EEG Demo
-------------------

Using public EEG datasets with gPAC.

.. code-block:: python

   import mne
   from mne.datasets import sample
   
   # Load sample data
   data_path = sample.data_path()
   raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
   
   # Preprocess
   raw.pick_types(eeg=True)
   raw.filter(0.5, 200)
   raw.resample(256)
   
   # Convert to torch tensor
   data = torch.tensor(raw.get_data(), dtype=torch.float32)
   
   # Compute PAC
   pac_model = gpac.PAC(
       seq_len=data.shape[-1],
       fs=raw.info['sfreq'],
       pha_start_hz=4, pha_end_hz=8,    # Theta
       amp_start_hz=30, amp_end_hz=100, # Gamma
       pha_n_bands=2,
       amp_n_bands=10
   )
   
   pac_results = pac_model(data.unsqueeze(0))

Batch Processing Large Datasets
-------------------------------

Efficient processing of large recordings.

.. code-block:: python

   def process_continuous_recording(file_path, chunk_duration=30):
       """Process long recordings in chunks."""
       
       # Initialize PAC model once
       pac_model = gpac.PAC(
           seq_len=int(chunk_duration * fs),
           fs=fs,
           pha_start_hz=2, pha_end_hz=20,
           amp_start_hz=30, amp_end_hz=150,
           pha_n_bands=10, amp_n_bands=20
       ).cuda()
       
       results = []
       
       # Process in chunks
       with h5py.File(file_path, 'r') as f:
           data = f['eeg_data']
           n_chunks = data.shape[-1] // (chunk_duration * fs)
           
           for i in tqdm(range(n_chunks)):
               # Load chunk
               start = i * chunk_duration * fs
               end = (i + 1) * chunk_duration * fs
               chunk = torch.tensor(data[..., start:end]).cuda()
               
               # Compute PAC
               with torch.no_grad():
                   pac_chunk = pac_model(chunk.unsqueeze(0))
                   results.append(pac_chunk.cpu())
               
               # Clear cache periodically
               if i % 10 == 0:
                   torch.cuda.empty_cache()
       
       return torch.cat(results, dim=0)

Custom Visualizations
---------------------

Creating publication-quality PAC figures.

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   def plot_pac_comodulogram(pac_values, phase_freqs, amp_freqs, 
                             title="Phase-Amplitude Coupling"):
       """Create beautiful PAC visualization."""
       
       fig, ax = plt.subplots(figsize=(10, 8))
       
       # Use custom colormap
       cmap = sns.color_palette("rocket", as_cmap=True)
       
       # Plot with contours
       im = ax.imshow(pac_values, aspect='auto', origin='lower',
                      extent=[amp_freqs[0], amp_freqs[-1], 
                              phase_freqs[0], phase_freqs[-1]],
                      cmap=cmap, interpolation='bilinear')
       
       # Add contour lines
       contours = ax.contour(amp_freqs, phase_freqs, pac_values, 
                             levels=5, colors='white', alpha=0.4, 
                             linewidths=0.5)
       
       # Labels and formatting
       ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=12)
       ax.set_ylabel('Phase Frequency (Hz)', fontsize=12)
       ax.set_title(title, fontsize=14, fontweight='bold')
       
       # Colorbar
       cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       cbar.set_label('PAC Strength', fontsize=12)
       
       # Add grid
       ax.grid(True, alpha=0.3, linestyle='--')
       
       return fig, ax

Integration with Deep Learning
------------------------------

Using PAC features in neural networks.

.. code-block:: python

   import torch.nn as nn
   
   class PACNet(nn.Module):
       """Neural network using PAC features."""
       
       def __init__(self, seq_len=1024, fs=256, n_classes=2):
           super().__init__()
           
           # PAC feature extractor
           self.pac = gpac.PAC(
               seq_len=seq_len,
               fs=fs,
               pha_start_hz=4, pha_end_hz=8,
               pha_n_bands=2,
               amp_start_hz=30, amp_end_hz=100,
               amp_n_bands=10,
               trainable=True  # Learnable filters
           )
           
           # Classifier
           pac_features = 2 * 10  # pha_bands * amp_bands
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(pac_features, 64),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(64, n_classes)
           )
       
       def forward(self, x):
           # Extract PAC features
           pac_features = self.pac(x)
           
           # Classify
           return self.classifier(pac_features)
   
   # Training example
   model = PACNet().cuda()
   optimizer = torch.optim.Adam(model.parameters())
   criterion = nn.CrossEntropyLoss()
   
   for epoch in range(100):
       for batch, labels in dataloader:
           optimizer.zero_grad()
           outputs = model(batch.cuda())
           loss = criterion(outputs, labels.cuda())
           loss.backward()
           optimizer.step()

Running Examples
----------------

All examples can be found in the ``examples/`` directory:

.. code-block:: bash

   # Basic demo
   python examples/simple_pac_demo.py
   
   # Cognitive workload analysis
   python examples/cognitive_workload/cognitive_workload_demo.py
   
   # Epilepsy classification
   python examples/epilepsy/epilepsy_classification_demo.py
   
   # Motor cortex analysis
   python examples/handgrasping/hand_grasping_demo.py