<!-- ---
!-- Timestamp: 2025-05-26 06:26:35
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/docs/real_world_data_sources.md
!-- --- -->

# Real-World EEG Data Sources for PAC Analysis

## Publicly Available Datasets with Known PAC

### 1. **Sleep EEG Data**
- **Source**: Sleep-EDF Database (PhysioNet)
- **URL**: https://physionet.org/content/sleep-edfx/
- **PAC Content**: Sleep spindles (11-15 Hz) coupled with slow oscillations (0.5-2 Hz)
- **Format**: European Data Format (EDF)
- **License**: Open access

### 2. **Motor Imagery EEG**
- **Source**: BCI Competition IV Dataset 2a
- **URL**: https://www.bbci.de/competition/iv/
- **PAC Content**: Sensorimotor rhythms with cross-frequency coupling
- **Format**: MATLAB/GDF
- **License**: Research use

### 3. **Cognitive Task EEG**
- **Source**: EEGBCI Database (PhysioNet)
- **URL**: https://physionet.org/content/eegmmidb/
- **PAC Content**: Mu (8-12 Hz) and beta (13-30 Hz) rhythm coupling
- **Format**: European Data Format (EDF)
- **License**: Open Database License

### 4. **Resting State EEG**
- **Source**: Temple University EEG Corpus
- **URL**: https://www.isip.piconepress.com/projects/tuh_eeg/
- **PAC Content**: Alpha-gamma coupling in resting state
- **Format**: European Data Format (EDF)
- **License**: Research agreement required

## Implementation Example

```python
import mne
import gpac
import numpy as np

def load_physionet_sleep_data():
    \"\"\"Load sleep EEG data with known PAC characteristics.\"\"\"
    
    # Download sleep EDF data
    raw = mne.io.read_raw_edf('sleep_data.edf', preload=True)
    
    # Extract single channel (e.g., C3-A2)
    raw.pick_channels(['EEG Fpz-Cz'])
    
    # Resample to 256 Hz for consistency
    raw.resample(256)
    
    # Extract 30-second epochs (typical sleep scoring window)
    events = mne.make_fixed_length_events(raw, duration=30.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=30.0, baseline=None)
    
    # Convert to gPAC format: (batch, channels, segments, time)
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Reshape for gPAC
    signal = torch.from_numpy(data).float()
    signal = signal.unsqueeze(2)  # Add segments dimension
    
    return signal

def analyze_real_pac():
    \"\"\"Analyze PAC in real EEG data.\"\"\"
    
    # Load real data
    signal = load_physionet_sleep_data()
    
    # Analyze with gPAC
    pac_values, pha_freqs, amp_freqs = gpac.calculate_pac(
        signal=signal,
        fs=256.0,
        pha_start_hz=0.5,   # Slow oscillations
        pha_end_hz=4.0,
        amp_start_hz=11.0,  # Sleep spindles
        amp_end_hz=15.0,
        device="cuda"
    )
    
    return pac_values, pha_freqs, amp_freqs
```

## Expected PAC Patterns

### Sleep Data
- **Slow-wave sleep**: Coupling between slow oscillations (0.5-2 Hz) and sleep spindles (11-15 Hz)
- **REM sleep**: Theta-gamma coupling (4-8 Hz with 30-100 Hz)


### Motor Tasks
- **Movement preparation**: Mu rhythm (8-12 Hz) coupled with high gamma (60-100 Hz)
- **Movement execution**: Beta suppression with gamma bursts

### Cognitive Tasks
- **Working memory**: Theta-gamma coupling (4-8 Hz with 25-40 Hz)
- **Attention**: Alpha-gamma coupling (8-12 Hz with 40-60 Hz)

## Validation Benefits

1. **Biological Relevance**: Demonstrates gPAC works on real neural data
2. **Cross-validation**: Compare with published PAC findings
3. **Performance Testing**: Real data computational benchmarks
4. **Method Validation**: Verify against known PAC phenomena

## Next Steps

1. Add example scripts for loading common EEG formats
2. Create validation against published PAC studies
3. Implement automatic dataset downloading
4. Add real-world performance benchmarks

<!-- EOF -->