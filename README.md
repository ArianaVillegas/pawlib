# PAWlib 

**Seismic amplitude window prediction using deep learning**

PAWlib uses the PAW (Predicting Amplitude Windows) deep learning model to predict amplitude windows in seismic waveforms. The library works with standard seismic data formats like SAC files and provides both container-based and Python library interfaces.

## What You Can Do

### 1. **🔍 Prediction** (Most Common Use)
   - **Container approach**: Analyze SAC files with no Python setup required
   - **Python library**: Integrate predictions into your existing workflows

### 2. **🎓 Advanced Training & Testing**
   - **Data preparation**: Create training datasets from your seismic data
   - **Container training**: Train models without managing Python environments
   - **Python training**: Full control over model development and testing

---

## 📋 Table of Contents

- [1. Prediction](#1-prediction)
  - [Container (Docker/Podman)](#container-dockerpodman)
  - [Python Library](#python-library)
- [2. Advanced Training & Testing](#2-advanced-training--testing)
  - [Data Preparation](#data-preparation)
  - [Python Training](#python-training)
- [Troubleshooting](#troubleshooting)

# 1. Prediction

## Container (Docker/Podman)

**Use this approach if:** You just want to analyze SAC files with no Python setup required.

### Setup (One-time)
```bash
# Get PAWlib
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib

# Build container
podman build -t pawlib:latest .
```

### Analyze Single SAC File
```bash
# Prepare directories
mkdir -p data outputs

# Get sample data (or use your own SAC file)
curl -o data/sample.sac https://examples.obspy.org/COP.BHZ.DK.2009.050

# Run prediction (onset_time in seconds)
podman run --rm \
  -v $(pwd)/data:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/sample.sac 10.5 /output/result.png

# View results
open outputs/result.png
```

### Batch Process Multiple SAC Files
```bash
# Process all SAC files in a directory
podman run --rm \
  -v $(pwd)/data:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_batch_predict.py /data /output/results.csv

# Results saved to CSV with timing data for each file
```

### GPU Acceleration (Optional)
```bash
# Enable GPU support for faster processing
podman run --rm --device nvidia.com/gpu=all \
  -v $(pwd)/data:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/your_file.sac 10.5 /output/result.png
```

## Python Library

**Use this approach if:** You want to integrate PAWlib into existing Python code or workflows.

### Setup with Conda Environment (Recommended)
```bash
# Create dedicated environment
conda create -n pawlib python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate pawlib

# Install scientific packages
conda install numpy scipy h5py pyyaml torchmetrics pandas matplotlib obspy -c conda-forge

# Install PAWlib
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git

# Verify installation
python -c "from pawlib import PAW; print('✅ PAWlib ready!')"
```

### Basic Prediction Example
```python
from pawlib import PAW, load_sac_waveform, preprocess_for_paw
import numpy as np

# Load and preprocess SAC file
waveform, meta = load_sac_waveform('earthquake.sac', onset_time=10.5)
waveform = preprocess_for_paw(waveform, meta['sampling_rate'])
waveform = np.pad(waveform, ((0,0), (20,20), (0,0)))  # Required padding

# Load model and predict
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
windows = model.predict_windows(waveform[:, 20:-20, :])

# Get results in seconds
start_time = windows[0, 0] / 40.0  
end_time = windows[0, 1] / 40.0
print(f"Predicted amplitude window: {start_time:.3f}s - {end_time:.3f}s")
```

### Batch Processing Example
```python
from pawlib import PAW, load_sac_waveform, preprocess_for_paw
import numpy as np
import pandas as pd

# Load model once
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')

# Process multiple files
files = ['event1.sac', 'event2.sac', 'event3.sac']
onset_times = [10.5, 15.2, 8.7]
results = []

for filename, onset_time in zip(files, onset_times):
    # Load and preprocess each file
    waveform, meta = load_sac_waveform(filename, onset_time=onset_time)
    waveform = preprocess_for_paw(waveform, meta['sampling_rate'])
    waveform = np.pad(waveform, ((0,0), (20,20), (0,0)))
    
    # Predict amplitude windows directly
    windows = model.predict_windows(waveform[:, 20:-20, :])
    
    # Store results
    results.append({
        'filename': filename,
        'start_time': windows[0, 0] / 40.0,
        'end_time': windows[0, 1] / 40.0,
        'duration': (windows[0, 1] - windows[0, 0]) / 40.0
    })

# Save to CSV
pd.DataFrame(results).to_csv('predictions.csv', index=False)
```

# 2. Advanced Training & Testing

## Data Preparation

**Prepare your seismic data for training custom PAW models.**

### HDF5 Dataset Format
PAWlib requires training data in HDF5 format with specific structure:

```python
import h5py
import numpy as np

# Create training dataset from your seismic data
with h5py.File('my_training_data.h5', 'w') as f:
    # Waveforms: (N, 200, 1) - N traces, 200 samples at 40Hz, 1 channel
    f.create_dataset('waveforms', data=your_waveforms)
    
    # Labels: (N, 2) - [start_time, end_time] in seconds for each trace
    f.create_dataset('labels', data=your_labels)
    
    # Optional metadata
    f.attrs['sampling_rate'] = 40.0
    f.attrs['window_duration'] = 5.0
    f.attrs['description'] = 'My earthquake dataset'
```

### Data Requirements
- **Waveforms**: Shape `(N, 200, 1)` at 40 Hz sampling rate
- **Labels**: Shape `(N, 2)` with `[start_time, end_time]` in seconds
- **Preprocessing**: Data must be filtered (1-15 Hz), normalized, and resampled to 40 Hz

### Paper Dataset
Download dataset from [here](https://huggingface.co/datasets/suroRitch/PAW/tree/main) and place it in the root directory of this project.

The dataset consists of 80,648 waveforms. The `dataset.h5` file contains two keys:
- `waveforms`: A numpy array with a shape of (80648, 200, 1) representing the waveforms.
- `labels`: A numpy array with a shape of (80648, 2) containing labels (start time, end time relative to a 5 second window) associated with the waveforms.

### Preprocessing Your Data
```python
from pawlib import preprocess_for_paw

# Example: preparing raw seismic traces
for i, raw_trace in enumerate(your_raw_traces):
    # Preprocess to PAW requirements (all parameters are optional with good defaults)
    processed = preprocess_for_paw(
        raw_trace, 
        sampling_rate=original_sampling_rate,
        target_freq=40.0,  # Default is 40.0 
        freqmin=1.0,       # Default is 1.0
        freqmax=15.0       # Default is 15.0
    )
    your_waveforms[i] = processed
```

## Python Training

**Use this approach if:** You want full control over model training and testing in Python.

### Environment Setup
```bash
# Create conda environment with all dependencies
conda create -n pawlib-train python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate pawlib-train
conda install numpy scipy h5py pyyaml torchmetrics pandas matplotlib obspy -c conda-forge

# Install PAWlib in development mode
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib
pip install -e ".[dev]"
```

### Training a Model
```python
from pawlib import PAW

# Initialize model for training
model = PAW(device='cuda')  # Use 'cpu' if no GPU available

# Train on your prepared HDF5 dataset
history = model.train(
    data='my_training_data.h5',     # Your prepared dataset
    epochs=100,                     # Adjust based on dataset size
    batch_size=64,                  # Adjust based on GPU memory
    loss='dice',                    # Good default for seismic detection
    lr=0.001,                       # Learning rate
    val_split=0.2,                  # Use 20% of data for validation
    save_best=True,                 # Save best model during training
    verbose=True                    # Show training progress
)

# Save your trained model
model.save('my_custom_paw_model.pt', metadata={
    'version': '1.0',
    'dataset': 'my_earthquake_data',
    'description': 'Custom PAW model for local seismicity'
})

print("✅ Training complete!")
```

### Testing a Model
```python
# Load your trained model
model = PAW.from_pretrained('my_custom_paw_model.pt')

# Test on held-out data
test_results = model.test(
    data='test_dataset.h5',
    batch_size=32
)

# View performance metrics
print(f"📊 Model Performance:")
print(f"   Window Accuracy: {test_results['window_accuracy']:.2%}")
print(f"   Dice Score: {test_results['dice_score']:.3f}")
print(f"   Average Error: {test_results['amplitude_rmse']:.3f}s")
```

### Advanced Training Options
```python
# Available loss functions
history = model.train(
    data='dataset.h5',
    epochs=100,
    batch_size=64,
    loss='bce+dice',        # Combined loss (options: 'dice', 'bce', 'amper', 'amperwdw', 'bce+dice')
    lr=0.001,
    val_split=0.2,
    checkpoint_dir='checkpoints',  # Save training checkpoints
    save_best=True,
    verbose=True
)

# Note: Advanced features like custom loss weights, learning rate scheduling, 
# gradient clipping, and multi-GPU training may require additional configuration
# or custom training loops not shown in this basic example.
```


## ❓ Troubleshooting

Having issues? Here are solutions to the most common problems:

### 🚫 Installation Problems

#### **"ObsPy not found" or SAC file errors**
```bash
# Solution: Install ObsPy for SAC file support
conda install obspy -c conda-forge
# OR
pip install obspy
```

#### **"PyTorch not found" or CUDA issues**
```bash
# Solution: Reinstall PyTorch with proper CUDA support
# Check your CUDA version first:
nvidia-smi

# Then install matching PyTorch version
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# Replace 11.8 with your CUDA version
```

#### **Import errors after installation**
```bash
# Solution: Reinstall in development mode
cd pawlib
pip install -e ".[dev]"

# Or try a fresh environment
conda create -n pawlib-fresh python=3.10
conda activate pawlib-fresh
# Then follow installation steps again
```

### 🐳 Container Problems

#### **"Permission denied" with Docker**
```bash
# Solution 1: Use Podman instead (recommended)
podman build -t pawlib:latest .

# Solution 2: Add yourself to docker group (requires logout/login)
sudo usermod -aG docker $USER

# Solution 3: Use sudo (not recommended)
sudo docker build -t pawlib:latest .
```

#### **Container build fails**
```bash
# Check disk space (needs ~2GB)
df -h

# Clean up old containers and images
podman system prune -a

# Try building with verbose output to see where it fails
podman build -t pawlib:latest . --progress=plain
```

#### **SELinux permission issues (RHEL/Fedora)**
```bash
# Make sure to use :Z flag in volume mounts
podman run --rm -v $(pwd)/data:/data:Z pawlib:latest [command]

# If still having issues, check SELinux context
ls -lZ data/
```

### 📊 Data and Analysis Problems

#### **"Waveform shape is incorrect" errors**
Your data needs to be exactly `(N, 200, 1)` shape at 40 Hz sampling rate.

```python
# Check your data shape
print(f"Current shape: {waveform.shape}")
print(f"Expected shape: (N, 200, 1)")

# Common fixes:
# 1. If your data is 1D, add batch and channel dimensions
waveform = waveform.reshape(1, -1, 1)

# 2. If wrong length, you may need to crop or pad
# For 5-second window at 40Hz, you need exactly 200 samples
target_length = 200
if waveform.shape[1] != target_length:
    print(f"Warning: adjusting length from {waveform.shape[1]} to {target_length}")
```

#### **"Model predictions look wrong"**
```python
# Check your data preprocessing
from pawlib import preprocess_for_paw

# Make sure you're preprocessing correctly
waveform = preprocess_for_paw(waveform, sampling_rate=original_rate)

# Verify the onset time is reasonable
# Should be within your 5-second window (0-5 seconds)
print(f"Onset time: {onset_time} seconds")
print(f"Window covers: 0-5 seconds")
```

#### **"No GPU detected" but you have one**
```python
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Running on CPU - install CUDA-compatible PyTorch for GPU acceleration")
```

### 📁 File and Path Issues

#### **"File not found" errors**
```bash
# Always use absolute paths or verify current directory
pwd
ls -la data/

# For containers, remember the mapping:
# Your local path -> Container path
# $(pwd)/data -> /data
# $(pwd)/outputs -> /output
```

#### **SAC file won't load**
```python
# Test your SAC file with ObsPy directly
from obspy import read
try:
    st = read('your_file.sac')
    print(f"✅ SAC file loaded: {len(st)} traces")
    print(f"   Duration: {st[0].stats.endtime - st[0].stats.starttime}s")
    print(f"   Sample rate: {st[0].stats.sampling_rate}Hz")
except Exception as e:
    print(f"❌ SAC file error: {e}")
```

### � Memory and Performance Issues

#### **"Out of memory" errors**
```python
# Reduce batch size
model = PAW()
# Instead of batch_size=64, try smaller:
history = model.train(data='dataset.h5', batch_size=16)

# For inference, process files one by one instead of batches
for sac_file in sac_files:
    result = process_single_file(sac_file)
```

#### **Slow processing**
```bash
# Enable GPU acceleration in containers
podman run --rm --device nvidia.com/gpu=all [rest of command]

# Check if you're using GPU
python -c "import torch; print(f'Using: {torch.cuda.get_device_name()}' if torch.cuda.is_available() else 'CPU only')"
```

### 🆘 Getting More Help

Still stuck? Try these debugging steps:

1. **Check versions:**
```python
import pawlib, torch, obspy
print(f"PAWlib: {pawlib.__version__}")  
print(f"PyTorch: {torch.__version__}")
print(f"ObsPy: {obspy.__version__}")
```

2. **Test with sample data:**
```bash
# Download known-good sample file
curl -o test.sac https://examples.obspy.org/COP.BHZ.DK.2009.050

# Test basic functionality
python -c "from pawlib import PAW; model = PAW(); print('✅ Basic import works')"
```

3. **Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now run your code - you'll see detailed output
```

---

## 📚 Complete Examples

| File | Description | Use Case |
|------|-------------|----------|
| `inference_demo.ipynb` | Interactive Jupyter notebook | Learning and exploration |
| `examples/docker_predict.py` | Single file prediction | Quick analysis |
| `examples/docker_batch_predict.py` | Batch processing | Production workflows |
| `examples/quick_start.py` | Basic Python usage | Integration testing |

## 📄 License
Proprietary - Internal Use Only. See LICENSE file for details.