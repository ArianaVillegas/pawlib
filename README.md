# PAWlib

A clean, production-ready PyTorch library for seismic phase arrival detection using the PAW model.

**Features:**
- 🚀 High-level API for training and inference
- 📁 SAC file loading with ObsPy integration
- 🔧 Preprocessing utilities (filter, resample, normalize)
- 📊 Comprehensive metrics and loss functions
- 💾 Hugging Face Hub integration for pretrained models

**Table of Contents:**
- [Quick Start (5 minutes)](#-quick-start-5-minutes)
- [Docker/Podman Usage](#-dockerpodman-usage)
- [Jupyter Notebook Examples](#-jupyter-notebook-examples)
- [Installation](#-installation)
- [Python Usage Examples](#-python-usage-examples)
- [Advanced Training & Testing](#-advanced-training--testing)
- [API Reference](#-api-reference)
- [Common Issues](#-common-issues)

## 🎯 Quick Start (5 minutes)

### Option 1: Using Docker/Podman (No installation needed)

```bash
# 1. Clone the repository
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib

# 2. Download sample seismic data
mkdir -p data/sample_sac
curl -o data/sample_sac/COP_BHZ_DK.sac https://examples.obspy.org/COP.BHZ.DK.2009.050
# Or if you have obspy: python examples/download_sample_sac.py

# 3. Build container image
podman build -t pawlib:latest .

# 4. Run prediction with visualization
mkdir -p outputs
podman run --rm \
  -v $(pwd)/data/sample_sac:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/COP_BHZ_DK.sac 200.0 /output/result.png

# 5. View the result
open outputs/result.png  # or xdg-open on Linux
```

### Option 2: Local Installation

```bash
# Install with dev dependencies
pip install "pawlib[dev] @ git+ssh://git@github.com/ArianaVillegas/pawlib.git"

# Run inference
python examples/quick_start.py
```

## 📦 Installation

### From Private GitHub (Recommended)

```bash
# Install latest version
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git

# Or specific version
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git@v1.0.0

# With optional dependencies for development
pip install "pawlib[dev] @ git+ssh://git@github.com/ArianaVillegas/pawlib.git"
```

### From Source (Development)

```bash
# Clone and install
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib
pip install -e ".[dev]"
```

### Requirements

- **Python:** 3.9+
- **PyTorch:** 2.0.0+ (with CUDA for GPU)
- **Core dependencies:** torch, numpy, h5py, torchmetrics, pyyaml (auto-installed)
- **Optional:** pandas, matplotlib, scipy (install with `[dev]`)

### Verify Installation

```python
from pawlib import PAW
model = PAW()
print(f"✅ pawlib installed! Model on {model.device}")
```

## 🐳 Docker/Podman Usage

### Build Container Image
```bash
# Clone repository and build
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib
podman build -t pawlib:latest .
```

### Basic Prediction with Visualization
```bash
# Download sample data
mkdir -p data/sample_sac outputs
curl -o data/sample_sac/COP_BHZ_DK.sac https://examples.obspy.org/COP.BHZ.DK.2009.050

# Run prediction with automatic half-cycle cropping
podman run --rm \
  -v $(pwd)/data/sample_sac:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/COP_BHZ_DK.sac 200.0 /output/result.png

# View results
open outputs/result.png  # or xdg-open on Linux
```

### GPU-Accelerated Processing
```bash
# With GPU support for faster inference
podman run --rm --device nvidia.com/gpu=all \
  -v /path/to/your/sac/files:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/your_file.sac 10.5 /output/result.png
```

### Batch Processing
```bash
# Process all SAC files in a directory
podman run --rm \
  -v /path/to/sac/files:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_batch_predict.py /data /output/results.csv
```

### Docker Alternative
```bash
# Using Docker instead of Podman
docker run --rm --gpus all \
  -v /path/to/your/sac/files:/data \
  -v $(pwd)/outputs:/output \
  pawlib:latest python examples/docker_predict.py /data/signal.sac 10.5 /output/result.png
```

## 📓 Jupyter Notebook Examples

### Interactive Demo Notebook
The `inference_demo.ipynb` provides a complete interactive workflow:

```bash
# Start Jupyter in the container
podman run --rm -p 8888:8888 \
  -v $(pwd):/workspace:Z \
  pawlib:latest jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

# Or run locally after installation
jupyter notebook inference_demo.ipynb
```

### Getting Both Predictions and Windows:
```python
# New feature: Get both raw predictions and refined windows
refined_windows, predictions = model.predict_windows(
    waveform[:, 20:-20, :], 
    return_predictions=True
)

# predictions: Raw model activations (N, C, T) 
# refined_windows: Half-cycle cropped boundaries (N, 2)
```

---

## � Data Requirements & Preprocessing

**Important**: PAW models have specific data requirements. 

### Required Data Format
- **Sampling Rate**: 40 Hz (model requirement)
- **Window Duration**: 5 seconds (200 samples at 40 Hz)
- **Shape**: `(N, T, C)` where N=batch, T=time samples, C=channels
- **Padding**: 20 samples on each side (handled automatically by model)
- **Preprocessing**: Detrend → Filter (1-15 Hz) → Resample → Normalize

### Docker Users (SAC Files): ✅ **Preprocessing Handled Automatically**
When using Docker with SAC files, **all preprocessing is done automatically**:
```bash
# Docker automatically handles: detrend, filter, resample, normalize
podman run --rm \
  -v $(pwd)/data:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_predict.py /data/signal.sac 10.5 /output/result.png
```

### Python Users: Manual Preprocessing Required

```python
from pawlib import preprocess_for_paw, load_sac_waveform
import numpy as np

# 1. Load SAC file with proper windowing
waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5)

# 2. Apply required preprocessing pipeline
waveform = preprocess_for_paw(
    waveform, 
    sampling_rate=meta['sampling_rate'],  # Original rate (e.g., 20 Hz)
    target_freq=40.0,                     # Required: 40 Hz
    freqmin=1.0,                         # Required: 1-15 Hz filter
    freqmax=15.0
)

# 3. Add required padding
waveform = np.pad(waveform, ((0,0), (20,20), (0,0)), mode='constant')

# 4. Ready for model inference
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
windows = model.predict_windows(waveform[:, 20:-20, :])
```

### Individual Preprocessing Steps
```python
from pawlib import filter_waveform, resample_waveform, normalize_waveform

# Step-by-step preprocessing for custom workflows
waveform = filter_waveform(waveform, sampling_rate=20.0, freqmin=1.0, freqmax=15.0)
waveform = resample_waveform(waveform, original_freq=20.0, target_freq=40.0)
waveform = normalize_waveform(waveform, method='max')
```

---

## �🔍 Python Usage Examples

### Single SAC File Prediction with Half-Cycle Cropping

```python
from pawlib import PAW, load_sac_waveform, preprocess_for_paw
import numpy as np

# 1. Load SAC file (5-second window around onset)
waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5)

# 2. Preprocess to PAW format (40 Hz)
waveform = preprocess_for_paw(waveform, meta['sampling_rate'])
waveform = np.pad(waveform, ((0,0), (20,20), (0,0)))  # Required padding

# 3. Run inference with automatic half-cycle cropping
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
windows = model.predict_windows(waveform[:, 20:-20, :])

# 4. Get results
start_time = windows[0, 0] / 40.0  # Convert to seconds  
end_time = windows[0, 1] / 40.0
print(f"Detected half-cycle window: {start_time:.3f}s - {end_time:.3f}s")
print(f"Duration: {end_time - start_time:.3f}s")
```

### Getting Raw Predictions + Windows

```python
# Get both model activations and refined windows
windows, predictions = model.predict_windows(
    waveform[:, 20:-20, :],
    return_predictions=True
)

# predictions: Raw PAW activations (N, C, T) for visualization
# windows: Half-cycle cropped boundaries (N, 2) for analysis
```

## 🎓 Advanced Training & Testing

### Training Your Own Model

```python
from pawlib import PAW

# Initialize model with custom architecture
model = PAW(
    device='cuda',  # or 'cpu'
    architecture='paw_reference',  # Model architecture
)

# Train with HDF5 dataset
history = model.train(
    data='dataset.h5',        # HDF5 file with 'waveforms' and 'labels'  
    epochs=100,
    batch_size=64,
    loss='dice',              # Options: 'dice', 'bce', 'amper', 'bce_dice'
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping=True,
    patience=10
)

# Save trained model
model.save('my_model.pt', metadata={'version': '1.0', 'dataset': 'custom'})
```

### Custom Dataset Preparation

```python
import h5py
import numpy as np

# Create HDF5 dataset for training
with h5py.File('dataset.h5', 'w') as f:
    # Waveforms: (N, 200, 1) at 40 Hz, 5-second windows
    f.create_dataset('waveforms', data=waveforms)  
    
    # Labels: (N, 2) with [start_time, end_time] in seconds
    f.create_dataset('labels', data=labels)
    
    # Optional metadata
    f.attrs['sampling_rate'] = 40.0
    f.attrs['window_duration'] = 5.0
```

### Model Evaluation & Testing

```python
# Comprehensive model testing
test_results = model.test(
    data='test_dataset.h5',
    batch_size=32,
    apply_half_cycle_cropping=True
)

print(f"Window Accuracy: {test_results['window_accuracy']:.4f}")
print(f"Amplitude RMSE: {test_results['amplitude_rmse']:.4f}")
print(f"Period RMSE: {test_results['period_rmse']:.4f}")
print(f"Dice Score: {test_results['dice_score']:.4f}")

# Test on specific subsets
subset_results = model.test_subsets(
    data='test_dataset.h5',
    subsets={
        'high_snr': [0, 100, 200],      # Sample indices
        'low_snr': [101, 201, 301],
        'complex': [500, 600, 700]
    }
)
```

### Advanced Training Options

```python
# Custom loss functions and metrics
from pawlib.losses import BCEDiceLoss, AmpPerLoss
from pawlib.metrics import WindowAccuracy, AmplitudeRMSE

model = PAW()

# Multi-loss training with custom weights
history = model.train(
    data='dataset.h5',
    epochs=100,
    loss={
        'dice': 0.7,        # 70% Dice loss weight
        'bce': 0.2,         # 20% BCE loss weight  
        'amper': 0.1        # 10% Amplitude-Period loss weight
    },
    scheduler='cosine',     # Learning rate scheduling
    warmup_epochs=10,
    gradient_clipping=1.0
)
```

### Distributed Training

```python
# Multi-GPU training
import torch.distributed as dist

# Initialize distributed training
model = PAW(device='cuda')

# Train with DataParallel or DistributedDataParallel
history = model.train(
    data='large_dataset.h5',
    epochs=200,
    batch_size=128,
    distributed=True,
    world_size=4,           # Number of GPUs
    rank=0                  # Current GPU rank
)
```

### Hyperparameter Tuning

```python
from itertools import product

# Grid search over hyperparameters
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'loss': ['dice', 'bce_dice'],
}

best_score = 0
best_params = {}

for lr, bs, loss in product(*param_grid.values()):
    model = PAW()
    history = model.train(
        data='train_dataset.h5',
        validation_data='val_dataset.h5',
        epochs=50,
        learning_rate=lr,
        batch_size=bs,
        loss=loss,
        verbose=False
    )
    
    val_score = max(history['val_dice_score'])
    if val_score > best_score:
        best_score = val_score
        best_params = {'lr': lr, 'batch_size': bs, 'loss': loss}
        model.save(f'best_model_{val_score:.4f}.pt')

print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score:.4f}")
```

### Model Analysis & Interpretation

```python
# Analyze model predictions vs ground truth
import matplotlib.pyplot as plt

# Get detailed predictions for analysis
waveforms, labels = load_test_data('test_dataset.h5')
windows, predictions = model.predict_windows(
    waveforms, 
    return_predictions=True
)

# Visualize model activations
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Raw waveform
axes[0].plot(waveforms[0, :, 0])
axes[0].set_title('Raw Waveform')

# Model activations
axes[1].plot(predictions[0, 0, :])
axes[1].set_title('PAW Model Activations')

# Ground truth vs prediction windows
axes[2].axvspan(labels[0, 0]*40, labels[0, 1]*40, alpha=0.3, label='Ground Truth')
axes[2].axvspan(windows[0, 0], windows[0, 1], alpha=0.3, label='Prediction')
axes[2].plot(waveforms[0, :, 0])
axes[2].set_title('Window Comparison')
axes[2].legend()

plt.tight_layout()
plt.show()
```

### Production Deployment

```python
# Optimize model for production inference
model = PAW.from_pretrained('trained_model.pt')

# Compile for faster inference (PyTorch 2.0+)
model.model = torch.compile(model.model, mode='max-autotune')

# Batch prediction for efficiency
batch_waveforms = load_batch_data(batch_size=128)
batch_windows = model.predict_windows(batch_waveforms)

# Convert to deployment format
results = {
    'windows': batch_windows.tolist(),
    'timestamps': [(w[0]/40.0, w[1]/40.0) for w in batch_windows],
    'durations': [(w[1]-w[0])/40.0 for w in batch_windows]
}
```

## 💡 API Reference

### SAC File Loading

```python
from pawlib import load_sac_waveform, batch_load_sac_files

# Load single file
waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5, window_duration=5.0, padding=0.5)

# Batch load
files = ['event1.sac', 'event2.sac', 'event3.sac']
onsets = [10.5, 15.2, 8.7]
waveforms, metadata = batch_load_sac_files(files, onsets)
```

### Preprocessing

```python
from pawlib import preprocess_for_paw, normalize_waveform, filter_waveform, resample_waveform

# Complete pipeline
waveform = preprocess_for_paw(waveform, sampling_rate=20.0, target_freq=40.0, 
                               freqmin=1.0, freqmax=15.0)

# Individual operations
waveform = filter_waveform(waveform, sampling_rate=20.0, freqmin=1.0, freqmax=15.0)
waveform = resample_waveform(waveform, original_freq=20.0, target_freq=40.0)
waveform = normalize_waveform(waveform, method='max')
```

### Model Operations

```python
# Load pretrained
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')

# Inference
predictions = model.predict(waveforms)  # Returns: (N, 1, T)
windows = extract_windows_from_prediction(predictions)  # Returns: (N, 2)

# Training
model = PAW()
history = model.train(data='dataset.h5', epochs=100, loss='dice')
model.save('my_model.pt')
```

---

## 📝 Important Notes

### Data Format
- **Input:** Waveforms shape `(N, T, 1)` - typically `(N, 200, 1)` at 40 Hz
- **Labels:** `(N, 2)` containing `[start_time, end_time]` in seconds
- **Padding:** Model requires 20 samples padding on each side (use `np.pad`)

### Key Parameters
- **Sampling rate:** 40 Hz (model requirement)
- **Window duration:** 5 seconds (default)
- **Loss functions:** `'dice'` (recommended), `'bce'`, `'amper'`, `'bce_dice'`

## 📚 More Examples

| File | Description | Key Features |
|------|-------------|--------------|
| `inference_demo.ipynb` | Interactive Jupyter notebook workflow | Half-cycle cropping, dual visualization, real data |
| `examples/docker_predict.py` | Single file prediction (Docker-ready) | GPU support, automatic visualization |
| `examples/docker_batch_predict.py` | Batch processing with CSV output | High-throughput processing |
| `examples/quick_start.py` | Basic local usage example | Simple API demonstration |
| `examples/high_level_api.py` | Advanced training example | Custom architectures, metrics |

## 🐛 Common Issues

1. **Missing ObsPy:** `pip install obspy` (required for SAC files)
2. **GPU not detected:** Ensure CUDA drivers are installed
3. **Import errors:** Reinstall with `pip install -e ".[dev]"` from repo root
4. **Docker permission denied:** Use `podman` (rootless) or add user to docker group

## 📄 License

Proprietary - Internal Use Only. See LICENSE file for details.