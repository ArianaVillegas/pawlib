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
- [Installation](#-installation)
- [Docker/Podman Usage](#-dockerpodman-usage)
- [Python Usage Examples](#-python-usage-examples)
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
podman run --rm --device nvidia.com/gpu=all \
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

### Basic Prediction
```bash
# Using your own SAC files
podman run --rm --device nvidia.com/gpu=all \
  -v /path/to/your/sac/files:/data:Z \
  pawlib:latest python examples/docker_predict.py /data/your_file.sac 10.5
```

### Batch Processing
```bash
# Process all SAC files in a directory
podman run --rm --device nvidia.com/gpu=all \
  -v /path/to/sac/files:/data:Z \
  -v $(pwd)/outputs:/output:Z \
  pawlib:latest python examples/docker_batch_predict.py /data /output/results.csv
```

### Docker Alternative
```bash
# Replace 'podman' with 'docker' and adjust GPU flag
docker run --rm --gpus all \
  -v /path/to/your/sac/files:/data \
  pawlib:latest python examples/docker_predict.py /data/signal.sac 10.5
```

---

## 🔍 Python Usage Examples

### Single SAC File Prediction

```python
from pawlib import PAW, load_sac_waveform, preprocess_for_paw, extract_windows_from_prediction
import numpy as np

# 1. Load SAC file (5-second window around onset)
waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5)

# 2. Preprocess to PAW format (40 Hz)
waveform = preprocess_for_paw(waveform, meta['sampling_rate'])
waveform = np.pad(waveform, ((0,0), (20,20), (0,0)))  # Required padding

# 3. Run inference
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
prediction = model.predict(waveform[:, 20:-20, :])
windows = extract_windows_from_prediction(prediction)

# 4. Get results
start_time = windows[0, 0] / 40.0  # Convert to seconds
end_time = windows[0, 1] / 40.0
print(f"Detected amplitude window: {start_time:.3f}s - {end_time:.3f}s")
```

### Training Your Own Model

```python
model = PAW()
history = model.train(
    data='dataset.h5',    # HDF5 file with 'waveforms' and 'labels'
    epochs=100,
    batch_size=64,
    loss='dice'          # Options: 'dice', 'bce', 'amper', 'bce_dice'
)
model.save('my_model.pt')
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

| File | Description |
|------|-------------|
| `inference_demo.ipynb` | Interactive Jupyter notebook workflow |
| `examples/docker_predict.py` | Single file prediction (Docker-ready) |
| `examples/docker_batch_predict.py` | Batch processing with CSV output |
| `examples/download_sample_sac.py` | Download real seismic data samples |

## 🐛 Common Issues

1. **Missing ObsPy:** `pip install obspy` (required for SAC files)
2. **GPU not detected:** Ensure CUDA drivers are installed
3. **Import errors:** Reinstall with `pip install -e ".[dev]"` from repo root
4. **Docker permission denied:** Use `podman` (rootless) or add user to docker group

## 📄 License

Proprietary - Internal Use Only. See LICENSE file for details.