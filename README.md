# PAWlib

A clean, production-ready PyTorch library for seismic phase arrival detection using the PAW model.

**Features:**
- 🚀 High-level API for training and inference
- 📁 SAC file loading with ObsPy integration
- 🔧 Preprocessing utilities (filter, resample, normalize)
- 📊 Comprehensive metrics and loss functions
- 💾 Hugging Face Hub integration for pretrained models

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

### 🐳 Container Option (Docker/Podman)

For reproducible environments, use containers:

```bash
# Podman (rootless, no sudo needed)
./install_podman.sh
podman-compose -f podman-compose.yml up -d pawlib

# Or Docker
docker-compose up -d pawlib
```

See [USAGE.md](USAGE.md) for complete container documentation.

---

## 🎯 Quick Start

### Inference with Pretrained Model

```python
from pawlib import PAW, load_sac_waveform, preprocess_for_paw, extract_windows_from_prediction
import numpy as np

# Load SAC file
waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5, window_duration=5.0)

# Preprocess
waveform = preprocess_for_paw(waveform, meta['sampling_rate'], target_freq=40.0)
waveform = np.pad(waveform, ((0,0), (20,20), (0,0)))  # Add model padding

# Load model and predict
model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
prediction = model.predict(waveform[:, 20:-20, :])
windows = extract_windows_from_prediction(prediction)

print(f"Detected window: {windows[0]} samples")
```

### Training

```python
# Train on your data
model = PAW()
history = model.train(data='dataset.h5', epochs=100, batch_size=64, loss='dice')
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

## 📝 Data Format

**Waveforms:** `(N, T, 1)` shape - typically `(N, 200, 1)` at 40 Hz  
**Labels:** `(N, 2)` - `[start_time, end_time]` in seconds  
**Model expects:** 20 samples padding on each side (use `np.pad`)

**Loss functions:** `'dice'`, `'bce'`, `'amper'`, `'bce_dice'`  
**Metrics:** Window Accuracy, Dice Score, RMSE (amplitude, period, window)

## 📚 Examples

- **`inference_demo.ipynb`** - Complete inference workflow with SAC files
- **`examples/quick_start.py`** - Basic training and inference
- **`examples/high_level_api.py`** - Advanced usage patterns

## 🐛 Troubleshooting

**Import errors:** `pip uninstall pawlib && pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git`  
**GPU issues:** Check with `python -c "import torch; print(torch.cuda.is_available())"`  
**ObsPy required:** Install with `pip install obspy` for SAC file support