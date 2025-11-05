# PAWlib

A clean, production-ready PyTorch library for seismic waveform analysis using the PAW (Picker for Arrival of Waves) model.

**Features:**
- 🚀 High-level API for training, testing, and inference
- 📊 Comprehensive metrics (Window Accuracy, Dice Score, RMSE)
- 🔧 Configurable loss functions (Dice, BCE, AmpPer)
- 💾 Model checkpointing and loading
- 🎨 Automatic visualization generation

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

---

## 🎯 Quick Start

### Basic Usage

```python
from pawlib import PAW
import numpy as np

# Create model
model = PAW()

# Train on your data
history = model.train(
    data='your_dataset.h5',  # or numpy array
    epochs=100,
    batch_size=64,
    loss='dice'
)

# Make predictions
predictions = model.predict(test_data)

# Evaluate
results = model.test(test_data, test_labels)
```

## 💡 API Reference

### Training

```python
model = PAW()

# Train from HDF5 file
history = model.train(
    data='dataset.h5',
    epochs=100,
    batch_size=64,
    loss='dice',  # Options: 'dice', 'bce', 'amper', 'bce_dice'
    checkpoint_dir='my_checkpoints'  # Optional
)

# Or train from numpy arrays
data = np.random.randn(1000, 200, 1).astype(np.float32)
labels = np.array([[0.5, 1.0]] * 1000)  # [start, end] in seconds
history = model.train(data, labels, epochs=50)
```

### Inference

```python
# Make predictions
predictions = model.predict(waveforms)
# Returns: binary masks (N, 1, 240)

# Evaluate on test set
results = model.test(test_waveforms, test_labels)
print(f"Accuracy: {results['window_accuracy']:.2%}")
```

### Save & Load

```python
# Save trained model
model.save('my_model.pt')

# Load model
model = PAW.load('my_model.pt')
```

---

## 🔧 Configuration

### Loss Functions

- **`dice`** (recommended) - Combines BCE + Dice + Temporal Consistency
- **`bce`** - Binary Cross Entropy
- **`amper`** - Amplitude-Period loss
- **`bce_dice`** - Combined BCE and Dice

### Training Parameters

```python
model.train(
    data='dataset.h5',
    epochs=100,           # Number of training epochs
    batch_size=64,        # Batch size
    loss='dice',          # Loss function
    lr=1e-3,             # Learning rate
    val_split=0.2,       # Validation split ratio
    checkpoint_dir='checkpoints',  # Save directory
    save_best=True,      # Save best model
    verbose=True         # Print progress
)
```

---

## 📊 Metrics

The library automatically computes:

- **Window Accuracy** - Exact window boundary matching
- **Dice Score** - Overlap between predicted and true masks
- **Amplitude RMSE** - Amplitude prediction error
- **Period RMSE** - Period prediction error
- **Magnitude RMSE** - Magnitude prediction error
- **Window RMSE** - Window boundary error

All metrics are returned as a dictionary from `model.test()`

## 📝 Data Format

### Input Data

**Waveforms:**
- Shape: `(N, T, C)` where N=samples, T=timesteps, C=channels
- Type: numpy array or HDF5 file with 'waveforms' dataset
- Typical: `(N, 200, 1)` for single-channel data

**Labels:**
- Shape: `(N, 2)` with `[start_time, end_time]` in seconds
- Type: numpy array or HDF5 file with 'labels' dataset
- Example: `[[0.5, 1.0], [1.2, 1.7], ...]`

### Model Architecture

- **Type**: CNN + LSTM + Transformer hybrid
- **Input**: Waveform sequences (automatically padded)
- **Output**: Binary masks indicating phase windows
- **Sampling**: 40 Hz (0.025s per sample)

## 📚 Examples

See the `examples/` directory for complete working examples:

- **`quick_start.py`** - Basic training and inference
- **`high_level_api.py`** - Advanced usage patterns


## 🐛 Troubleshooting

### Import Error
```bash
# Check installation
pip list | grep pawlib

# Reinstall
pip uninstall pawlib
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git
```

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```