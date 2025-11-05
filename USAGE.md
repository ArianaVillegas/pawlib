# PAWlib Usage Guide

Quick guide for using pawlib with Python or containers.

---

## 📦 Installation Options

### Option 1: Python Package (Recommended for Development)

```bash
# Install from GitHub
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git

# Or install from source
git clone git@github.com:ArianaVillegas/pawlib.git
cd pawlib
pip install -e .
```

### Option 2: Docker Container

```bash
# Build and run
docker-compose up -d pawlib

# Access container
docker exec -it pawlib bash

# Use pawlib
docker exec pawlib python -c "from pawlib import PAW; model = PAW()"
```

### Option 3: Podman Container (Rootless - No sudo!)

```bash
# Install Podman (first time only)
./install_podman.sh

# Build and run
podman-compose -f podman-compose.yml up -d pawlib

# Access container
podman exec -it pawlib bash

# Use pawlib
podman exec pawlib python -c "from pawlib import PAW; model = PAW()"
```

---

## 🚀 Quick Start

### Basic Training

```python
from pawlib import PAW

# Create model
model = PAW()

# Train
history = model.train(
    data='dataset.h5',
    epochs=100,
    batch_size=64,
    loss='dice'
)

# Make predictions
predictions = model.predict(test_data)

# Evaluate
results = model.test(test_data, test_labels)
print(f"Accuracy: {results['window_accuracy']:.2%}")
```

### Save & Load Models

```python
# Save
model.save('my_model.pt')

# Load
from pawlib import PAW
model = PAW.load('my_model.pt')
```

---

## 🐳 Container Usage

### Data Volumes

Containers automatically mount these directories:

| Host Directory | Container Path | Purpose |
|----------------|----------------|---------|
| `./data/` | `/workspace/data/` | Your datasets |
| `./outputs/` | `/workspace/outputs/` | Training outputs |
| `./checkpoints/` | `/workspace/checkpoints/` | Saved models |
| `./notebooks/` | `/workspace/notebooks/` | Jupyter notebooks |

**Example workflow:**

```bash
# 1. Prepare data on your machine
mkdir -p data outputs checkpoints
cp your_dataset.h5 ./data/

# 2. Start container
podman-compose -f podman-compose.yml up -d pawlib

# 3. Train inside container
podman exec pawlib python << 'EOF'
from pawlib import PAW

model = PAW()
model.train(
    data='/workspace/data/your_dataset.h5',
    epochs=100,
    checkpoint_dir='/workspace/checkpoints'
)
model.save('/workspace/checkpoints/final_model.pt')
EOF

# 4. Results are saved on your machine
ls checkpoints/  # See saved models
ls outputs/      # See visualizations
```

### Interactive Development

```bash
# Enter container shell
podman exec -it pawlib bash

# Inside container - use pawlib normally
python
>>> from pawlib import PAW
>>> model = PAW()
>>> # Your code here...
```

### Jupyter Notebook (Optional)

```bash
# Start Jupyter server
podman-compose -f podman-compose.yml up -d jupyter

# Get access token from logs
podman-compose -f podman-compose.yml logs jupyter

# Open browser at: http://localhost:8888
# Enter the token from logs
```

---

## 🔧 Configuration

### Loss Functions

- `dice` (recommended) - Combines BCE + Dice + Temporal Consistency
- `bce` - Binary Cross Entropy
- `amper` - Amplitude-Period loss
- `bce_dice` - Combined BCE and Dice

### Training Parameters

```python
model.train(
    data='dataset.h5',       # HDF5 file or numpy array
    epochs=100,              # Number of epochs
    batch_size=64,           # Batch size
    loss='dice',             # Loss function
    lr=1e-3,                 # Learning rate
    val_split=0.2,           # Validation split
    checkpoint_dir='./checkpoints',  # Save directory
    save_best=True,          # Save best model
    verbose=True             # Print progress
)
```

---

## 📊 Data Format

### Input Waveforms

- Shape: `(N, T, C)` where N=samples, T=timesteps, C=channels
- Type: numpy array or HDF5 file with 'waveforms' dataset
- Example: `(1000, 200, 1)` for 1000 single-channel waveforms

### Labels

- Shape: `(N, 2)` with `[start_time, end_time]` in seconds
- Type: numpy array or HDF5 file with 'labels' dataset
- Example: `[[0.5, 1.0], [1.2, 1.7], ...]`

---

## 🐛 Troubleshooting

### Import Error

```bash
# Check installation
pip list | grep pawlib

# Reinstall
pip uninstall pawlib
pip install git+ssh://git@github.com/ArianaVillegas/pawlib.git
```

### Container Won't Start

```bash
# Check logs
podman-compose -f podman-compose.yml logs pawlib

# Restart
podman-compose -f podman-compose.yml down
podman-compose -f podman-compose.yml up -d pawlib
```

### Permission Issues with Volumes

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER data/ outputs/ checkpoints/
```

---

## 📚 Examples

See the `examples/` directory for complete examples:

- `quick_start.py` - Basic training and inference
- `high_level_api.py` - Advanced usage patterns

---

## 💡 Tips

- **Development:** Use pip install for quick iteration
- **Production:** Use containers for reproducibility
- **GPU:** Containers use CPU by default (GPU setup in install_podman.sh)
- **Data:** Always mount data as volumes, don't copy into containers
- **Checkpoints:** Save to mounted volumes to persist models

