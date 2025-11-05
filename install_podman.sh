#!/bin/bash
# Podman installation script for pawlib
# Rootless container runtime - no sudo needed for usage!

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Installing Podman (Rootless Docker Alternative)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "❌ Please run this script as a normal user (not root/sudo)"
    exit 1
fi

echo "📦 Step 1: Installing Podman..."
sudo apt update
sudo apt install -y podman

echo ""
echo "🐍 Step 2: Installing podman-compose..."
pip install --user podman-compose

echo ""
echo "🎮 Step 3: Setting up GPU support (NVIDIA)..."
if command -v nvidia-smi &> /dev/null; then
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    echo "✅ GPU support configured"
else
    echo "⚠️  nvidia-smi not found - skipping GPU setup"
    echo "   If you have a GPU, install NVIDIA drivers first"
fi

echo ""
echo "✅ Step 4: Verifying installation..."
podman --version
podman-compose --version

echo ""
echo "🧪 Step 5: Testing GPU access..."
if [ -f /etc/cdi/nvidia.yaml ]; then
    echo "Running GPU test..."
    if podman run --rm --device nvidia.com/gpu=all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>/dev/null; then
        echo "✅ GPU access works!"
    else
        echo "⚠️  GPU test failed - you may need to configure CDI"
    fi
else
    echo "⚠️  Skipping GPU test (no GPU detected)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Installation Complete!                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Next steps:"
echo ""
echo "   1. Build pawlib image:"
echo "      cd /home/ariana/AFRL/Seismology/pawlib"
echo "      podman-compose -f podman-compose.yml build"
echo ""
echo "   2. Run container (NO SUDO NEEDED!):"
echo "      podman-compose -f podman-compose.yml up -d pawlib"
echo ""
echo "   3. Access container:"
echo "      podman exec -it pawlib bash"
echo ""
echo "   4. Test pawlib:"
echo "      podman exec pawlib python -c 'from pawlib import PAW; print(\"Works!\")'"
echo ""
echo "💡 Tip: You can alias docker to podman if you want:"
echo "   echo 'alias docker=podman' >> ~/.bashrc"
echo "   echo 'alias docker-compose=podman-compose' >> ~/.bashrc"
echo ""
