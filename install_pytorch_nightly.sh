#!/bin/bash
# Script to install PyTorch nightly builds for RTX 5090 (sm_120) support
# This script installs PyTorch nightly builds with CUDA 12.8 support

set -e

echo "Installing PyTorch nightly builds for RTX 5090 (sm_120) support..."
echo "This will replace your current PyTorch installation with nightly builds."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv package manager..."
    uv pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "Using pip..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

echo ""
echo "Installation complete!"
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
else:
    print('CUDA is not available')
"

echo ""
echo "If you see your RTX 5090 GPU name and CUDA capability (12, 0), the installation was successful!"

