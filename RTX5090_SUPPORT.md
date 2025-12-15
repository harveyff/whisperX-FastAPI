# RTX 5090 (sm_120) Support Guide

## Problem

If you're using an NVIDIA RTX 5090 GPU and encounter this error:

```
NVIDIA GeForce RTX 5090 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
CUDA error: no kernel image is available for execution on the device
```

This is because RTX 5090 uses CUDA compute capability 12.0 (sm_120), which requires PyTorch nightly builds.

## Solution

### Option 1: Docker Build (Recommended)

Build the Docker image with RTX 5090 support:

```bash
docker build --build-arg RTX5090_SUPPORT=true -t whisperx-service .
docker run -d --gpus all -p 8000:8000 --env-file .env whisperx-service
```

Or update `docker-compose.yml`:

```yaml
services:
  whisperx-service:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        RTX5090_SUPPORT: "true"
```

Then run:
```bash
docker-compose up
```

### Option 2: Local Installation

If running locally (not in Docker):

**Using uv:**
```bash
uv pip uninstall torch torchvision torchaudio
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Using pip:**
```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Or use the provided script:**
```bash
chmod +x install_pytorch_nightly.sh
./install_pytorch_nightly.sh
```

### Verification

After installation, verify PyTorch can detect your GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
```

You should see:
- CUDA available: `True`
- Device name: Your RTX 5090 GPU name
- CUDA capability: `(12, 0)` for sm_120

## Important Notes

1. **PyTorch nightly builds** may contain untested features and should be used with caution in production environments.

2. **NVIDIA Driver Requirements**: Ensure you have NVIDIA drivers version 566.03 or later installed.

3. **CUDA Version**: RTX 5090 requires CUDA 12.8+ support.

4. **Compatibility**: This configuration is specifically for RTX 5090. For other GPUs, use the standard installation without nightly builds.

## Troubleshooting

If you still encounter issues:

1. **Check NVIDIA driver version:**
   ```bash
   nvidia-smi
   ```
   Update to 566.03+ if needed.

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```
   Should show CUDA 12.8 or later.

3. **Check PyTorch installation:**
   ```python
   import torch
   print(torch.__version__)
   print(torch.version.cuda)
   ```

4. **Reinstall PyTorch nightly:**
   ```bash
   # Uninstall completely
   pip uninstall -y torch torchvision torchaudio
   # Install nightly
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA RTX 5090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-5090/)

