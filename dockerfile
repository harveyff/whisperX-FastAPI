FROM nvidia/cuda:13.0.1-base-ubuntu22.04

# Build argument to choose PyTorch version (stable or nightly for RTX 5090)
ARG USE_PYTORCH_NIGHTLY=false

ENV PYTHON_VERSION=3.11
# Set LD_LIBRARY_PATH to prioritize PyTorch's bundled NCCL libraries
# PyTorch includes its own NCCL libraries that are compatible
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH

# Install dependencies and clean up in the same layer
# Fix dpkg configuration issues by running configure first
# Install NCCL for PyTorch distributed operations from NVIDIA official repository
# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && dpkg --configure -a || true \
    && apt-get -y install --no-install-recommends \
    wget \
    ca-certificates \
    gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb || true \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libcudnn9-cuda-12 \
    libnccl2 \
    libnccl-dev \
    libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install UV for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY app app/
COPY tests tests/
COPY app/gunicorn_logging.conf .

# Install Python dependencies using UV with pyproject.toml
# UV automatically selects CUDA 12.8 wheels on Linux (compatible with CUDA 13.0 runtime)
# First install all dependencies, then fix torchaudio compatibility
RUN uv pip install --system . \
    && uv pip install --system ctranslate2==4.6.0

# Fix torchaudio compatibility with pyannote.audio
# pyannote.audio requires torchaudio with AudioMetaData (available in torchaudio < 2.4)
# Try installing compatible version from CUDA 11.8 or 12.1 index (compatible with CUDA 12.8 runtime)
RUN echo "Fixing torchaudio compatibility for pyannote.audio..." \
    && uv pip uninstall --system torchaudio || true \
    && (uv pip install --system --index-url https://download.pytorch.org/whl/cu118 "torchaudio==2.3.1+cu118" || \
        uv pip install --system --index-url https://download.pytorch.org/whl/cu121 "torchaudio==2.3.1+cu121" || \
        uv pip install --system --index-url https://download.pytorch.org/whl/cu118 "torchaudio==2.2.2+cu118" || \
        echo "Warning: Could not install compatible torchaudio version")

# Install PyTorch - use nightly for RTX 5090 support if requested
# Only upgrade torch and torchvision, torchaudio is already at compatible version
RUN if [ "$USE_PYTORCH_NIGHTLY" = "true" ]; then \
        echo "Installing PyTorch nightly for RTX 5090 support..." \
        && uv pip uninstall --system torch torchvision || true \
        && uv pip install --system --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision; \
    else \
        echo "Upgrading PyTorch to latest stable version..." \
        && uv pip uninstall --system torch torchvision || true \
        && uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch torchvision; \
    fi \
    && python -c "import torch; import torchaudio; print(f'PyTorch: {torch.__version__}, torchaudio: {torchaudio.__version__}')" \
    && python -c "import torchaudio; assert hasattr(torchaudio, 'AudioMetaData'), 'AudioMetaData not found in torchaudio'; print('✓ AudioMetaData available')" \
    && python -c "import torch; import os; torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'PyTorch lib path: {torch_lib}')" \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

# Create startup script that sets LD_LIBRARY_PATH before starting gunicorn
# This ensures PyTorch's bundled NCCL libraries are found first
RUN echo '#!/bin/bash\n\
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH\n\
exec python -m gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 0 --log-config gunicorn_logging.conf app.main:app -k uvicorn.workers.UvicornWorker "$@"' > /usr/local/bin/start.sh \
    && chmod +x /usr/local/bin/start.sh

EXPOSE 8000

# Use startup script to ensure PyTorch's NCCL libraries are prioritized
ENTRYPOINT ["/usr/local/bin/start.sh"]
