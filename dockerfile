# Use CUDA runtime image which includes more complete library set including newer NCCL
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# Build argument to choose PyTorch version (stable or nightly for RTX 5090)
ARG USE_PYTORCH_NIGHTLY=false

ENV PYTHON_VERSION=3.11
# LD_LIBRARY_PATH will be set after NCCL installation to ensure correct library is found
# System NCCL (if version 2.18+) takes priority, then PyTorch bundled libraries
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/lib

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
    libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install latest NCCL from NVIDIA repository (needed for PyTorch 2.8+)
# PyTorch 2.8 requires NCCL 2.18+ which includes ncclGroupSimulateEnd symbol
# Try multiple methods to get compatible NCCL version
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends binutils wget \
    && echo "Attempting to install NCCL 2.18+..." \
    && (apt-get -y install --no-install-recommends libnccl2=2.19.3-1+cuda12.0 2>&1 || \
        apt-get -y install --no-install-recommends libnccl2=2.18.5-1+cuda12.0 2>&1 || \
        apt-get -y install --no-install-recommends libnccl2=2.18.3-1+cuda12.0 2>&1 || \
        (echo "Trying to download and install NCCL 2.19.3 directly..." \
         && cd /tmp \
         && wget -q --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnccl2_2.19.3-1+cuda12.0_amd64.deb 2>&1 || \
         wget -q --no-check-certificate https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/libnccl2_2.19.3-1+cuda12.0_amd64.deb 2>&1 || true \
         && if [ -f /tmp/libnccl2_*.deb ]; then \
             dpkg -i /tmp/libnccl2_*.deb || apt-get install -f -y || true \
             && rm -f /tmp/libnccl2_*.deb; \
         fi) || \
        apt-get -y install --no-install-recommends libnccl2 || true) \
    && apt-get -y install --no-install-recommends libnccl-dev || true \
    && NCCL_VERSION=$(dpkg -l | grep "^ii.*libnccl2" | awk '{print $3}' | cut -d'+' -f1 2>/dev/null || echo "unknown") \
    && echo "Installed NCCL version: $NCCL_VERSION" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && echo "Verifying NCCL installation..." \
    && find /usr -name "libnccl.so*" 2>/dev/null | head -5 \
    && if [ -f "/usr/lib/x86_64-linux-gnu/libnccl.so.2" ]; then \
        echo "Checking for ncclGroupSimulateEnd symbol..." \
        && if nm -D /usr/lib/x86_64-linux-gnu/libnccl.so.2 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then \
            echo "✓ Symbol found in system NCCL - version is compatible" \
            && echo "/usr/lib/x86_64-linux-gnu" > /tmp/nccl_lib_path.txt; \
        else \
            echo "✗ Symbol NOT found in system NCCL" \
            && echo "" > /tmp/nccl_lib_path.txt; \
        fi; \
    else \
        echo "System NCCL library not found" \
        && echo "" > /tmp/nccl_lib_path.txt; \
    fi

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
    && python -c "import torch; import os; torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'PyTorch lib path: {torch_lib}'); import glob; nccl_libs = glob.glob(os.path.join(torch_lib, '*nccl*')); print(f'NCCL libraries in PyTorch: {nccl_libs}')" \
    && ldconfig \
    && echo "Determining correct NCCL library to use..." \
    && TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib" \
    && SYSTEM_NCCL="/usr/lib/x86_64-linux-gnu/libnccl.so.2" \
    && NCCL_TO_USE="" \
    && if [ -f "$SYSTEM_NCCL" ] && nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then \
        echo "✓ Using system NCCL (has required symbol)" \
        && NCCL_TO_USE="$SYSTEM_NCCL" \
        && echo "$SYSTEM_NCCL" > /etc/pytorch_nccl_lib.txt; \
    elif [ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 1> /dev/null 2>&1; then \
        NCCL_TO_USE=$(ls "$TORCH_LIB"/libnccl*.so* 2>/dev/null | head -1) \
        && echo "✓ Using PyTorch bundled NCCL: $NCCL_TO_USE" \
        && echo "$NCCL_TO_USE" > /etc/pytorch_nccl_lib.txt; \
    else \
        echo "ERROR: No compatible NCCL found!" \
        && exit 1; \
    fi \
    && echo "NCCL library to use: $(cat /etc/pytorch_nccl_lib.txt)" \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

# Create startup script that sets environment variables before starting gunicorn
# Intelligently handle NCCL library loading to avoid symbol errors
# Use LD_PRELOAD to force loading correct NCCL library if needed
RUN cat > /usr/local/bin/start.sh << 'EOF' && chmod +x /usr/local/bin/start.sh
#!/bin/bash
set -e

# Update library cache
ldconfig

# Try to use NCCL library determined at build time (preferred method)
if [ -f /etc/pytorch_nccl_lib.txt ]; then
    NCCL_LIB=$(cat /etc/pytorch_nccl_lib.txt)
    if [ -f "$NCCL_LIB" ]; then
        echo "Using build-time determined NCCL library: $NCCL_LIB"
        export LD_PRELOAD="$NCCL_LIB"
        NCCL_DIR=$(dirname "$NCCL_LIB")
        TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib"
        export LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        # Skip runtime detection and go directly to testing
        echo "Testing PyTorch import with build-time NCCL..."
        python -c "import torch; print(f'✓ PyTorch {torch.__version__} imported successfully')" || {
            echo "ERROR: PyTorch import failed with build-time NCCL"
            exit 1
        }
        export NCCL_P2P_DISABLE=1
        export NCCL_SHM_DISABLE=1
        exec python -m gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 0 --log-config gunicorn_logging.conf app.main:app -k uvicorn.workers.UvicornWorker "$@"
    fi
fi

# Fallback to runtime detection if build-time file not found
# Find NCCL libraries
TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib"
SYSTEM_NCCL="/usr/lib/x86_64-linux-gnu/libnccl.so.2"

# Check if system NCCL has the required symbol
NCCL_LIB=""
if [ -f "$SYSTEM_NCCL" ]; then
    if nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then
        echo "System NCCL has required symbol, using system NCCL"
        NCCL_LIB="$SYSTEM_NCCL"
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    elif [ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 1> /dev/null 2>&1; then
        echo "System NCCL missing symbol, trying PyTorch bundled NCCL"
        NCCL_LIB=$(ls "$TORCH_LIB"/libnccl.so* 2>/dev/null | head -1)
        export LD_LIBRARY_PATH=$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    else
        echo "Warning: No suitable NCCL found, using system NCCL anyway"
        NCCL_LIB="$SYSTEM_NCCL"
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    fi
else
    echo "System NCCL not found, using PyTorch bundled NCCL if available"
    if [ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 1> /dev/null 2>&1; then
        NCCL_LIB=$(ls "$TORCH_LIB"/libnccl*.so* 2>/dev/null | head -1)
    fi
    export LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Use LD_PRELOAD to force loading correct NCCL library before PyTorch loads
if [ -n "$NCCL_LIB" ] && [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD="$NCCL_LIB:$LD_PRELOAD"
    echo "Using LD_PRELOAD to force NCCL: $NCCL_LIB"
fi

# Set NCCL environment variables for single GPU inference
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

exec python -m gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 0 --log-config gunicorn_logging.conf app.main:app -k uvicorn.workers.UvicornWorker "$@"
EOF

EXPOSE 8000

# Use startup script to ensure PyTorch's NCCL libraries are prioritized
ENTRYPOINT ["/usr/local/bin/start.sh"]
