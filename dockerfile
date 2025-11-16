# Use CUDA runtime image which includes more complete library set including newer NCCL
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# Build argument to choose PyTorch version (stable or nightly for RTX 5090)
ARG USE_PYTORCH_NIGHTLY=false

ENV PYTHON_VERSION=3.11
# LD_LIBRARY_PATH will be set after NCCL installation to ensure correct library is found
# System NCCL (if version 2.18+) takes priority, then PyTorch bundled libraries
# Include CUDA library paths for torchaudio compatibility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/lib
# LD_PRELOAD will be set by startup script based on build-time detection

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
# First install all dependencies except torch/torchvision/torchaudio
RUN uv pip install --system . \
    && uv pip install --system ctranslate2==4.6.0

# Install PyTorch first - use nightly for RTX 5090 support if requested
# Install torch and torchvision together to ensure compatibility
# CRITICAL: PyTorch 2.8 requires NCCL 2.18+ for ncclGroupSimulateEnd symbol
# CRITICAL: torchaudio 2.3.1 is NOT compatible with PyTorch 2.8+ due to ABI changes
# Solution: Use PyTorch 2.7 which may still support RTX 5090 and is compatible with torchaudio 2.3.1
# OR: Use PyTorch 2.8+ with matching torchaudio version (but pyannote.audio needs AudioMetaData)
RUN if [ "$USE_PYTORCH_NIGHTLY" = "true" ]; then \
        echo "Installing PyTorch nightly for RTX 5090 support..." \
        && uv pip uninstall --system torch torchvision torchaudio || true \
        && uv pip install --system --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision \
        && echo "Verifying torch and torchvision compatibility..." \
        && python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}'); print('✓ torch and torchvision are compatible')"; \
    else \
        echo "Installing PyTorch for RTX 5090 support..." \
        && echo "Note: Will try PyTorch 2.3.0 first (compatible with torchaudio 2.3.1, may support RTX 5090)" \
        && uv pip uninstall --system torch torchvision torchaudio || true \
        && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torch==2.3.0" "torchvision==0.18.0" || \
            echo "PyTorch 2.3.0 not available, trying 2.4.0..." \
            && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torch==2.4.0" "torchvision==0.19.0" || \
                echo "PyTorch 2.4.0 not available, trying 2.5.0..." \
                && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torch==2.5.0" "torchvision==0.20.0" || \
                    echo "PyTorch 2.5.0 not available, trying 2.6.0..." \
                    && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torch==2.6.0" "torchvision==0.21.0" || \
                        echo "Specific versions not available, installing torch first, then matching torchvision..." \
                        && uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch \
                        && TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "") \
                        && echo "Installed torch version: $TORCH_VER" \
                        && if [ -n "$TORCH_VER" ]; then \
                            TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1) \
                            && TORCH_MINOR=$(echo "$TORCH_VER" | cut -d. -f2) \
                            && TORCHVISION_VER="0.$((TORCH_MINOR + 15)).0" \
                            && echo "Installing matching torchvision version: $TORCHVISION_VER for torch $TORCH_MAJOR.$TORCH_MINOR" \
                            && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torchvision==$TORCHVISION_VER" || \
                                echo "torchvision $TORCHVISION_VER not available, trying alternative versions..." \
                                && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.$((TORCH_MINOR + 14)).0" || \
                                    uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.$((TORCH_MINOR + 16)).0" || \
                                    echo "WARNING: Could not install matching torchvision version, installing latest..." \
                                    && uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torchvision)); \
                        else \
                            echo "ERROR: Could not determine torch version" \
                            && uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torchvision; \
                        fi)))) \
        && echo "Verifying torch and torchvision compatibility..." \
        && python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}'); print('Testing torchvision import...'); from torchvision import transforms; print('✓ torch and torchvision are compatible')" || \
            (echo "ERROR: torch and torchvision are not compatible. Attempting to fix..." \
            && TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "") \
            && echo "Detected torch version: $TORCH_VER" \
            && if [ -n "$TORCH_VER" ]; then \
                TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1) \
                && TORCH_MINOR=$(echo "$TORCH_VER" | cut -d. -f2) \
                && echo "Installing matching torchvision for torch $TORCH_MAJOR.$TORCH_MINOR..." \
                && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps "torchvision==0.$((TORCH_MINOR + 15)).0" || \
                    uv pip install --system --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps "torchvision==0.$((TORCH_MINOR + 14)).0" || \
                    uv pip install --system --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps "torchvision==0.$((TORCH_MINOR + 16)).0" || \
                    echo "WARNING: Could not install matching torchvision version"); \
            fi \
            && echo "Re-verifying compatibility..." \
            && python -c "import torch; import torchvision; from torchvision import transforms; print('✓ torch and torchvision are now compatible')" || \
            (echo "ERROR: Still incompatible. This may cause runtime errors." && exit 1)); \
    fi \
    && echo "PyTorch installed, version: $(python -c 'import torch; print(torch.__version__)')" \
    && python -c "import torch; import os; torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'PyTorch lib path: {torch_lib}'); import glob; nccl_libs = glob.glob(os.path.join(torch_lib, '*nccl*')); print(f'NCCL libraries in PyTorch: {nccl_libs}')" \
    && ldconfig

# Fix torchaudio compatibility with pyannote.audio
# pyannote.audio requires torchaudio with AudioMetaData (available in torchaudio < 2.4)
# CRITICAL: torchaudio 2.3.1 is NOT compatible with PyTorch 2.8+ due to ABI changes
# Try to install torchaudio version that matches PyTorch version
# If PyTorch 2.8+, we need to use matching torchaudio version (but it may not have AudioMetaData)
RUN echo "Fixing torchaudio compatibility for pyannote.audio..." \
    && PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown") \
    && echo "Detected PyTorch version: $PYTORCH_VERSION" \
    && uv pip uninstall --system torchaudio || true \
    && echo "Installing torchaudio matching PyTorch version..." \
    && echo "NOTE: If PyTorch 2.8+, torchaudio 2.3.1 may not be compatible. Will try matching version first." \
    && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torchaudio || \
        echo "WARNING: Could not install matching torchaudio, trying 2.3.1..." \
        && (uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "torchaudio==2.3.1+cu128" || \
            uv pip install --system --index-url https://download.pytorch.org/whl/cu121 "torchaudio==2.3.1+cu121" || \
            uv pip install --system --index-url https://download.pytorch.org/whl/cu118 "torchaudio==2.3.1+cu118" || \
            echo "ERROR: Could not install torchaudio" && exit 1)) \
    && python -c "import torch; import torchaudio; print(f'PyTorch: {torch.__version__}, torchaudio: {torchaudio.__version__}')" \
    && (python -c "import torchaudio; assert hasattr(torchaudio, 'AudioMetaData'), 'AudioMetaData not found in torchaudio'; print('✓ AudioMetaData available')" || echo "WARNING: AudioMetaData not found - this may cause pyannote.audio to fail. Consider using PyTorch 2.7 or updating pyannote.audio.") \
    && echo "Determining correct NCCL library to use..." \
    && TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib" \
    && ARCH=$(uname -m) \
    && SYSTEM_NCCL="/usr/lib/${ARCH}-linux-gnu/libnccl.so.2" \
    && NCCL_TO_USE="" \
    && echo "Architecture: $ARCH" \
    && echo "Checking system NCCL at $SYSTEM_NCCL..." \
    && if [ -f "$SYSTEM_NCCL" ]; then \
        if nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then \
            echo "✓ System NCCL has required symbol" \
            && NCCL_TO_USE="$SYSTEM_NCCL" \
            && echo "$SYSTEM_NCCL" > /etc/pytorch_nccl_lib.txt; \
        else \
            echo "✗ System NCCL missing symbol"; \
        fi; \
    else \
        echo "✗ System NCCL not found at $SYSTEM_NCCL"; \
    fi \
    && if [ -z "$NCCL_TO_USE" ] && [ -d "$TORCH_LIB" ]; then \
        echo "Checking PyTorch bundled NCCL..." \
        && for nccl_file in "$TORCH_LIB"/libnccl*.so*; do \
            if [ -f "$nccl_file" ]; then \
                echo "Found PyTorch NCCL: $nccl_file" \
                && if nm -D "$nccl_file" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then \
                    echo "✓ PyTorch bundled NCCL has required symbol" \
                    && NCCL_TO_USE="$nccl_file" \
                    && echo "$nccl_file" > /etc/pytorch_nccl_lib.txt \
                    && break; \
                else \
                    echo "✗ PyTorch bundled NCCL missing symbol: $nccl_file"; \
                fi; \
            fi; \
        done; \
    fi \
    && if [ -z "$NCCL_TO_USE" ]; then \
        echo "ERROR: No compatible NCCL found with ncclGroupSimulateEnd symbol!" \
        && echo "System NCCL: $([ -f "$SYSTEM_NCCL" ] && echo "exists at $SYSTEM_NCCL" || echo "not found")" \
        && if [ -f "$SYSTEM_NCCL" ]; then \
            echo "System NCCL version check:" \
            && dpkg -l | grep libnccl2 || echo "NCCL not in dpkg" \
            && echo "System NCCL symbols:" \
            && nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -i "ncclGroup" | head -5 || echo "No ncclGroup symbols found"; \
        fi \
        && echo "PyTorch NCCL: $([ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 2>/dev/null | head -1 || echo "not found")" \
        && if [ -d "$TORCH_LIB" ]; then \
            echo "PyTorch NCCL files:" \
            && ls -la "$TORCH_LIB"/libnccl* 2>/dev/null || echo "No NCCL files in PyTorch lib"; \
        fi \
        && echo "" \
        && echo "CRITICAL: PyTorch 2.8 requires NCCL 2.18+ with ncclGroupSimulateEnd symbol" \
        && echo "Possible solutions:" \
        && echo "1. Use PyTorch 2.7 (if compatible with RTX 5090)" \
        && echo "2. Use conda to install PyTorch (includes complete dependencies)" \
        && echo "3. Manually install NCCL 2.19+ from NVIDIA" \
        && exit 1; \
    fi \
    && echo "NCCL library to use: $(cat /etc/pytorch_nccl_lib.txt)" \
    && echo "Verifying NCCL symbol in selected library..." \
    && if ! nm -D "$(cat /etc/pytorch_nccl_lib.txt)" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then \
        echo "ERROR: Selected NCCL library does not contain ncclGroupSimulateEnd symbol!" \
        && exit 1; \
    fi \
    && echo "✓ NCCL symbol verified successfully" \
    && echo "Testing PyTorch import at build time..." \
    && NCCL_LIB_PATH="$(cat /etc/pytorch_nccl_lib.txt)" \
    && NCCL_DIR="$(dirname "$NCCL_LIB_PATH")" \
    && if ! env LD_PRELOAD="$NCCL_LIB_PATH" LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" python -c "import torch; print(f'✓ PyTorch {torch.__version__} imported successfully at build time')" 2>&1; then \
        echo "ERROR: PyTorch import failed at build time even with correct NCCL!" \
        && echo "This indicates a deeper compatibility issue." \
        && exit 1; \
    fi \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

# Create startup script that sets environment variables before starting gunicorn
# Intelligently handle NCCL library loading to avoid symbol errors
# Use LD_PRELOAD to force loading correct NCCL library if needed
RUN cat > /usr/local/bin/start.sh << 'EOF' && chmod +x /usr/local/bin/start.sh
#!/bin/bash
# Don't use set -e initially, we want to see all errors
set +e

# Force all output to stderr (which gunicorn logs capture)
exec 1>&2

# CRITICAL: Force output immediately to verify script execution
echo "========================================" >&2
echo "=== WhisperX Service Startup Script ===" >&2
echo "========================================" >&2
echo "Starting at $(date)" >&2
echo "Script PID: $$" >&2
echo "Current user: $(whoami)" >&2
echo "Script path: $0" >&2
echo "Arguments: $@" >&2

# Now enable error handling
set -e

# Update library cache
ldconfig

# Initialize variables
NCCL_LIB=""
TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib"
ARCH=$(uname -m)
SYSTEM_NCCL="/usr/lib/${ARCH}-linux-gnu/libnccl.so.2"

echo "Architecture: $ARCH" >&2
echo "System NCCL path: $SYSTEM_NCCL" >&2
echo "PyTorch lib path: $TORCH_LIB" >&2
echo "Checking for build-time NCCL file: /etc/pytorch_nccl_lib.txt" >&2

# Try to use NCCL library determined at build time (preferred method)
if [ -f /etc/pytorch_nccl_lib.txt ]; then
    NCCL_LIB=$(cat /etc/pytorch_nccl_lib.txt)
    if [ -f "$NCCL_LIB" ]; then
        echo "Using build-time determined NCCL library: $NCCL_LIB" >&2
        # Set LD_PRELOAD before any Python process starts
        export LD_PRELOAD="$NCCL_LIB"
        NCCL_DIR=$(dirname "$NCCL_LIB")
        TORCH_LIB="/usr/local/lib/python3.11/dist-packages/torch/lib"
        export LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH"
        
        # Verify the symbol is present
        echo "Verifying NCCL symbol..." >&2
        if ! nm -D "$NCCL_LIB" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then
            echo "ERROR: Selected NCCL library does not contain ncclGroupSimulateEnd symbol!" >&2
            echo "Library: $NCCL_LIB" >&2
            exit 1
        fi
        echo "✓ NCCL symbol verified" >&2
        
        # Set NCCL environment variables for single GPU inference
        export NCCL_P2P_DISABLE=1
        export NCCL_SHM_DISABLE=1
        export NCCL_IB_DISABLE=1
        
        # Test PyTorch import with LD_PRELOAD set before starting gunicorn
        echo "Testing PyTorch import with build-time NCCL..." >&2
        echo "LD_PRELOAD: $LD_PRELOAD" >&2
        echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
        if ! python -c "import torch; print(f'✓ PyTorch {torch.__version__} imported successfully')" 2>&1; then
            echo "ERROR: PyTorch import failed with build-time NCCL!" >&2
            exit 1
        fi
        
        # CRITICAL: Set LD_PRELOAD in the environment before starting gunicorn
        # This ensures it's inherited by all child processes including worker processes
        echo "Starting gunicorn with LD_PRELOAD=$LD_PRELOAD" >&2
        echo "Environment variables:" >&2
        echo "  LD_PRELOAD=$LD_PRELOAD" >&2
        echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
        echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE" >&2
        echo "  NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE" >&2
        
        # Use exec to replace shell process with gunicorn
        # Environment variables are inherited by all child processes
        exec env LD_PRELOAD="$LD_PRELOAD" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" python -m gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 0 --log-config gunicorn_logging.conf app.main:app -k uvicorn.workers.UvicornWorker "$@"
    fi
fi

# Fallback to runtime detection if build-time file not found
echo "" >&2
echo "========================================" >&2
echo "Build-time NCCL file not found!" >&2
echo "File /etc/pytorch_nccl_lib.txt exists: $([ -f /etc/pytorch_nccl_lib.txt ] && echo 'YES' || echo 'NO')" >&2
echo "Performing runtime detection..." >&2
echo "========================================" >&2

# Check if system NCCL has the required symbol
if [ -f "$SYSTEM_NCCL" ]; then
    if nm -D "$SYSTEM_NCCL" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then
        echo "System NCCL has required symbol, using system NCCL" >&2
        NCCL_LIB="$SYSTEM_NCCL"
    elif [ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 1> /dev/null 2>&1; then
        echo "System NCCL missing symbol, trying PyTorch bundled NCCL" >&2
        NCCL_LIB=$(ls "$TORCH_LIB"/libnccl.so* 2>/dev/null | head -1)
    else
        echo "Warning: No suitable NCCL found, using system NCCL anyway" >&2
        NCCL_LIB="$SYSTEM_NCCL"
    fi
else
    echo "System NCCL not found, using PyTorch bundled NCCL if available" >&2
    if [ -d "$TORCH_LIB" ] && ls "$TORCH_LIB"/libnccl*.so* 1> /dev/null 2>&1; then
        NCCL_LIB=$(ls "$TORCH_LIB"/libnccl*.so* 2>/dev/null | head -1)
    fi
fi

# Use LD_PRELOAD to force loading correct NCCL library before PyTorch loads
if [ -n "$NCCL_LIB" ] && [ -f "$NCCL_LIB" ]; then
    export LD_PRELOAD="$NCCL_LIB"
    NCCL_DIR=$(dirname "$NCCL_LIB")
    export LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH"
    echo "Using LD_PRELOAD to force NCCL: $NCCL_LIB" >&2
    
    # Verify symbol
    if ! nm -D "$NCCL_LIB" 2>/dev/null | grep -q "ncclGroupSimulateEnd"; then
        echo "WARNING: NCCL library does not contain ncclGroupSimulateEnd symbol!" >&2
    else
        echo "✓ NCCL symbol verified in runtime-detected library" >&2
    fi
else
    echo "WARNING: No NCCL library found for LD_PRELOAD!" >&2
    echo "Attempting to find any NCCL library in PyTorch..." >&2
    if [ -d "$TORCH_LIB" ]; then
        # Try to find any NCCL library, even if symbol check fails
        FOUND_NCCL=$(find "$TORCH_LIB" -name "libnccl*.so*" -type f 2>/dev/null | head -1)
        if [ -n "$FOUND_NCCL" ] && [ -f "$FOUND_NCCL" ]; then
            echo "Found PyTorch NCCL library (unverified): $FOUND_NCCL" >&2
            NCCL_LIB="$FOUND_NCCL"
            export LD_PRELOAD="$NCCL_LIB"
            NCCL_DIR=$(dirname "$NCCL_LIB")
            export LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        fi
    fi
fi

# Final check: if still no NCCL_LIB, try system NCCL as last resort
if [ -z "$NCCL_LIB" ] && [ -f "$SYSTEM_NCCL" ]; then
    echo "Using system NCCL as last resort: $SYSTEM_NCCL" >&2
    NCCL_LIB="$SYSTEM_NCCL"
    export LD_PRELOAD="$NCCL_LIB"
    NCCL_DIR=$(dirname "$NCCL_LIB")
    export LD_LIBRARY_PATH="$NCCL_DIR:$TORCH_LIB:/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH"
fi

# Set NCCL environment variables for single GPU inference
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

# Test PyTorch import before starting gunicorn
echo "" >&2
echo "========================================" >&2
echo "=== Final Configuration ===" >&2
echo "========================================" >&2
echo "LD_PRELOAD: ${LD_PRELOAD:-NOT SET}" >&2
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE" >&2
echo "NCCL_SHM_DISABLE: $NCCL_SHM_DISABLE" >&2
echo "NCCL_LIB variable: ${NCCL_LIB:-NOT SET}" >&2
echo "" >&2

echo "Testing PyTorch import..." >&2
if ! python -c "import torch; print(f'✓ PyTorch {torch.__version__} imported successfully')" 2>&1; then
    echo "" >&2
    echo "========================================" >&2
    echo "ERROR: PyTorch import failed!" >&2
    echo "========================================" >&2
    echo "LD_PRELOAD: ${LD_PRELOAD:-NOT SET}" >&2
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
    echo "This usually means LD_PRELOAD is not set correctly or NCCL library is incompatible." >&2
    echo "" >&2
    exit 1
fi
echo "✓ PyTorch import test passed" >&2

# Start gunicorn with environment variables explicitly set
# CRITICAL: Use env to ensure LD_PRELOAD is passed to all child processes including workers
if [ -n "$LD_PRELOAD" ]; then
    echo "Starting gunicorn with LD_PRELOAD=$LD_PRELOAD" >&2
    # If arguments are provided, use them; otherwise use default gunicorn command
    if [ $# -gt 0 ]; then
        # User provided custom command, execute it with environment variables
        echo "Executing custom command: $@" >&2
        exec env LD_PRELOAD="$LD_PRELOAD" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" "$@"
    else
        # No arguments, use default gunicorn command
        exec env LD_PRELOAD="$LD_PRELOAD" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" python -m gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 0 --log-config gunicorn_logging.conf app.main:app -k uvicorn.workers.UvicornWorker
    fi
else
    echo "ERROR: LD_PRELOAD is not set! Cannot start gunicorn safely." >&2
    echo "Please check NCCL installation and ensure a compatible library is available." >&2
    exit 1
fi
EOF

EXPOSE 8000

# Use startup script to ensure PyTorch's NCCL libraries are prioritized
# Use shell form to ensure environment variables are properly set
# Also set as CMD as fallback in case ENTRYPOINT is overridden
ENTRYPOINT ["/bin/bash", "/usr/local/bin/start.sh"]
CMD []
