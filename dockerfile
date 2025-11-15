FROM nvidia/cuda:13.0.1-base-ubuntu22.04

# Build argument to choose PyTorch version (stable or nightly for RTX 5090)
ARG USE_PYTORCH_NIGHTLY=false

ENV PYTHON_VERSION=3.11
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install dependencies and clean up in the same layer
# Fix dpkg configuration issues by running configure first
# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && dpkg --configure -a || true \
    && apt-get -y install --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libcudnn9-cuda-12 \
    libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
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
# Use uv pip install to install project and dependencies directly to system Python
# Note: uv pip install . will install the project with all dependencies from pyproject.toml
RUN uv pip install --system . \
    && uv pip install --system ctranslate2==4.6.0

# Install PyTorch - use nightly for RTX 5090 support if requested
# Important: Only upgrade torch and torchvision, keep torchaudio version from pyproject.toml
# to maintain compatibility with pyannote.audio which requires torchaudio.AudioMetaData
RUN if [ "$USE_PYTORCH_NIGHTLY" = "true" ]; then \
        echo "Installing PyTorch nightly for RTX 5090 support..." \
        && uv pip uninstall --system -y torch torchvision || true \
        && uv pip install --system --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision; \
    else \
        echo "Upgrading PyTorch to latest stable version (keeping torchaudio from dependencies)..." \
        && uv pip uninstall --system -y torch torchvision || true \
        && uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch torchvision; \
    fi \
    && python -c "import torch; import torchaudio; print(f'PyTorch: {torch.__version__}, torchaudio: {torchaudio.__version__}')" \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

EXPOSE 8000

# Use python -m gunicorn to ensure gunicorn is found in system Python
ENTRYPOINT ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "0", "--log-config", "gunicorn_logging.conf", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
