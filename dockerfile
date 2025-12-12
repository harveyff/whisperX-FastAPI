FROM nvidia/cuda:13.0.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install dependencies and clean up in the same layer
# hadolint ignore=DL3008
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends \
    python3.11=3.11.0~rc1-1~22.04 \
    python3-pip \
    curl \
    git \
    ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
    libcudnn9-cuda-12=9.8.0.87-1 \
    libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install UV for package management using official installer
# This avoids ghcr.io access issues (403 Forbidden)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx \
    && rm -rf /root/.local/bin

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY app app/
COPY tests tests/
COPY app/gunicorn_logging.conf .

# Install Python dependencies using UV with pyproject.toml
# UV automatically selects CUDA 12.8 wheels on Linux
# Install PyTorch nightly builds for RTX 5090 (sm_120) and other latest GPU support
# Use uv pip install --system to install packages to system Python
# Note: Install torch and torchvision from nightly, but keep torchaudio from stable to avoid compatibility issues
RUN uv pip install --system -e . \
    && uv pip install --system ctranslate2==4.6.0 \
    && echo "Installing PyTorch nightly builds for latest GPU support (including RTX 5090 sm_120)..." \
    && uv pip uninstall --system -y torch torchvision torchaudio || true \
    && uv pip install --system --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && uv pip install --system torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "0", "--log-config", "gunicorn_logging.conf", "--log-level", "info", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
