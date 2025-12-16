FROM nvidia/cuda:13.0.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:/usr/bin:/bin:$PATH

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
COPY web_interface web_interface/
COPY app/gunicorn_logging.conf .
COPY patch_whisperx.py /tmp/
COPY patch_diarize.py /tmp/

# Install Python dependencies using UV with pyproject.toml
# UV automatically selects CUDA 12.8 wheels on Linux
# Install PyTorch nightly builds for RTX 5090 (sm_120) and other latest GPU support
# Use uv pip install --system to install packages to system Python
# Install all PyTorch packages from nightly to ensure version compatibility
# pyannote.audio compatibility: force upgrade to 4.0.1+ which removes AudioMetaData dependency
RUN uv pip install --system -e . \
    && uv pip install --system ctranslate2==4.6.0 \
    && echo "Explicitly installing gunicorn and uvicorn with pip3 to ensure they're in system Python..." \
    && pip3 install --no-cache-dir --force-reinstall "gunicorn==23.0.0" "uvicorn==0.38.0" \
    && echo "Verifying gunicorn installation..." \
    && python3 -c "import gunicorn; print(f'✓ gunicorn={gunicorn.__version__}')" || (echo "ERROR: gunicorn still not found!" && python3 -c "import sys; print('Python paths:', sys.path)" && exit 1) \
    && echo "Installing PyTorch nightly builds for latest GPU support (including RTX 5090 sm_120)..." \
    && uv pip uninstall --system -y torch torchvision torchaudio || true \
    && uv pip install --system --pre --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && echo "Applying torchvision compatibility patch..." \
    && python3 -c "import torch; exec('try:\n    @torch.library.register_fake(\"torchvision::nms\")\n    def nms_fake(boxes, scores, iou_threshold):\n        return torch.tensor([], dtype=torch.long)\nexcept Exception as e:\n    print(f\"Patch failed: {e}\")\n    pass')" || true \
    && find /usr/local/lib/python3.*/dist-packages/torchvision -name "_meta_registrations.py" -exec sed -i 's/@torch.library.register_fake("torchvision::nms")/# @torch.library.register_fake("torchvision::nms")  # Patched for compatibility/' {} \; || true \
    && echo "Force upgrading pyannote.audio and numpy for torchaudio compatibility..." \
    && uv pip uninstall --system -y pyannote.audio pyannote.core pyannote.metrics pyannote.pipeline pyannote.database || true \
    && uv pip install --system --upgrade --force-reinstall --no-cache-dir "numpy>=2.3" "pyannote.audio>=4.0.1" \
    && echo "Patching whisperx to use 'token' instead of 'use_auth_token' for pyannote.audio>=4.0.1..." \
    && python3 /tmp/patch_whisperx.py 2>&1 || echo "Warning: patch script had errors, continuing..." \
    && find /usr/local/lib/python3.*/dist-packages/whisperx -name "*.py" -type f -exec sed -i 's/\buse_auth_token\b/token/g' {} \; 2>/dev/null || true \
    && find /usr/local/lib/python3.*/site-packages/whisperx -name "*.py" -type f -exec sed -i 's/\buse_auth_token\b/token/g' {} \; 2>/dev/null || true \
    && echo "Patching whisperx diarize.py for pyannote.audio>=4.0.1 DiarizeOutput API compatibility..." \
    && python3 /tmp/patch_diarize.py 2>&1 || echo "Warning: diarize patch script had errors, continuing..." \
    && rm -f /tmp/patch_diarize.py \
    && rm -f /tmp/patch_whisperx.py \
    && echo "Fixing huggingface-hub version compatibility..." \
    && uv pip install --system --no-cache-dir "huggingface-hub>=0.34.0,<1.0" \
    && echo "Final verification: ensuring gunicorn is available..." \
    && python3 -c "import gunicorn; print(f'✓ Final check: gunicorn={gunicorn.__version__}')" || (echo "ERROR: gunicorn still not found! Installing with pip3..." && pip3 install --no-cache-dir --force-reinstall "gunicorn==23.0.0" && python3 -c "import gunicorn; print(f'✓ gunicorn={gunicorn.__version__}')")

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "0", "--log-config", "gunicorn_logging.conf", "--log-level", "info", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
