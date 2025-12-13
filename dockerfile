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
# Install all PyTorch packages from nightly to ensure version compatibility
# pyannote.audio compatibility: force upgrade to 4.0.1+ which removes AudioMetaData dependency
RUN uv pip install --system -e . \
    && uv pip install --system ctranslate2==4.6.0 \
    && echo "Installing PyTorch nightly builds for latest GPU support (including RTX 5090 sm_120)..." \
    && uv pip uninstall --system -y torch torchvision torchaudio || true \
    && echo "Installing all PyTorch packages from nightly to ensure version compatibility..." \
    && uv pip install --system --pre --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && echo "Applying torchvision compatibility patch..." \
    && python3 -c "import torch; exec('try:\n    @torch.library.register_fake(\"torchvision::nms\")\n    def nms_fake(boxes, scores, iou_threshold):\n        return torch.tensor([], dtype=torch.long)\nexcept Exception as e:\n    print(f\"Patch failed: {e}\")\n    pass')" || true \
    && find /usr/local/lib/python3.*/dist-packages/torchvision -name "_meta_registrations.py" -exec sed -i 's/@torch.library.register_fake("torchvision::nms")/# @torch.library.register_fake("torchvision::nms")  # Patched for compatibility/' {} \; || true \
    && echo "Force upgrading pyannote.audio and numpy for torchaudio compatibility..." \
    && uv pip uninstall --system -y pyannote.audio pyannote.core pyannote.metrics pyannote.pipeline pyannote.database || true \
    && uv pip install --system --upgrade --force-reinstall --no-cache-dir "numpy>=2.3" "pyannote.audio>=4.0.1" \
    && echo "Patching whisperx to use 'token' instead of 'use_auth_token' for pyannote.audio>=4.0.1..." \
    && python3 << 'EOF'
import re
import glob

# Find all whisperx Python files
whisperx_path = glob.glob('/usr/local/lib/python3.*/dist-packages/whisperx/**/*.py', recursive=True)

for filepath in whisperx_path:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for line in lines:
            original_line = line
            
            # Replace parameter definitions: use_auth_token= -> token=
            line = re.sub(r'\buse_auth_token\s*=', 'token=', line)
            
            # Replace in function calls: use_auth_token=variable -> token=variable
            line = re.sub(r'\buse_auth_token\s*=\s*(\w+)', r'token=\1', line)
            
            # Replace variable references when used as value: token=use_auth_token -> token=token
            # This handles the case where parameter name was changed but variable name wasn't
            line = re.sub(r'token\s*=\s*use_auth_token\b', 'token=token', line)
            
            # Replace standalone variable references (but be careful with context)
            # Only replace if it's clearly a variable (not in strings, comments, etc.)
            if 'use_auth_token' in line and not (line.strip().startswith('#') or '"use_auth_token"' in line or "'use_auth_token'" in line):
                # Replace use_auth_token as a variable name with token
                # But preserve it if it's part of a larger identifier
                line = re.sub(r'\buse_auth_token\b(?=\s*[,\)\]\}:]|\s*$)', 'token', line)
            
            if line != original_line:
                modified = True
            new_lines.append(line)
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        pass
EOF


    && echo "Fixing huggingface-hub version compatibility..." \
    && uv pip install --system --no-cache-dir "huggingface-hub>=0.34.0,<1.0" \
    && rm -rf /root/.cache /tmp/* /root/.uv /var/cache/* \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.pyo' -delete

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "0", "--log-config", "gunicorn_logging.conf", "--log-level", "info", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
