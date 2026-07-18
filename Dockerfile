# =============================================================================
# ACE-Step 1.5 + Vinahouse LoRA training image for Vast.ai
# =============================================================================
# Design principles:
#   - Layer 1 (base): CUDA + Python + system deps      -> rarely changes
#   - Layer 2 (deps): uv + ACE-Step Python packages     -> changes weekly
#   - Layer 3 (code): repo code + custom scripts        -> changes daily
#   - Runtime: model checkpoints downloaded on first run (NOT baked in)
# =============================================================================

# ---------------------------------------------------------------------------
# Layer 1: CUDA base with Python 3.11
# ---------------------------------------------------------------------------
# Use CUDA 12.4 runtime (matches PyTorch 2.4+ requirements).
# 'devel' variant includes nvcc for building bitsandbytes/flash-attn wheels.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# System packages layer 1: minimal + software-properties for adding PPAs
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Layer 2: Python dependencies (leverages Docker layer caching)
# ---------------------------------------------------------------------------
# Copy pyproject.toml + uv.lock first so 'uv sync' cache is reused when
# only source code changes.
# Also copy vendored 'nano-vllm' because pyproject.toml has a local path
# dependency to it (acestep/third_parts/nano-vllm/). Without this, uv sync
# fails with "Distribution not found".
COPY pyproject.toml uv.lock ./
COPY acestep/third_parts/nano-vllm ./acestep/third_parts/nano-vllm

# Install project deps into a .venv (matches local dev environment)
RUN uv sync --frozen --no-install-project

# ---------------------------------------------------------------------------
# Layer 3: Project code + scripts + dataset
# ---------------------------------------------------------------------------
# Copy everything (respecting .dockerignore).
# This will overwrite the earlier pyproject.toml/uv.lock (same content, no-op).
COPY . .

# Install ace-step from source (was skipped earlier with --no-install-project)
RUN uv sync --frozen

# Formally add training extras via 'uv add' so they:
#   1) get pinned into pyproject.toml + uv.lock
#   2) survive future 'uv run' auto-sync (which purges packages not in lock)
# This is the KEY fix - 'uv pip install' would be purged by the next auto-sync.
RUN uv add --no-sync \
        "bitsandbytes>=0.45.0" \
        "librosa>=0.11.0" \
        "soundfile>=0.13.0" \
        "openai>=2.0.0" \
    && uv sync --frozen

# Make pipeline + vast helper scripts executable
RUN chmod +x pipeline/*.sh vast/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Environment defaults (override at runtime with -e)
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B \
    ACESTEP_LM_BACKEND=pt \
    ACESTEP_NO_INIT=true \
    HF_HOME=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache

# Expose Gradio UI port (in case you launch UI on Vast.ai)
EXPOSE 7860

# Default command: interactive bash so you can inspect/run manually
# Override with 'docker run ... bash datasets/vinahouse/scripts/train_lokr_pilot.sh'
CMD ["/bin/bash"]
