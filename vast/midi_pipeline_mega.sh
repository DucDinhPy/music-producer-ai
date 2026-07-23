#!/usr/bin/env bash
# =============================================================================
# vast/midi_pipeline_mega.sh  --  Run this ON a Vast.ai instance
# =============================================================================
# What it does:
#   1. Installs system tools: git, ffmpeg, Python venv tools, MEGA tools.
#   2. Creates a separate Python venv for the MIDI/stem pipeline.
#   3. Downloads/installs BS-RoFormer tooling and Muscriptor.
#   4. Downloads your Vinahouse audio dataset from MEGA.
#   5. Runs YOUR processing script:
#          audio -> BS-RoFormer stems -> Muscriptor MIDI -> output folder
#   6. Uploads the output folder back to your MEGA account.
#
# Usage on Vast.ai:
#   cd /workspace
#   git pull
#   bash vast/midi_pipeline_mega.sh
#
# Recommended: keep secrets out of git by exporting env vars before running:
#   export MEGA_EMAIL="your@email.com"
#   export MEGA_PASSWORD="your-mega-password"
#   export MEGA_DATA_URL="https://mega.nz/folder/..."
#   export PROCESS_SCRIPT="/workspace/scripts/run_bs_roformer_muscriptor_batch.sh"
#   bash vast/midi_pipeline_mega.sh
#
# Optional: create vast/midi_pipeline.env on the Vast machine only. Do NOT commit it.
# =============================================================================
set -euo pipefail

# ============================ CONFIG (edit me) ===============================
# Workspace/repo layout on Vast.ai.
WORK="${WORK:-/workspace}"
TOOLS_DIR="${TOOLS_DIR:-$WORK/tools}"
DATA_DIR="${DATA_DIR:-$WORK/datasets/vinahouse/mega_audio}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/output/midi_pipeline}"

# Your script. You will write this file yourself.
# It will receive DATA_DIR, OUTPUT_DIR, TOOLS_DIR, BS_ROFORMER_DIR, MUSCRIPTOR_DIR
# as environment variables.
PROCESS_SCRIPT="${PROCESS_SCRIPT:-$WORK/scripts/run_bs_roformer_muscriptor_batch.sh}"

# MEGA download.
# DOWNLOAD_MODE=public  -> use MEGA_DATA_URL with MEGAcmd's mega-get.
# DOWNLOAD_MODE=account -> use MEGA_DOWNLOAD_REMOTE_DIR from your MEGA account.
DOWNLOAD_MODE="${DOWNLOAD_MODE:-public}"
MEGA_DATA_URL="${MEGA_DATA_URL:-}"                    # Example: https://mega.nz/folder/xxxx#yyyy
MEGA_DOWNLOAD_REMOTE_DIR="${MEGA_DOWNLOAD_REMOTE_DIR:-/Root/vinahouse_audio}"

# MEGA upload requires an account.
# Prefer exporting MEGA_EMAIL and MEGA_PASSWORD instead of editing them here.
MEGA_EMAIL="${MEGA_EMAIL:-}"
MEGA_PASSWORD="${MEGA_PASSWORD:-}"
MEGA_UPLOAD_REMOTE_DIR="${MEGA_UPLOAD_REMOTE_DIR:-/Root/vinahouse_midi_pipeline_output}"

# BS-RoFormer install.
# package: install bs-roformer-infer from pip.
# repo: clone BS_ROFORMER_REPO_URL and pip install it.
BS_ROFORMER_INSTALL_MODE="${BS_ROFORMER_INSTALL_MODE:-package}"
BS_ROFORMER_REPO_URL="${BS_ROFORMER_REPO_URL:-https://github.com/lucidrains/BS-RoFormer.git}"
BS_ROFORMER_DIR="${BS_ROFORMER_DIR:-$TOOLS_DIR/BS-RoFormer}"

# Muscriptor install. This matches the command you used manually:
#   python -m pip install git+https://github.com/muscriptor/muscriptor.git
MUSCRIPTOR_PIP_SPEC="${MUSCRIPTOR_PIP_SPEC:-git+https://github.com/muscriptor/muscriptor.git}"
MUSCRIPTOR_DIR="${MUSCRIPTOR_DIR:-}"

# Python environment for this pipeline. Separate from ACE-Step's env.
VENV_DIR="${VENV_DIR:-$WORK/.venv-midi-pipeline}"

# Vast images can have CUDA 12.x drivers. Installing an incompatible newer
# PyTorch wheel (for example cu130) makes torch.cuda unavailable.
INSTALL_TORCH_CUDA="${INSTALL_TORCH_CUDA:-true}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
# =============================================================================

if [ -f "$WORK/vast/midi_pipeline.env" ]; then
    # Use this for local Vast-only config/secrets. Do not commit the env file.
    # shellcheck disable=SC1091
    source "$WORK/vast/midi_pipeline.env"
fi

echo "==> [0/7] Validate config"
if [ "$DOWNLOAD_MODE" = "public" ] && [ -z "$MEGA_DATA_URL" ]; then
    echo "[error] DOWNLOAD_MODE=public requires MEGA_DATA_URL."
    exit 1
fi

if [ "$DOWNLOAD_MODE" = "account" ] && { [ -z "$MEGA_EMAIL" ] || [ -z "$MEGA_PASSWORD" ]; }; then
    echo "[error] DOWNLOAD_MODE=account requires MEGA_EMAIL and MEGA_PASSWORD."
    exit 1
fi

if [ -z "$MEGA_EMAIL" ] || [ -z "$MEGA_PASSWORD" ]; then
    echo "[error] Upload requires MEGA_EMAIL and MEGA_PASSWORD."
    exit 1
fi

cd "$WORK"
mkdir -p "$TOOLS_DIR" "$DATA_DIR" "$OUTPUT_DIR"

echo "==> [1/7] Install system tools"
if ! command -v git >/dev/null 2>&1 || \
   ! command -v ffmpeg >/dev/null 2>&1 || \
   ! command -v megadl >/dev/null 2>&1 || \
   ! command -v mega-get >/dev/null 2>&1 || \
   ! command -v python3 >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        megatools \
        python3 \
        python3-pip \
        python3-venv \
        rsync \
        unzip \
        wget
fi

if ! command -v mega-get >/dev/null 2>&1; then
    echo "==> Install MEGAcmd for public MEGA folder downloads"
    if ! apt-get install -y -qq megacmd; then
        . /etc/os-release
        MEGACMD_UBUNTU_VERSION="${MEGACMD_UBUNTU_VERSION:-$VERSION_ID}"
        MEGACMD_DEB_URL="${MEGACMD_DEB_URL:-https://mega.nz/linux/repo/xUbuntu_${MEGACMD_UBUNTU_VERSION}/amd64/megacmd-xUbuntu_${MEGACMD_UBUNTU_VERSION}_amd64.deb}"
        echo "    Downloading: $MEGACMD_DEB_URL"
        wget -q -O /tmp/megacmd.deb "$MEGACMD_DEB_URL"
        apt-get install -y -qq /tmp/megacmd.deb
    fi
fi

echo "==> [2/7] Create Python venv: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ "$INSTALL_TORCH_CUDA" = "true" ]; then
    echo "==> Install PyTorch CUDA wheel from $PYTORCH_CUDA_INDEX_URL"
    python -m pip install --upgrade \
        --index-url "$PYTORCH_CUDA_INDEX_URL" \
        torch torchvision torchaudio
fi

echo "==> [3/7] Install BS-RoFormer tooling"
if [ "$BS_ROFORMER_INSTALL_MODE" = "package" ]; then
    # Inference-focused package. Your PROCESS_SCRIPT can call its CLI/API.
    python -m pip install --upgrade bs-roformer-infer
elif [ "$BS_ROFORMER_INSTALL_MODE" = "repo" ]; then
    if [ ! -d "$BS_ROFORMER_DIR/.git" ]; then
        git clone "$BS_ROFORMER_REPO_URL" "$BS_ROFORMER_DIR"
    else
        git -C "$BS_ROFORMER_DIR" pull --ff-only
    fi
    python -m pip install -e "$BS_ROFORMER_DIR"
else
    echo "[error] Invalid BS_ROFORMER_INSTALL_MODE: $BS_ROFORMER_INSTALL_MODE"
    echo "        Use: package or repo"
    exit 1
fi

echo "==> [4/7] Download/install Muscriptor"
python -m pip install --upgrade "$MUSCRIPTOR_PIP_SPEC"

echo "==> [5/7] Download dataset from MEGA -> $DATA_DIR"
if [ "$DOWNLOAD_MODE" = "public" ]; then
    # Public MEGA folder/file link.
    # MEGAcmd handles modern links such as:
    #   https://mega.nz/folder/<id>#<key>
    mega-get "$MEGA_DATA_URL" "$DATA_DIR"
elif [ "$DOWNLOAD_MODE" = "account" ]; then
    # Private folder inside your MEGA account.
    megacopy \
        --download \
        --local "$DATA_DIR" \
        --remote "$MEGA_DOWNLOAD_REMOTE_DIR" \
        --username "$MEGA_EMAIL" \
        --password "$MEGA_PASSWORD"
fi

echo "==> [6/7] Run your processing script"
echo "    PROCESS_SCRIPT=$PROCESS_SCRIPT"
echo "    DATA_DIR=$DATA_DIR"
echo "    OUTPUT_DIR=$OUTPUT_DIR"
echo "    BS_ROFORMER_DIR=$BS_ROFORMER_DIR"
echo "    MUSCRIPTOR_DIR=$MUSCRIPTOR_DIR"

if [ ! -f "$PROCESS_SCRIPT" ]; then
    echo "[error] PROCESS_SCRIPT does not exist yet."
    echo "        Create it, then re-run this script."
    exit 1
fi

DATA_DIR="$DATA_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
TOOLS_DIR="$TOOLS_DIR" \
BS_ROFORMER_DIR="$BS_ROFORMER_DIR" \
MUSCRIPTOR_DIR="$MUSCRIPTOR_DIR" \
bash "$PROCESS_SCRIPT"

echo "==> [7/7] Upload output back to MEGA"
echo "    Remote: $MEGA_UPLOAD_REMOTE_DIR"
megamkdir \
    --username "$MEGA_EMAIL" \
    --password "$MEGA_PASSWORD" \
    "$MEGA_UPLOAD_REMOTE_DIR" >/dev/null 2>&1 || true

megacopy \
    --local "$OUTPUT_DIR" \
    --remote "$MEGA_UPLOAD_REMOTE_DIR" \
    --username "$MEGA_EMAIL" \
    --password "$MEGA_PASSWORD"

echo ""
echo "[midi_pipeline_mega] Done."
echo "Local output on Vast: $OUTPUT_DIR"
echo "MEGA output folder:   $MEGA_UPLOAD_REMOTE_DIR"
