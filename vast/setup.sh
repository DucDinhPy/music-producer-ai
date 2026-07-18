#!/usr/bin/env bash
# =============================================================================
# vast/setup.sh  --  Run this ON a fresh Vast.ai instance (inside the container)
# =============================================================================
# What it does (idempotent, safe to re-run):
#   1. Sync latest CODE from your GitHub repo (data/venv/models are gitignored,
#      so they are left untouched by 'git reset --hard').
#   2. Download base models from HuggingFace (fast on Vast's network).
#   3. Recreate the 'xl_sft' symlink (the training CLI variant map lacks XL).
#   4. Create the directory tree that push_data.sh / training expect.
#
# Usage:
#   REPO_URL=https://github.com/<you>/<repo>.git bash vast/setup.sh
#   (or edit the default REPO_URL below once)
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/DucDinhPy/music-producer-ai.git}"
BRANCH="${BRANCH:-main}"
WORK=/workspace

cd "$WORK"

echo "==> [1/4] Sync code from $REPO_URL ($BRANCH)"
if [ ! -d .git ]; then
    git init -q
    git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
fi
git fetch --depth 1 origin "$BRANCH"
# reset ONLY tracked files; ignored data/venv/checkpoints/output are untouched
git reset --hard "origin/$BRANCH"

echo "==> [2/4] Download base models (skips if already present)"
uv run acestep-download --model acestep-v15-base
uv run acestep-download --model acestep-v15-xl-sft
uv run acestep-download --model acestep-5Hz-lm-1.7B

echo "==> [3/4] Recreate xl_sft symlink"
cd "$WORK/checkpoints"
ln -sf acestep-v15-xl-sft xl_sft
cd "$WORK"

echo "==> [4/4] Ensure data/output directories"
mkdir -p "$WORK/output"
mkdir -p "$WORK/datasets/vinahouse/phase_b_dataset_v2/tensors"

echo ""
echo "[setup] Done. Next, from your LOCAL machine run:"
echo "    bash vast/push_data.sh <IP> <PORT>          # upload tensors"
echo "    bash vast/push_data.sh <IP> <PORT> resume   # + upload outputs to resume"
