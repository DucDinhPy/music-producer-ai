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

echo "==> [1/5] Install system tools (nano, tmux)"
# Only run apt if any tool is missing (keeps re-runs fast).
if ! command -v nano >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq nano tmux || \
        echo "[warn] apt install failed; install nano/tmux manually if needed"
fi

echo "==> [2/5] Sync code from $REPO_URL ($BRANCH)"
git init -q                                   # no-op if already a git repo
# Always repoint origin to YOUR repo (the baked image ships upstream's origin)
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git fetch origin "$BRANCH"
# Sync working tree, then create a TRACKING branch so plain 'git pull' works
# forever after. Ignored data/venv/checkpoints/output are left untouched.
git reset --hard "origin/$BRANCH"
git checkout -B "$BRANCH" --track "origin/$BRANCH"

echo "==> [3/5] Download base models (skips if already present)"
uv run acestep-download --model acestep-v15-base
uv run acestep-download --model acestep-v15-xl-sft
uv run acestep-download --model acestep-5Hz-lm-4B

echo "==> [4/5] Recreate xl_sft symlink"
cd "$WORK/checkpoints"
ln -sf acestep-v15-xl-sft xl_sft
cd "$WORK"

echo "==> [5/5] Ensure data/output directories"
mkdir -p "$WORK/output"
mkdir -p "$WORK/datasets/vinahouse/phase_b_dataset_v2/tensors"

echo ""
echo "[setup] Done. Next, from your LOCAL machine run:"
echo "    bash vast/push_data.sh <IP> <PORT>          # upload tensors"
echo "    bash vast/push_data.sh <IP> <PORT> resume   # + upload outputs to resume"
