#!/usr/bin/env bash
# =============================================================================
# vast/push_data.sh  --  Run this on your LOCAL machine (WSL)
# =============================================================================
# Uploads the training data (and optionally prior outputs) to a Vast instance.
# Training only needs the preprocessed tensors (~2.8GB), NOT the audio.
#
# Usage:
#   bash vast/push_data.sh <IP> <PORT>            # tensors only
#   bash vast/push_data.sh <IP> <PORT> resume     # tensors + outputs (to resume)
# =============================================================================
set -euo pipefail

IP="${1:?usage: push_data.sh <IP> <PORT> [resume]}"
PORT="${2:?usage: push_data.sh <IP> <PORT> [resume]}"
MODE="${3:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Uploading tensors -> $IP:$PORT"
rsync -avz --mkpath -e "ssh -p $PORT" \
    ./datasets/vinahouse/phase_b_dataset_v2/tensors/ \
    root@"$IP":/workspace/datasets/vinahouse/phase_b_dataset_v2/tensors/

if [ "$MODE" = "resume" ]; then
    echo "==> Uploading outputs (resume) -> $IP:$PORT"
    rsync -avz --mkpath -e "ssh -p $PORT" \
        ./vast_backup/output/ \
        root@"$IP":/workspace/output/
fi

echo "[push_data] Done."
