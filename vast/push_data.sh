#!/usr/bin/env bash
# =============================================================================
# vast/push_data.sh  --  Run this on your LOCAL machine (WSL)
# =============================================================================
# Uploads training tensors and/or prior outputs to a Vast instance.
# Training only needs the preprocessed tensors (~2.8GB), NOT the audio.
#
# Usage:
#   bash vast/push_data.sh <IP> <PORT>             # tensors only
#   bash vast/push_data.sh <IP> <PORT> tensors     # tensors only
#   bash vast/push_data.sh <IP> <PORT> output      # outputs only
#   bash vast/push_data.sh <IP> <PORT> resume      # tensors + outputs
# =============================================================================
set -euo pipefail

IP="${1:?usage: push_data.sh <IP> <PORT> [tensors|output|resume]}"
PORT="${2:?usage: push_data.sh <IP> <PORT> [tensors|output|resume]}"
MODE="${3:-tensors}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$MODE" = "tensors" ]; then
    echo "==> Uploading tensors -> $IP:$PORT"

    rsync -avz --partial --info=progress2 --mkpath \
        -e "ssh -p $PORT" \
        ./datasets/vinahouse/phase_b_dataset_v2/tensors/ \
        root@"$IP":/workspace/datasets/vinahouse/phase_b_dataset_v2/tensors/

elif [ "$MODE" = "output" ]; then
    echo "==> Uploading outputs only -> $IP:$PORT"

    rsync -avz --partial --info=progress2 --mkpath \
        -e "ssh -p $PORT" \
        ./vast_backup/output/ \
        root@"$IP":/workspace/output/

elif [ "$MODE" = "resume" ]; then
    echo "==> Uploading tensors -> $IP:$PORT"

    rsync -avz --partial --info=progress2 --mkpath \
        -e "ssh -p $PORT" \
        ./datasets/vinahouse/phase_b_dataset_v2/tensors/ \
        root@"$IP":/workspace/datasets/vinahouse/phase_b_dataset_v2/tensors/

    echo "==> Uploading outputs -> $IP:$PORT"

    rsync -avz --partial --info=progress2 --mkpath \
        -e "ssh -p $PORT" \
        ./vast_backup/output/ \
        root@"$IP":/workspace/output/

else
    echo "[ERROR] Invalid mode: $MODE"
    echo "Usage:"
    echo "  bash vast/push_data.sh <IP> <PORT>"
    echo "  bash vast/push_data.sh <IP> <PORT> tensors"
    echo "  bash vast/push_data.sh <IP> <PORT> output"
    echo "  bash vast/push_data.sh <IP> <PORT> resume"
    exit 1
fi

echo "[push_data] Done."