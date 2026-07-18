#!/usr/bin/env bash
# =============================================================================
# vast/pull_output.sh  --  Run this on your LOCAL machine (WSL)
# =============================================================================
# Downloads trained adapters from a Vast instance into ./vast_backup/output/.
# Run periodically during training and always BEFORE destroying an instance.
#
# Usage:
#   bash vast/pull_output.sh <IP> <PORT>
# =============================================================================
set -euo pipefail

IP="${1:?usage: pull_output.sh <IP> <PORT>}"
PORT="${2:?usage: pull_output.sh <IP> <PORT>}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Downloading outputs <- $IP:$PORT"
rsync -avz --mkpath -e "ssh -p $PORT" \
    root@"$IP":/workspace/output/ \
    ./vast_backup/output/

echo "[pull_output] Done. Backup at ./vast_backup/output/"
