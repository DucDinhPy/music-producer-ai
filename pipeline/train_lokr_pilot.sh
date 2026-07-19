#!/bin/bash
# Phase A: Train LoKr pilot on 44 Vinahouse clips
# Target: 8GB VRAM (RTX 4060) with AdamW8bit + offload-encoder + gradient-checkpointing
#
# Usage:
#     cd /mnt/d/AI\ project/music_producer_ai/ACE-Step-1.5
#     bash datasets/vinahouse/scripts/train_lokr_pilot.sh

set -e

cd "$(dirname "$0")/../../.."  # Move to ACE-Step-1.5 root

echo "=== Training LoKr pilot on Vinahouse Phase A ==="
echo "Working dir: $(pwd)"
echo ""

uv run python train.py --yes fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --adapter-type lokr \
    --lokr-linear-dim 16 \
    --lokr-linear-alpha 32 \
    --lokr-weight-decompose \
    --dataset-dir ./datasets/vinahouse/phase_a/tensors \
    --output-dir ./output/vinahouse_lokr_pilot_v2 \
    --optimizer-type adamw8bit \
    --offload-encoder \
    --precision bf16 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --epochs 100 \
    --lr 0.005 \
    --warmup-steps 250 \
    --max-grad-norm 0.5 \
    --shift 3.0 \
    --save-every 10 \
    --seed 42
