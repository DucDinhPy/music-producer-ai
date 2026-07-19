#!/usr/bin/env bash
# =============================================================================
# start_api.sh  --  Launch ACE-Step REST API with the WORKING UI config.
# Run ON the Vast instance (inside tmux). Auto-loads your LoRA.
# =============================================================================
# Matches the Gradio setup that successfully generates music:
#   Main: acestep-v15-xl-sft | LM: 1.7B (vllm) | Compile ON | FlashAttn ON
#   LoRA: epoch_200 vinahouse xl_sft | scale 1.0
# =============================================================================
set -euo pipefail

# ============================ CONFIG (edit me) ===============================
# ---- Main DiT model (Service Configuration) ----
MAIN_MODEL="acestep-v15-xl-sft"
# ---- Trained adapter (leave empty = base only) ----
LORA_PATH="./output/vinahouse_phase_b_v2_lokr_xl_sft/checkpoints/epoch_200_loss_0.8098/lokr_weights.safetensors"
LORA_SCALE="1.0"
# ---- 5Hz language model ----
INIT_LLM="true"
LM_MODEL="acestep-5Hz-lm-1.7B"          # matches working UI
LM_BACKEND="vllm"
# ---- Server ----
HOST="0.0.0.0"
PORT="8001"
API_KEY=""
# ---- Performance (match working UI) ----
COMPILE_MODEL="true"                    # ON in the working UI setup
USE_FLASH_ATTENTION="true"
OFFLOAD_TO_CPU="false"
EAGER_INIT="true"                       # must be true so LoRA can load at boot
# =============================================================================

WORK=/workspace
cd "$WORK"

export ACESTEP_CONFIG_PATH="$MAIN_MODEL"
export ACESTEP_INIT_LLM="$INIT_LLM"
export ACESTEP_LM_BACKEND="$LM_BACKEND"
export ACESTEP_COMPILE_MODEL="$COMPILE_MODEL"
export ACESTEP_USE_FLASH_ATTENTION="$USE_FLASH_ATTENTION"
export ACESTEP_OFFLOAD_TO_CPU="$OFFLOAD_TO_CPU"
if [ "$EAGER_INIT" = "true" ]; then export ACESTEP_NO_INIT=false; else export ACESTEP_NO_INIT=true; fi
[ -n "$API_KEY" ] && export ACESTEP_API_KEY="$API_KEY"

ARGS=(acestep-api --host "$HOST" --port "$PORT")
[ -n "$API_KEY" ]  && ARGS+=(--api-key "$API_KEY")
[ -n "$LM_MODEL" ] && ARGS+=(--lm-model-path "$LM_MODEL")

echo "==> Starting API: model=$MAIN_MODEL lm=$LM_MODEL backend=$LM_BACKEND compile=$COMPILE_MODEL"
echo "    http://$HOST:$PORT  |  docs: http://$HOST:$PORT/docs"
uv run --no-sync "${ARGS[@]}" &
SERVER_PID=$!

if [ -n "$LORA_PATH" ]; then
    AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
    echo "==> Waiting for model init (up to 10 min)..."
    ready=0
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[error] API process exited during startup"; exit 1
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" \
            "http://127.0.0.1:$PORT/v1/lora/status" 2>/dev/null || echo 000)
        if [ "$code" = "200" ]; then ready=1; break; fi
        sleep 5
    done
    if [ "$ready" != "1" ]; then
        echo "[warn] model not ready in time; load LoRA manually via /v1/lora/load"
    else
        echo "==> Loading LoRA: $LORA_PATH"
        curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
            -X POST "http://127.0.0.1:$PORT/v1/lora/load" \
            -d "{\"lora_path\": \"$LORA_PATH\"}"; echo
        curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
            -X POST "http://127.0.0.1:$PORT/v1/lora/toggle" \
            -d '{"use_lora": true}'; echo
        curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
            -X POST "http://127.0.0.1:$PORT/v1/lora/scale" \
            -d "{\"scale\": $LORA_SCALE}"; echo
        echo "==> LoRA status:"
        curl -s "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/lora/status"; echo
    fi
fi

echo "==> API ready. Ctrl+C to stop."
wait "$SERVER_PID"
