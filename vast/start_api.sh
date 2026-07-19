#!/usr/bin/env bash
# =============================================================================
# start_api.sh  --  Launch the ACE-Step REST API with a customizable config.
# Run this ON the Vast instance (inside tmux). Loads your LoRA automatically.
# =============================================================================
# Endpoints once running:
#   POST /release_task     submit a generation      (see test_run.py)
#   POST /query_result     poll task status
#   GET  /v1/lora/status   inspect adapter state
#   POST /v1/lora/load|toggle|scale|unload   adapter controls
#   GET  /docs             OpenAPI docs
# =============================================================================
set -euo pipefail

# ============================ CONFIG (edit me) ===============================
# ---- Main DiT model ----
MAIN_MODEL="acestep-v15-xl-sft"          # or: acestep-v15-xl-sft
# ---- Your trained adapter (leave empty to run base model only) ----
LORA_PATH="./output/vinahouse_base_lokr/checkpoints/epoch_200_loss_0.8098/lokr_weights.safetensors"
LORA_SCALE="1.0"                        # 0.0 - 1.0
# ---- 5Hz language model ----
INIT_LLM="true"                         # auto | true | false
LM_MODEL="acestep-5Hz-lm-4B"          # 0.6B | 1.7B | 4B
# ---- Server ----
HOST="0.0.0.0"
PORT="8001"
API_KEY=""                              # optional bearer key; empty = no auth
# ---- Performance ----
COMPILE_MODEL="false"                   # keep false on new CUDA drivers (noise risk)
OFFLOAD_TO_CPU="false"
EAGER_INIT="true"                       # load model at startup (required for LoRA-before-gen)
# =============================================================================

WORK=/workspace
cd "$WORK"

# ---- Export runtime config the API reads from the environment ----
export ACESTEP_CONFIG_PATH="$MAIN_MODEL"
export ACESTEP_INIT_LLM="$INIT_LLM"
export ACESTEP_COMPILE_MODEL="$COMPILE_MODEL"
export ACESTEP_OFFLOAD_TO_CPU="$OFFLOAD_TO_CPU"
if [ "$EAGER_INIT" = "true" ]; then export ACESTEP_NO_INIT=false; else export ACESTEP_NO_INIT=true; fi
[ -n "$API_KEY" ] && export ACESTEP_API_KEY="$API_KEY"

# ---- Build launch command ----
ARGS=(acestep-api --host "$HOST" --port "$PORT")
[ -n "$API_KEY" ]  && ARGS+=(--api-key "$API_KEY")
[ -n "$LM_MODEL" ] && ARGS+=(--lm-model-path "$LM_MODEL")

echo "==> Starting API: model=$MAIN_MODEL lm=$LM_MODEL http://$HOST:$PORT"
uv run --no-sync "${ARGS[@]}" &
SERVER_PID=$!

# ---- Auto-load LoRA once the model has finished initializing ----
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
        echo "==> LoRA active. Check: curl http://127.0.0.1:$PORT/v1/lora/status"
    fi
fi

echo "==> API ready. Ctrl+C to stop."
wait "$SERVER_PID"
