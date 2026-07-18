# Vinahouse LoRA — Workflow & Structure

How code, data, models, and outputs are organized and operated across
ephemeral Vast.ai instances.

## Mental model: 4 asset classes by lifecycle

| Asset | Size | Source of truth | How it reaches an instance |
|-------|------|-----------------|----------------------------|
| **Code** | small | GitHub (your repo) | `git` sync via `vast/setup.sh` |
| **Env / deps** | medium | `pyproject.toml` + `uv.lock` | baked into the Docker image |
| **Base models** | ~20GB | HuggingFace | `acestep-download` via `vast/setup.sh` |
| **Dataset (tensors)** | ~2.8GB | local (`datasets/`) | `rsync` via `vast/push_data.sh` |
| **Outputs (adapters)** | small | generated on Vast | `rsync` back via `vast/pull_output.sh` |

Key facts:
- **Training only needs `tensors/`** (audio is already encoded into them). Audio
  stays local and is only needed to *re-preprocess*.
- **`datasets/` is gitignored and excluded from the Docker image.** It never
  travels via git or the image — only via `rsync`.
- The Docker **image = environment only** (rebuild rarely, when deps change).
  Code updates come from `git`, not image rebuilds.

## Local folder layout

```
ACE-Step-1.5/
  pipeline/                       # data-prep scripts 01-12 (git-tracked CODE)
  vast/                           # instance lifecycle helpers
    setup.sh                      #   run ON instance: git sync + models + dirs
    push_data.sh                  #   run on LOCAL: upload tensors (+outputs)
    pull_output.sh                #   run on LOCAL: download trained adapters
  datasets/vinahouse/             # DATA (gitignored, local source-of-truth)
    audio/                        #   raw mixes + full songs (immutable)
    phase_b/                      #   working intermediates (regenerable)
    phase_b_dataset_v2/           #   current dataset
      dataset.json                #     manifest (for preprocessing)
      tensors/                    #     preprocessed  <-- the only thing pushed to Vast
      audio/                      #     clip copies (for re-preprocess only)
  vast_backup/output/            # trained adapters pulled back from Vast (gitignored)
  Dockerfile, pyproject.toml, ... # environment definition
```

## Instance lifecycle (the whole loop)

### 1. Create instance from the image
Launch a Vast.ai instance using your GHCR image. Note its **IP** and **SSH PORT**.

### 2. Setup (on the instance)
```bash
cd /workspace
REPO_URL=https://github.com/<you>/<repo>.git bash vast/setup.sh
```
Downloads models, recreates the `xl_sft` symlink, makes directories, syncs code.

> First time only: the image must contain `vast/setup.sh`. If it doesn't yet,
> `git clone` your repo manually once, then use `setup.sh` thereafter.

### 3. Push data (on local / WSL)
```bash
bash vast/push_data.sh <IP> <PORT>            # tensors only (fresh train)
bash vast/push_data.sh <IP> <PORT> resume     # tensors + outputs (resume train)
```

### 4. Train (on the instance, inside tmux)
```bash
tmux new -s dl
cd /workspace
uv run python train.py --yes fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant base --base-model base \
    --adapter-type lokr \
    --dataset-dir ./datasets/vinahouse/phase_b_dataset_v2/tensors \
    --output-dir ./output/vinahouse_base_lokr \
    --optimizer-type adamw8bit --lr 0.005 \
    --batch-size 24 --gradient-accumulation 1 \
    --shift 1.0 --num-inference-steps 50 \
    --epochs 200 --save-every 10 \
    2>&1 | tee /workspace/train.log
```
(For `xl_sft`, match the saved LoKr config — see "Resuming xl_sft" below.)

### 5. Pull outputs (on local, periodically + before destroy)
```bash
bash vast/pull_output.sh <IP> <PORT>
```

### 6. Destroy the instance
Only after step 5 confirms outputs are backed up locally.

## Resuming xl_sft with the correct LoKr architecture

The XL-SFT adapter was trained with a **non-default** config. Resuming with
wrong flags produces a mismatched param count and fails to load the optimizer.
The exact config is stored inside each checkpoint:

```bash
uv run python -c "
from safetensors import safe_open
p='./output/.../epoch_XX_.../lokr_weights.safetensors'
with safe_open(p,'pt') as f: print(f.metadata())
"
```

Known xl_sft config (3,421,184 params):
`--lokr-linear-dim 32 --lokr-linear-alpha 64 --lokr-factor=-1 --lokr-weight-decompose --attention-type both --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`

## Gotchas

- **UI noise after toggling LoRA**: the Gradio "Initialize Service" does NOT
  reset a corrupted process. Fully restart: `pkill -f acestep` then relaunch.
  Use one fixed LoRA state per process launch — never Load/Unload repeatedly.
- **`torch.compile` on new CUDA drivers** can produce noise — keep it OFF.
- **Flow-matching loss barely moves** (~0.86 plateau) — this is normal. Judge by
  listening to output, not the loss number.
- **base/sft need** `--shift 1.0 --num-inference-steps 50`; turbo uses `3.0`/`8`.
