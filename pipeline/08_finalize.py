"""Finalize captioned Vinahouse clips into a Side-Step / ACE-Step-ready dataset.

Reads all `{stem}.wav` + `{stem}.json` pairs from the source folder, keeps only
those where the JSON contains a non-empty `caption`, copies them to the phase
folder, and emits a master `dataset.json` in the full ACE-Step Dataset Builder
format that Side-Step (`train.py fixed --dataset-json ...`) understands.

Phase A design:
    - All clips marked instrumental (skip lyrics for pilot -- focus on genre style)
    - custom_tag = "vinahouse" prepended to every caption at training time
    - genre_ratio = 0 (use caption, not genre, as prompt)

Usage:
    .venv/bin/python datasets/vinahouse/scripts/08_finalize.py \\
        --source-dir datasets/vinahouse/audio_clean \\
        --phase-dir datasets/vinahouse/phase_a \\
        --custom-tag vinahouse \\
        --all-instrumental

    # Dry-run to preview without copying
    .venv/bin/python datasets/vinahouse/scripts/08_finalize.py \\
        --source-dir datasets/vinahouse/audio_clean \\
        --phase-dir datasets/vinahouse/phase_a \\
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def stable_id(stem: str) -> str:
    """Deterministic 8-char hex id derived from filename (stable across runs)."""
    return hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]


def audio_duration_seconds(path: Path) -> float:
    """Read audio duration without loading full waveform. Returns 0 on failure.

    Tries stdlib `wave` first (works for .wav on any platform, no deps),
    falls back to `soundfile` for non-WAV formats. Silent on missing deps
    so this script is runnable in any Python env.
    """
    if path.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0:
                    return round(frames / rate, 2)
        except Exception:
            pass

    try:
        import soundfile as sf
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 2)
    except ImportError:
        return 0.0
    except Exception as e:
        print(f"[WARN] Cannot read duration for {path.name}: {e}", file=sys.stderr)
        return 0.0


def build_sample(
    audio_path: Path,
    meta: dict,
    custom_tag: str,
    force_all_instrumental: bool,
    audio_dir_relative: str = "./audio",
    trigger_on_kick_only: bool = False,
) -> dict:
    """Convert per-clip JSON metadata into ACE-Step sample dict.

    is_instrumental / lyrics resolution priority:
      1. If --force-all-instrumental: mark instrumental for all clips
      2. Else if meta has 'is_instrumental' key (from Whisper 09_transcribe): use it
      3. Else: default to instrumental (safe fallback)

    custom_tag resolution:
      - If trigger_on_kick_only=True: use custom_tag only when meta['has_kick'] is True,
        else set to "" so the trigger word does not get bound to no-kick clips.
      - Else: use custom_tag for every sample (legacy behavior).
    """
    stem = audio_path.stem
    caption = meta.get("caption", "").strip()
    bpm = meta.get("bpm")
    keyscale = meta.get("keyscale") or "N/A"
    timesignature = meta.get("timesignature", "4")

    if force_all_instrumental:
        is_instrumental = True
        lyrics = "[Instrumental]"
        raw_lyrics = ""
        vocal_lang = "unknown"
    elif "is_instrumental" in meta:
        is_instrumental = bool(meta["is_instrumental"])
        if is_instrumental:
            lyrics = "[Instrumental]"
            raw_lyrics = ""
            vocal_lang = "unknown"
        else:
            lyrics = (meta.get("lyrics") or "").strip() or "[Instrumental]"
            raw_lyrics = (meta.get("raw_lyrics") or "").strip()
            vocal_lang = meta.get("vocal_language", "vi")
    else:
        is_instrumental = True
        lyrics = "[Instrumental]"
        raw_lyrics = ""
        vocal_lang = "unknown"

    language = vocal_lang if not is_instrumental else meta.get("language", "vi")

    has_kick = bool(meta.get("has_kick", False))
    if trigger_on_kick_only:
        effective_tag = custom_tag if has_kick else ""
    else:
        effective_tag = custom_tag

    return {
        "id": stable_id(stem),
        "audio_path": f"{audio_dir_relative}/{audio_path.name}",
        "filename": audio_path.name,
        "caption": caption,
        "genre": "vinahouse",
        "lyrics": lyrics,
        "raw_lyrics": raw_lyrics,
        "formatted_lyrics": lyrics,
        "bpm": int(bpm) if isinstance(bpm, (int, float)) and bpm > 0 else "N/A",
        "keyscale": keyscale,
        "timesignature": str(timesignature),
        "duration": audio_duration_seconds(audio_path),
        "language": language,
        "is_instrumental": is_instrumental,
        "custom_tag": effective_tag,
        "has_kick": has_kick,
        "labeled": True,
        "prompt_override": None,
    }


def validate_sample(sample: dict) -> list[str]:
    """Return list of warning strings for a sample (empty = clean)."""
    warns = []
    if not sample.get("caption"):
        warns.append("empty caption")
    if not sample.get("bpm") or sample["bpm"] == "N/A":
        warns.append("missing bpm")
    if not sample.get("keyscale") or sample["keyscale"] == "N/A":
        warns.append("missing keyscale")
    if sample.get("duration", 0) < 20:
        warns.append(f"duration too short ({sample['duration']}s)")
    if sample.get("duration", 0) > 300:
        warns.append(f"duration too long ({sample['duration']}s)")
    return warns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--custom-tag", default="vinahouse", help="Trigger word prepended to caption at training time")
    parser.add_argument("--tag-position", default="prepend", choices=["prepend", "append", "replace"])
    parser.add_argument("--force-all-instrumental", "--all-instrumental",
                        dest="force_all_instrumental", action="store_true",
                        help="Force all samples to instrumental, ignoring per-clip is_instrumental from Whisper.")
    parser.add_argument("--trigger-on-kick-only", action="store_true",
                        help="Apply --custom-tag only to clips where has_kick=True (from 12_relabel_kick.py). "
                             "Prevents the trigger word from being bound to no-kick / breakdown clips, "
                             "which caused the 'vinahouse = no kick' bias in Phase B v1.")
    parser.add_argument("--genre-ratio", type=int, default=0, help="0-100. 0 = caption only, 100 = genre only")
    parser.add_argument("--dataset-name", default="vinahouse_phase_a")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy files or write dataset.json")
    parser.add_argument("--force", action="store_true", help="Overwrite existing phase-dir contents")
    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"[ERROR] Source dir not found: {args.source_dir}", file=sys.stderr)
        return 1

    # Collect audio files that have a matching JSON with non-empty caption
    all_audio = sorted([p for p in args.source_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    if not all_audio:
        print(f"[ERROR] No audio files in {args.source_dir}", file=sys.stderr)
        return 1

    ready = []
    skipped = []
    for a in all_audio:
        j = a.with_suffix(".json")
        if not j.exists():
            skipped.append((a.name, "no json"))
            continue
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append((a.name, "invalid json"))
            continue
        if not meta.get("caption", "").strip():
            skipped.append((a.name, "empty caption"))
            continue
        ready.append((a, meta))

    print("=" * 70)
    print(f"Source: {args.source_dir}")
    print(f"Total audio files:  {len(all_audio)}")
    print(f"Ready (has caption): {len(ready)}")
    print(f"Skipped:             {len(skipped)}")
    print("=" * 70)

    if skipped:
        print("\nSkipped files (first 10):")
        for name, reason in skipped[:10]:
            print(f"  - {name}  [{reason}]")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more")

    if not ready:
        print("\n[ERROR] Nothing to finalize. Ensure captioning finished first.", file=sys.stderr)
        return 1

    # Build phase-dir/audio/  (copies) + dataset.json
    audio_target = args.phase_dir / "audio"

    if args.phase_dir.exists() and any(args.phase_dir.iterdir()):
        if not args.force and not args.dry_run:
            print(f"\n[ERROR] Phase dir already has content: {args.phase_dir}", file=sys.stderr)
            print("        Pass --force to overwrite, or use a new phase-dir path.")
            return 1

    samples = []
    warnings_report = []
    for i, (audio_path, meta) in enumerate(ready, start=1):
        sample = build_sample(
            audio_path=audio_path,
            meta=meta,
            custom_tag=args.custom_tag,
            force_all_instrumental=args.force_all_instrumental,
            trigger_on_kick_only=args.trigger_on_kick_only,
        )
        w = validate_sample(sample)
        if w:
            warnings_report.append((audio_path.name, w))
        samples.append(sample)

    n_kick_samples = sum(1 for s in samples if s.get("has_kick"))
    n_tag_active = sum(1 for s in samples if s.get("custom_tag"))

    dataset = {
        "metadata": {
            "name": args.dataset_name,
            "custom_tag": args.custom_tag,
            "tag_position": args.tag_position,
            "trigger_on_kick_only": args.trigger_on_kick_only,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(samples),
            "num_vocal":        sum(1 for s in samples if not s["is_instrumental"]),
            "num_instrumental": sum(1 for s in samples if s["is_instrumental"]),
            "num_with_kick":    n_kick_samples,
            "num_with_active_tag": n_tag_active,
            "force_all_instrumental": args.force_all_instrumental,
            "genre_ratio": args.genre_ratio,
        },
        "samples": samples,
    }

    # Print stats before commit
    n_vocal = sum(1 for s in samples if not s["is_instrumental"])
    n_inst = len(samples) - n_vocal

    print("\nDataset stats:")
    print(f"  Vocal / Instr: {n_vocal} vocal, {n_inst} instrumental ({n_vocal / len(samples) * 100:.0f}% vocal)")
    print(f"  Has kick:      {n_kick_samples} / {len(samples)} ({n_kick_samples / len(samples) * 100:.0f}%)")
    if args.trigger_on_kick_only:
        print(f"  Active '{args.custom_tag}' trigger tag: {n_tag_active} clips  <- only samples WITH kick")
    else:
        print(f"  Active '{args.custom_tag}' trigger tag: {n_tag_active} clips  (applied to all)")
    bpm_vals = [s["bpm"] for s in samples if isinstance(s["bpm"], int)]
    if bpm_vals:
        print(f"  BPM range:    {min(bpm_vals)} - {max(bpm_vals)}  (avg {sum(bpm_vals)/len(bpm_vals):.1f})")
    key_counts = {}
    for s in samples:
        key_counts[s["keyscale"]] = key_counts.get(s["keyscale"], 0) + 1
    top_keys = sorted(key_counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Top keys:     {', '.join(f'{k}({v})' for k, v in top_keys)}")
    dur_vals = [s["duration"] for s in samples if s["duration"] > 0]
    if dur_vals:
        print(f"  Duration:     {min(dur_vals):.1f}s - {max(dur_vals):.1f}s  (avg {sum(dur_vals)/len(dur_vals):.1f}s, total {sum(dur_vals)/60:.1f}min)")

    if warnings_report:
        print(f"\n[!] {len(warnings_report)} sample(s) with warnings:")
        for name, ws in warnings_report[:10]:
            print(f"  - {name}: {', '.join(ws)}")
        if len(warnings_report) > 10:
            print(f"  ... {len(warnings_report) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files copied, no JSON written.")
        print("First sample preview:")
        print(json.dumps(samples[0], indent=2, ensure_ascii=False))
        return 0

    # Actually copy files
    audio_target.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying {len(ready)} audio files -> {audio_target}")
    t0 = time.time()
    for audio_path, _ in ready:
        dst = audio_target / audio_path.name
        if not dst.exists() or args.force:
            shutil.copy2(audio_path, dst)
    print(f"Copied in {time.time() - t0:.1f}s")

    # Write dataset.json at phase_dir root
    dataset_json_path = args.phase_dir / "dataset.json"
    dataset_json_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote dataset manifest: {dataset_json_path}")
    print(f"  {len(samples)} samples")
    print(f"  custom_tag={args.custom_tag!r} ({args.tag_position})")
    print(f"  vocal:instr = {n_vocal}:{n_inst}")
    if args.force_all_instrumental:
        print(f"  [!] force_all_instrumental=True — all clips overridden as instrumental")
    print(f"  genre_ratio={args.genre_ratio}")

    print("\n" + "=" * 70)
    print("Next steps (assumes Vast RTX 4090 48GB / XL-SFT, LoKr strong config):")
    print("=" * 70)
    print(f"""
# -----------------------------------------------------------------
#  A. LOCAL -> VAST: upload dataset (audio + dataset.json)
# -----------------------------------------------------------------
rsync -avz --progress -e "ssh -p <VAST_PORT>" \\
    {args.phase_dir}/ \\
    root@<VAST_IP>:/workspace/{args.phase_dir}/

# -----------------------------------------------------------------
#  B. ON VAST: preprocess audio -> latent tensors (~40-60 min)
# -----------------------------------------------------------------
cd /workspace
tmux new -s preprocess

uv run python train.py --yes fixed \\
    --checkpoint-dir ./checkpoints \\
    --model-variant xl_sft \\
    --preprocess \\
    --dataset-json ./{args.phase_dir}/dataset.json \\
    --audio-dir ./{args.phase_dir}/audio \\
    --dataset-dir ./{args.phase_dir}/tensors \\
    --output-dir ./output/preprocess_stub \\
    --tensor-output ./{args.phase_dir}/tensors 2>&1 | tee preprocess.log

# Detach: Ctrl+B D

# -----------------------------------------------------------------
#  C. ON VAST: train LoKr adapter from SCRATCH (do not resume epoch_90 - it has bias)
# -----------------------------------------------------------------
tmux new -s train

uv run python train.py --yes fixed \\
    --checkpoint-dir ./checkpoints \\
    --model-variant xl_sft \\
    --adapter-type lokr \\
    --lokr-linear-dim 32 \\
    --lokr-linear-alpha 64 \\
    --lokr-weight-decompose \\
    --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \\
    --dataset-dir ./{args.phase_dir}/tensors \\
    --output-dir ./output/{args.dataset_name}_lokr_xl_sft \\
    --optimizer-type adamw8bit \\
    --precision bf16 \\
    --batch-size 3 \\
    --gradient-accumulation 2 \\
    --epochs 200 \\
    --lr 0.003 \\
    --warmup-steps 300 \\
    --max-grad-norm 0.5 \\
    --shift 1.0 \\
    --save-every 10 \\
    --offload-encoder \\
    --seed 42 \\
    2>&1 | tee train_phase_b_v2.log

# Detach: Ctrl+B D
# Monitor: tmux attach -t train
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
