"""Extract BPM from audio clips using librosa.

For Vinahouse (steady 4/4 kick pattern) librosa's beat tracker is very accurate.
Reads all audio files in a folder and writes:
    - CSV report with per-file BPM
    - Optionally updates/creates a {filename}.json per clip with BPM info

Usage:
    .venv/bin/python datasets/vinahouse/scripts/05_extract_bpm.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --out-csv   datasets/vinahouse/bpm_report.csv \\
        --write-json

Notes:
    - Vinahouse typically 130-145 BPM. If librosa returns half/double tempo
      (65 or 280), the script auto-corrects to expected range.
    - JSON files are merged (not overwritten) if they already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("[ERROR] librosa not installed. Run: .venv/bin/pip install librosa", file=sys.stderr)
    raise


TARGET_SR = 22050
HOP_LENGTH = 512

VINAHOUSE_BPM_MIN = 120.0
VINAHOUSE_BPM_MAX = 150.0
VINAHOUSE_BPM_TARGET = 138.0  # Center of expected Vinahouse tempo


def normalize_bpm(bpm: float, target: float = VINAHOUSE_BPM_TARGET) -> float:
    """Correct half/double-tempo detections by picking the octave closest to target.

    librosa's beat_track can under-/over-estimate by 2x (or 4x for very slow music).
    For a known-genre dataset (Vinahouse ~138 BPM), pick whichever multiple of the
    raw BPM lies closest to the target.

    Example (raw=76, target=138):
        candidates = [19, 38, 76, 152, 304]
        distances  = [119, 100, 62, 14, 166]
        -> pick 152 (correct: 76 * 2)
    """
    if bpm <= 0:
        return 0.0
    candidates = [bpm / 4.0, bpm / 2.0, bpm, bpm * 2.0, bpm * 4.0]
    return min(candidates, key=lambda x: abs(x - target))


def extract_bpm(path: Path) -> tuple[float, float, float]:
    """Return (raw_bpm, normalized_bpm, confidence)."""
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH, tightness=100)
    raw_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    norm_bpm = normalize_bpm(raw_bpm)

    if len(beats) < 4:
        return raw_bpm, norm_bpm, 0.0

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
    intervals = np.diff(beat_times)
    if intervals.size == 0 or np.mean(intervals) == 0:
        return raw_bpm, norm_bpm, 0.0

    cv = float(np.std(intervals) / np.mean(intervals))
    confidence = float(max(0.0, min(1.0, 1.0 - cv * 3.0)))
    return raw_bpm, norm_bpm, confidence


def update_json(audio_path: Path, bpm: float, confidence: float) -> None:
    """Create or update {stem}.json alongside the audio file."""
    json_path = audio_path.with_suffix(".json")
    data: dict = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data["bpm"] = round(bpm)
    data["bpm_confidence"] = round(confidence, 2)
    if "timesignature" not in data:
        data["timesignature"] = "4"
    if "language" not in data:
        data["language"] = "vi"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg"],
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also create/update {stem}.json next to each audio file",
    )
    parser.add_argument(
        "--target-bpm",
        type=float,
        default=VINAHOUSE_BPM_TARGET,
        help=f"Target BPM for octave-normalization (default {VINAHOUSE_BPM_TARGET} for Vinahouse)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in args.exts])
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Extracting BPM from {len(files)} files.")
    print(f"Normalizing to closest octave of target BPM = {args.target_bpm}.\n")

    results: list[dict] = []
    t_start = time.time()
    for i, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            raw_bpm, _, conf = extract_bpm(path)
            norm_bpm = normalize_bpm(raw_bpm, target=args.target_bpm)
        except Exception as e:
            print(f"[WARN] Failed {path.name}: {e}", file=sys.stderr)
            continue

        results.append(
            {
                "filename": path.name,
                "bpm_raw": round(raw_bpm, 2),
                "bpm_normalized": round(norm_bpm, 2),
                "bpm_rounded": round(norm_bpm),
                "confidence": round(conf, 2),
            }
        )

        if args.write_json:
            update_json(path, norm_bpm, conf)

        elapsed = time.time() - t0
        conf_label = "HIGH" if conf > 0.85 else "MED" if conf > 0.65 else "LOW"
        print(
            f"[{i:3d}/{len(files)}] "
            f"raw={raw_bpm:6.2f}  norm={norm_bpm:6.2f}  conf={conf:.2f} [{conf_label:4s}]  "
            f"{path.name[:50]}  ({elapsed:.1f}s)"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - t_start
    bpm_values = np.array([r["bpm_normalized"] for r in results if r["confidence"] > 0.65])
    n_low = sum(1 for r in results if r["confidence"] <= 0.65)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:     {len(results)}")
    if bpm_values.size:
        print(f"BPM mean:        {bpm_values.mean():.1f}")
        print(f"BPM median:      {np.median(bpm_values):.1f}")
        print(f"BPM min-max:     {bpm_values.min():.1f} - {bpm_values.max():.1f}")
    print(f"Low confidence:  {n_low}  (may need manual verify)")
    print(f"Total time:      {total_time:.1f}s")
    print(f"CSV:             {args.out_csv}")
    if args.write_json:
        print(f"JSON per clip:   written next to each audio (with bpm, timesignature=4, language=vi)")
    print()
    print("Next step: extract Key (06_extract_key.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
