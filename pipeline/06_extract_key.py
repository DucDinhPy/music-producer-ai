"""Extract musical key from audio clips using Krumhansl-Schmuckler algorithm.

Approach:
    1. Compute constant-Q chroma (12 pitch classes) with librosa
    2. Sum over time -> single 12-dim vector representing overall tonal content
    3. Correlate against Krumhansl 24 key profiles (12 major + 12 minor)
    4. Pick the key with highest correlation

Vinahouse tracks are usually in minor keys (F#m, Am, Cm, C#m are common),
so we optionally bias toward minor keys with --minor-bias.

Usage:
    .venv/bin/python datasets/vinahouse/scripts/06_extract_key.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --out-csv   datasets/vinahouse/key_report.csv \\
        --write-json
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
HOP_LENGTH = 4096  # coarser hop for key detection (harmonic content changes slowly)

# Krumhansl-Schmuckler key profiles (from empirical listener studies).
# Index 0 = C. Rotate to match candidate root.
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(y: np.ndarray, sr: int, minor_bias: float = 0.0) -> tuple[str, float, str]:
    """Return (key_name, confidence, mode) where key_name is like 'F# minor'.

    minor_bias: additive score bonus for minor profiles (0.0 = neutral,
                0.05-0.10 recommended for EDM/Vinahouse which is minor-dominated)
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma_avg = chroma.mean(axis=1)
    chroma_avg = chroma_avg / (chroma_avg.sum() + 1e-9)

    def correlate(profile: np.ndarray, offset: int) -> float:
        rotated = np.roll(profile, offset)
        rotated = rotated / (rotated.sum() + 1e-9)
        return float(np.corrcoef(chroma_avg, rotated)[0, 1])

    scores: list[tuple[float, int, str]] = []
    for i in range(12):
        maj_score = correlate(KRUMHANSL_MAJOR, i)
        min_score = correlate(KRUMHANSL_MINOR, i) + minor_bias
        scores.append((maj_score, i, "major"))
        scores.append((min_score, i, "minor"))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_root, best_mode = scores[0]
    second_score = scores[1][0]

    confidence = float(max(0.0, min(1.0, best_score - max(0.0, second_score))))
    key_name = f"{PITCH_NAMES[best_root]} {best_mode}"
    return key_name, confidence, best_mode


def update_json(audio_path: Path, key_name: str, confidence: float) -> None:
    """Merge key info into {stem}.json (create if missing)."""
    json_path = audio_path.with_suffix(".json")
    data: dict = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data["keyscale"] = key_name
    data["key_confidence"] = round(confidence, 2)
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
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument(
        "--minor-bias",
        type=float,
        default=0.05,
        help="Additive bias toward minor keys (Vinahouse ~90%% minor). Default 0.05.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in args.exts])
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Detecting key from {len(files)} files.")
    print(f"Minor bias: {args.minor_bias}\n")

    results: list[dict] = []
    t_start = time.time()
    for i, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
            key, conf, mode = detect_key(y, sr, minor_bias=args.minor_bias)
        except Exception as e:
            print(f"[WARN] Failed {path.name}: {e}", file=sys.stderr)
            continue

        results.append(
            {
                "filename": path.name,
                "key": key,
                "mode": mode,
                "confidence": round(conf, 3),
            }
        )

        if args.write_json:
            update_json(path, key, conf)

        elapsed = time.time() - t0
        conf_label = "HIGH" if conf > 0.5 else "MED" if conf > 0.3 else "LOW"
        print(
            f"[{i:3d}/{len(files)}] "
            f"key={key:12s} conf={conf:.3f} [{conf_label:4s}]  "
            f"{path.name[:50]}  ({elapsed:.1f}s)"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - t_start
    from collections import Counter

    key_counts = Counter(r["key"] for r in results)
    mode_counts = Counter(r["mode"] for r in results)
    top_keys = key_counts.most_common(5)
    n_low = sum(1 for r in results if r["confidence"] < 0.3)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:      {len(results)}")
    print(f"Mode distribution: major={mode_counts.get('major', 0)}, minor={mode_counts.get('minor', 0)}")
    print(f"Top 5 keys:")
    for k, c in top_keys:
        pct = c * 100 / len(results)
        print(f"  {k:15s}  {c:3d}  ({pct:.1f}%)")
    print(f"Low confidence:   {n_low}  (may need manual verify)")
    print(f"Total time:       {total_time:.1f}s")
    print(f"CSV:              {args.out_csv}")
    if args.write_json:
        print(f"JSON updated with 'keyscale' field for each clip")
    print()
    print("Next step: generate captions (07_captions.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
