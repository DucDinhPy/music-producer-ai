"""Auto-audit audio clips to prioritize manual review.

Computes 4 features per clip and combines into a cleanliness score [0, 1]:

    1. rms_low_ratio     Fraction of time the clip is quiet (breakdown / vocal-only / FX loop).
    2. bpm_split_diff    |BPM(first_half) - BPM(second_half)|. Large diff => transition.
    3. onset_density_cv  Coefficient of variation of onset density across windows. High => uneven rhythm.
    4. onset_density_mean  Mean beats/sec. Very low => breakdown / a cappella / FX loop.

Verdict:
    CLEAN       score >= 0.75  -> highly likely usable, skip audit
    SUSPICIOUS  0.50 <= score < 0.75  -> listen briefly
    DIRTY       score < 0.50  -> almost certainly bad, likely delete

Usage:
    .venv/bin/python datasets/vinahouse/scripts/04_audit_audio.py \\
        --input-dir datasets/vinahouse/audio_from_mix \\
        --out-csv  datasets/vinahouse/audit_report.csv

Deps: librosa, numpy, soundfile (all present in ACE-Step's .venv).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError:
    print("[ERROR] librosa not available in this Python env.", file=sys.stderr)
    print("        Run: .venv/bin/pip install librosa", file=sys.stderr)
    raise


TARGET_SR = 22050  # analysis rate; downsampled for speed. VAE later uses 48kHz.
HOP_LENGTH = 512
FRAME_LENGTH = 2048

VERDICT_CLEAN = "CLEAN"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
VERDICT_DIRTY = "DIRTY"


def compute_rms_low_ratio(y: np.ndarray) -> float:
    """Fraction of time where RMS < 30% of median RMS.

    Clean Vinahouse drop stays loud throughout. Breakdown / vocal-only sections
    have periods below this threshold.
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    if rms.size == 0:
        return 1.0
    median = np.median(rms)
    if median == 0:
        return 1.0
    threshold = 0.3 * median
    return float(np.mean(rms < threshold))


def compute_bpm_split(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    """BPM on first half vs second half. Returns (bpm_first, bpm_second, abs_diff)."""
    mid = len(y) // 2
    y_first = y[:mid]
    y_second = y[mid:]

    def _tempo(chunk: np.ndarray) -> float:
        if chunk.size < sr:  # < 1 sec
            return 0.0
        try:
            tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr, hop_length=HOP_LENGTH)
            return float(tempo) if np.isscalar(tempo) else float(tempo[0])
        except Exception:
            return 0.0

    bpm1 = _tempo(y_first)
    bpm2 = _tempo(y_second)
    return bpm1, bpm2, abs(bpm1 - bpm2)


def compute_onset_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Onset density mean and coefficient of variation across 6-second windows.

    Returns:
        (mean_onsets_per_sec, cv)
        cv = std / mean; high => uneven rhythm (transition / breakdown mid-clip)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units="time"
    )
    duration = librosa.get_duration(y=y, sr=sr)
    if duration <= 0:
        return 0.0, 0.0

    total_density = len(onsets) / duration

    window_sec = 6.0
    n_windows = int(duration // window_sec)
    if n_windows < 2:
        return total_density, 0.0

    densities = []
    for w in range(n_windows):
        start = w * window_sec
        end = start + window_sec
        count = int(np.sum((onsets >= start) & (onsets < end)))
        densities.append(count / window_sec)

    densities_np = np.array(densities)
    mean = float(np.mean(densities_np))
    if mean == 0:
        return total_density, 0.0
    cv = float(np.std(densities_np) / mean)
    return mean, cv


def score_clip(
    rms_low_ratio: float,
    bpm_diff: float,
    onset_density_mean: float,
    onset_density_cv: float,
) -> tuple[float, list[str]]:
    """Combine features into cleanliness score [0, 1]. Also return reasons for penalties."""
    reasons: list[str] = []
    penalty = 0.0

    # 1) Silence penalty: >15% of clip is quiet
    if rms_low_ratio > 0.35:
        penalty += 0.40
        reasons.append(f"very quiet {rms_low_ratio*100:.0f}%")
    elif rms_low_ratio > 0.20:
        penalty += 0.20
        reasons.append(f"quiet {rms_low_ratio*100:.0f}%")
    elif rms_low_ratio > 0.10:
        penalty += 0.08
        reasons.append(f"slight-quiet {rms_low_ratio*100:.0f}%")

    # 2) Transition penalty: BPM differs a lot between halves
    if bpm_diff > 8:
        penalty += 0.40
        reasons.append(f"BPM shift {bpm_diff:.1f}")
    elif bpm_diff > 4:
        penalty += 0.20
        reasons.append(f"BPM drift {bpm_diff:.1f}")

    # 3) Low onset density: breakdown / a cappella / FX loop
    if onset_density_mean < 1.5:
        penalty += 0.35
        reasons.append(f"low activity {onset_density_mean:.1f}/s")
    elif onset_density_mean < 3.0:
        penalty += 0.10
        reasons.append(f"mild activity {onset_density_mean:.1f}/s")

    # 4) Uneven rhythm (transitions inside clip)
    if onset_density_cv > 0.60:
        penalty += 0.25
        reasons.append(f"uneven CV {onset_density_cv:.2f}")
    elif onset_density_cv > 0.40:
        penalty += 0.10
        reasons.append(f"slight-uneven CV {onset_density_cv:.2f}")

    score = max(0.0, 1.0 - penalty)
    return score, reasons


def verdict_for(score: float) -> str:
    if score >= 0.75:
        return VERDICT_CLEAN
    if score >= 0.50:
        return VERDICT_SUSPICIOUS
    return VERDICT_DIRTY


def analyze_file(path: Path) -> Optional[dict]:
    try:
        y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load {path.name}: {e}", file=sys.stderr)
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    rms_low = compute_rms_low_ratio(y)
    bpm1, bpm2, bpm_diff = compute_bpm_split(y, sr)
    onset_mean, onset_cv = compute_onset_stats(y, sr)

    score, reasons = score_clip(rms_low, bpm_diff, onset_mean, onset_cv)
    return {
        "filename": path.name,
        "duration_sec": round(duration, 1),
        "rms_low_ratio": round(rms_low, 3),
        "bpm_first": round(bpm1, 1),
        "bpm_second": round(bpm2, 1),
        "bpm_diff": round(bpm_diff, 1),
        "onset_density": round(onset_mean, 2),
        "onset_cv": round(onset_cv, 2),
        "score": round(score, 3),
        "verdict": verdict_for(score),
        "reasons": "; ".join(reasons) if reasons else "clean",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more folders containing audio clips (space-separated)",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg"],
        help="File extensions to include",
    )
    parser.add_argument(
        "--copy-clean-to",
        type=Path,
        default=None,
        help="If set, copy all CLEAN (and optionally SUSPICIOUS) files to this folder",
    )
    parser.add_argument(
        "--include-suspicious",
        action="store_true",
        help="Also copy SUSPICIOUS files (score 0.50-0.75) when --copy-clean-to is set",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Custom score threshold for copying (overrides CLEAN/SUSPICIOUS logic)",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for d in args.input_dir:
        if not d.exists():
            print(f"[WARN] Input dir not found (skipping): {d}", file=sys.stderr)
            continue
        found = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in args.exts]
        print(f"[INFO] Found {len(found):3d} files in {d}")
        files.extend(found)

    files.sort()
    if not files:
        print("[ERROR] No audio files across the given input dirs", file=sys.stderr)
        return 1

    print(f"\nAuditing {len(files)} files total.")
    print("This uses librosa (CPU-only), ~2-3 sec per 60s clip. Please wait.\n")

    results: list[dict] = []
    t_start = time.time()
    for i, path in enumerate(files, start=1):
        t0 = time.time()
        row = analyze_file(path)
        if row is None:
            continue
        row["source_dir"] = str(path.parent)
        row["source_path"] = str(path)
        results.append(row)
        elapsed = time.time() - t0
        print(
            f"[{i:3d}/{len(files)}] {row['verdict']:11s} "
            f"score={row['score']:.2f}  {path.name[:50]:50s}  "
            f"({elapsed:.1f}s)  reasons: {row['reasons']}"
        )

    if not results:
        print("[ERROR] No files analyzed", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r["score"], reverse=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n_clean = sum(1 for r in results if r["verdict"] == VERDICT_CLEAN)
    n_susp = sum(1 for r in results if r["verdict"] == VERDICT_SUSPICIOUS)
    n_dirty = sum(1 for r in results if r["verdict"] == VERDICT_DIRTY)
    total_time = time.time() - t_start

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total analyzed:  {len(results)}")
    print(f"  CLEAN       (score >= 0.75):  {n_clean:3d}   -> can skip audit")
    print(f"  SUSPICIOUS  (0.50 - 0.75)  :  {n_susp:3d}   -> listen briefly")
    print(f"  DIRTY       (score < 0.50) :  {n_dirty:3d}   -> likely delete")
    print(f"Total time: {total_time:.1f}s")
    print()
    print(f"Report written to: {args.out_csv}")

    if args.copy_clean_to is not None:
        import shutil

        target = args.copy_clean_to
        target.mkdir(parents=True, exist_ok=True)

        if args.min_score is not None:
            def _should_copy(row: dict) -> bool:
                return row["score"] >= args.min_score
            filter_desc = f"score >= {args.min_score}"
        elif args.include_suspicious:
            def _should_copy(row: dict) -> bool:
                return row["verdict"] in (VERDICT_CLEAN, VERDICT_SUSPICIOUS)
            filter_desc = "CLEAN + SUSPICIOUS"
        else:
            def _should_copy(row: dict) -> bool:
                return row["verdict"] == VERDICT_CLEAN
            filter_desc = "CLEAN only"

        to_copy = [r for r in results if _should_copy(r)]
        print()
        print("=" * 70)
        print(f"COPY {len(to_copy)} files ({filter_desc}) -> {target}")
        print("=" * 70)

        copied = 0
        skipped_dup = 0
        for row in to_copy:
            src = Path(row["source_path"])
            dst = target / src.name
            if dst.exists():
                skipped_dup += 1
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"[WARN] Failed to copy {src.name}: {e}", file=sys.stderr)

        print(f"Copied: {copied}    Skipped (already exists): {skipped_dup}")

    print()
    print("Next step:")
    print(f"  1. Open {args.out_csv.name} in Excel / VS Code")
    print("  2. Sort by 'score' (already sorted descending)")
    print("  3. Focus manual audit on SUSPICIOUS files (~1 min each)")
    print("  4. Trust CLEAN files (spot-check a few)")
    print("  5. Delete or skip DIRTY files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
