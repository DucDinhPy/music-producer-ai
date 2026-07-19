"""Audit Whisper transcriptions to detect hallucinations and reclassify as instrumental.

Whisper's Vietnamese model was trained on massive YouTube subtitle data. When it
encounters audio WITHOUT clear speech (instrumental music, chopped vocal samples,
background music with unclear singing), it hallucinates by regurgitating common
training phrases like "Hãy subscribe cho kênh...", "Cảm ơn các bạn đã xem video", etc.

This script:
  1. Detects known hallucination patterns in lyrics
  2. Detects other suspicious signals (heavy repetition, extreme brevity)
  3. Reclassifies hallucinated clips as instrumental (is_instrumental=true)
  4. Optionally copies suspicious clips to a --rewhisper-dir for manual re-processing

Usage:
    # Dry-run (audit only, no changes)
    .venv/bin/python datasets/vinahouse/scripts/10_audit_lyric.py \\
        --input-dir datasets/vinahouse/phase_b/audio_clean \\
        --dry-run

    # Auto-fix: rewrite hallucinated JSONs as instrumental
    .venv/bin/python datasets/vinahouse/scripts/10_audit_lyric.py \\
        --input-dir datasets/vinahouse/phase_b/audio_clean \\
        --reclassify-as-instrumental

    # Also copy suspicious clips to rewhisper folder for manual review
    .venv/bin/python datasets/vinahouse/scripts/10_audit_lyric.py \\
        --input-dir datasets/vinahouse/phase_b/audio_clean \\
        --reclassify-as-instrumental \\
        --rewhisper-dir datasets/vinahouse/phase_b/rewhisper
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


HALLUCINATION_PATTERNS = [
    r"subscribe",
    r"ghi[eê]n m[ìi] g[õo]",
    r"c[aả]m [ơo]n c[aá]c b[aạ]n \u0111[aã] xem",
    r"c[aả]m [ơo]n c[aá]c b[aạ]n \u0111[aã] theo d[oõ]i",
    r"c[aả]m [ơo]n qu[ýyy] v[ị]",
    r"\u0111[ừùu]ng qu[êe]n like",
    r"like v[àa] share",
    r"like v[àa] subscribe",
    r"x[iíi]n ch[aàà]o c[aá]c b[aạ]n",
    r"ch[aà]o m[ừừu]ng c[aá]c b[aạ]n",
    r"h[eẹh][nn] g[ăa]p l[aạ]i",
    r"h[aẹà]y \u0111[aă]ng k[iyýy] k[eê]nh",
    r"\u0111[aă]ng k[iyýy] k[eê]nh",
    r"ph\u1ee5 \u0111\u1ec1 vi\u1ec7t",
    r"ph\u1ee5 \u0111\u1ec1 b\u1edfi",
    r"vtv[123]",
    r"htv[0-9]?",
    r"vietsub",
    r"vietsubs",
    r"transcribed by",
    r"amara\.org",
    r"copyright",
    r"\u1edcn gi\u1eddi c\u1eadu \u0111\u00e2y r\u1ed3i",
    r"video n\u00e0y",
    r"trong video",
    r"ki\u00eanh youtube",
    r"m\u00ecgo",
    r"my dear vietnam",
]

REPETITION_THRESHOLD = 0.6

MIN_UNIQUE_WORDS = 4


def find_hallucination_hits(text: str) -> list[str]:
    """Return list of matched hallucination phrases."""
    text_lower = text.lower()
    hits = []
    for pat in HALLUCINATION_PATTERNS:
        for m in re.finditer(pat, text_lower, re.IGNORECASE):
            hits.append(m.group(0))
    return hits


def repetition_ratio(text: str) -> tuple[float, int]:
    """Ratio of most-common word to total words. Returns (ratio, unique_word_count).

    High ratio (> 0.6) suggests degenerate output like:
       "cảm ơn cảm ơn cảm ơn cảm ơn..."
    """
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    if len(tokens) < 3:
        return 0.0, len(set(tokens))
    counter = Counter(tokens)
    top_word, top_count = counter.most_common(1)[0]
    ratio = top_count / len(tokens)
    return ratio, len(set(tokens))


def is_looped_repeat(text: str) -> bool:
    """Detect exact substring repetition: 'ABC ABC ABC' or '[00:00]X\\n[00:30]X'.

    Whisper sometimes duplicates the same phrase across timestamps.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines_no_ts = [re.sub(r"\[\d+:\d+\]", "", l).strip() for l in lines]
    if len(lines_no_ts) < 2:
        return False
    unique_lines = set(lines_no_ts)
    if len(unique_lines) == 1:
        return True
    if len(unique_lines) <= 2 and len(lines_no_ts) >= 3:
        return True
    return False


def audit_clip(data: dict) -> tuple[str, list[str]]:
    """Return (verdict, reasons) where verdict is CLEAN | SUSPICIOUS | HALLUCINATION."""
    if data.get("is_instrumental"):
        return "CLEAN", ["already instrumental"]

    raw = (data.get("raw_lyrics") or "").strip()
    ts_lyrics = (data.get("lyrics") or "").strip()

    if not raw or ts_lyrics == "[Instrumental]":
        return "CLEAN", ["empty vocal"]

    reasons = []

    hits = find_hallucination_hits(raw)
    if hits:
        reasons.append(f"YouTube-phrase: {hits[:3]}")
        return "HALLUCINATION", reasons

    if is_looped_repeat(ts_lyrics):
        reasons.append("looped repeat across timestamps")
        return "HALLUCINATION", reasons

    ratio, uniq = repetition_ratio(raw)
    if ratio > REPETITION_THRESHOLD:
        reasons.append(f"single-word dominance {ratio:.2f}")
        return "HALLUCINATION", reasons

    if uniq < MIN_UNIQUE_WORDS:
        reasons.append(f"too few unique words ({uniq})")
        return "SUSPICIOUS", reasons

    return "CLEAN", ["ok"]


def reclassify_as_instrumental(json_path: Path) -> None:
    """Update JSON file to mark clip as instrumental (in-place)."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    data["is_instrumental"] = True
    data["lyrics"] = "[Instrumental]"
    data["raw_lyrics"] = ""
    data["vocal_language"] = "unknown"
    data["was_hallucinated"] = True
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder containing per-clip .json files")
    parser.add_argument("--reclassify-as-instrumental", action="store_true",
                        help="Rewrite hallucinated JSONs as instrumental (safe fix)")
    parser.add_argument("--rewhisper-dir", type=Path, default=None,
                        help="Optional: also copy suspicious clips (JSON + WAV) here for manual review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print audit report, do not modify or copy anything")
    parser.add_argument("--verbose", action="store_true",
                        help="Show every clip's verdict (not just HALLUCINATION/SUSPICIOUS)")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([f for f in args.input_dir.iterdir() if f.suffix == ".json"])
    if not files:
        print(f"[ERROR] No .json files in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Auditing {len(files)} JSON files in {args.input_dir}\n")

    verdicts = Counter()
    hallucinated: list[Path] = []
    suspicious: list[Path] = []

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[SKIP] {jf.name}: invalid JSON", file=sys.stderr)
            continue

        verdict, reasons = audit_clip(data)
        verdicts[verdict] += 1

        if verdict == "HALLUCINATION":
            hallucinated.append(jf)
            print(f"[HALLUC ] {jf.name}: {', '.join(reasons)}")
            preview = (data.get("raw_lyrics") or "")[:60]
            print(f"          -> \"{preview}...\"")
        elif verdict == "SUSPICIOUS":
            suspicious.append(jf)
            print(f"[SUSPECT] {jf.name}: {', '.join(reasons)}")
            preview = (data.get("raw_lyrics") or "")[:60]
            print(f"          -> \"{preview}...\"")
        elif args.verbose:
            print(f"[CLEAN  ] {jf.name}")

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total JSONs:       {len(files)}")
    print(f"  CLEAN:           {verdicts['CLEAN']}")
    print(f"  HALLUCINATION:   {verdicts['HALLUCINATION']}  <-- fix by reclassifying instrumental")
    print(f"  SUSPICIOUS:      {verdicts['SUSPICIOUS']}      <-- manual review needed")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return 0

    if args.reclassify_as_instrumental and hallucinated:
        print(f"\n[FIX] Reclassifying {len(hallucinated)} hallucinated clips as instrumental...")
        for jf in hallucinated:
            reclassify_as_instrumental(jf)
        print(f"[FIX] Done. All hallucinated clips now: is_instrumental=true, lyrics='[Instrumental]', was_hallucinated=true")

    if args.rewhisper_dir:
        args.rewhisper_dir.mkdir(parents=True, exist_ok=True)
        candidates = hallucinated + suspicious
        print(f"\n[COPY] Copying {len(candidates)} suspicious/hallucinated clips to {args.rewhisper_dir}")
        n_copied = 0
        for jf in candidates:
            audio = jf.with_suffix(".wav")
            shutil.copy2(jf, args.rewhisper_dir / jf.name)
            if audio.exists():
                shutil.copy2(audio, args.rewhisper_dir / audio.name)
                n_copied += 1
            else:
                print(f"[WARN] audio missing: {audio}", file=sys.stderr)
        print(f"[COPY] Copied {n_copied} clip pairs (JSON + WAV)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
