"""Parse existing captions to classify kick presence.

Sets `has_kick` (bool) in per-clip JSON metadata. Reads existing caption text
to determine whether the clip has a kick drum pattern. Later, `08_finalize.py`
uses this flag to selectively apply the 'vinahouse' trigger word ONLY to
clips WITH kick, so the model learns:
    "vinahouse" -> kick pattern activate
instead of the current (broken) association:
    "vinahouse" -> anything (breakdown / verse / drop)

Detection rules (based on descriptor vocabulary from 07b_captions_audio.py):

    HAS KICK (has_kick=True):
        - "off-beat kick" / "offbeat kick"
        - "four-on-the-floor" / "4-on-the-floor"
        - "driving kick" / "steady kick" / "pounding kick" / "pumping kick"
        - "kick pattern" (positive, e.g. "off-beat kick pattern reintroducing")

    NO KICK (has_kick=False):
        - "absent kick" / "absent kicks"
        - "no kick" / "without kick"
        - "kicks missing" / "kicks absent"

    AMBIGUOUS: fallback = False (safer, aligns with default breakdown behavior)

Usage:
    .venv/bin/python datasets/vinahouse/scripts/12_relabel_kick.py \\
        --input-dir datasets/vinahouse/phase_b/audio_clean

    # Dry-run to preview counts without modifying JSONs
    .venv/bin/python datasets/vinahouse/scripts/12_relabel_kick.py \\
        --input-dir datasets/vinahouse/phase_b/audio_clean --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


HAS_KICK_PATTERNS = [
    r"off-?beat kicks?",
    r"four[- ]on[- ](?:the[- ])?floor",
    r"4[- ]on[- ](?:the[- ])?floor",
    r"driving kicks?",
    r"steady kicks?",
    r"pounding kicks?",
    r"pumping kicks?",
    r"punchy kicks?",
    r"heavy kicks?",
    r"sidechained kicks?",
    r"kicks? on beats?",
    r"kicks? pattern reintroducing",
    r"kicks? pattern (?:enter|kick|drop)",
    r"kick drop",
    r"^drop\b",
    r"\bdrop with .*kicks?",
]

NO_KICK_PATTERNS = [
    r"absent kicks?",
    r"no kicks?",
    r"without kicks?",
    r"kicks? absent",
    r"kicks? missing",
    r"kicks? drop out",
    r"kicks? drops? out",
    r"missing kicks?",
]


def classify_kick(caption: str) -> tuple[bool, str]:
    """Return (has_kick, reason).

    Priority: HAS_KICK patterns beat NO_KICK. This handles edge cases like
        "Breakdown with absent kick, followed by off-beat kick pattern reintroducing at 0:32"
    where the drop reintroduces kick after a breakdown intro.
    """
    text = caption.lower()

    for pat in HAS_KICK_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return True, f"HAS_KICK matched: {m.group(0)!r}"

    for pat in NO_KICK_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return False, f"NO_KICK matched: {m.group(0)!r}"

    return False, "no explicit kick descriptor (default: no kick)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Folder containing per-clip .json files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print classification but do not modify JSONs")
    ap.add_argument("--verbose", action="store_true",
                    help="Print classification for every file")
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([f for f in args.input_dir.iterdir() if f.suffix == ".json"])
    if not files:
        print(f"[ERROR] No .json files in {args.input_dir}", file=sys.stderr)
        return 1

    counts = Counter()
    ambiguous_files: list[str] = []
    updated = 0
    skipped_no_caption = 0

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[SKIP] {jf.name}: invalid JSON", file=sys.stderr)
            continue

        caption = (data.get("caption") or "").strip()
        if not caption:
            skipped_no_caption += 1
            continue

        has_kick, reason = classify_kick(caption)
        counts["has_kick" if has_kick else "no_kick"] += 1

        if "no explicit" in reason:
            counts["ambiguous"] += 1
            ambiguous_files.append(jf.name)

        if args.verbose:
            label = "KICK   " if has_kick else "NO-KICK"
            print(f"[{label}] {jf.name}  ({reason})")

        if not args.dry_run:
            data["has_kick"] = has_kick
            data["kick_reason"] = reason
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1

    total = counts["has_kick"] + counts["no_kick"]
    print("\n" + "=" * 70)
    print("KICK CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total captioned files:    {total}")
    print(f"Skipped (no caption):     {skipped_no_caption}")
    print()
    print(f"  HAS_KICK (trigger=on):   {counts['has_kick']:4d}  ({counts['has_kick']/total*100:.1f}%)")
    print(f"  NO_KICK  (trigger=off):  {counts['no_kick']:4d}  ({counts['no_kick']/total*100:.1f}%)")
    print()
    print(f"  Ambiguous (defaulted to NO_KICK): {counts['ambiguous']}")

    if ambiguous_files and args.verbose:
        print("\nAmbiguous files (first 10):")
        for f in ambiguous_files[:10]:
            print(f"  - {f}")

    if args.dry_run:
        print("\n[DRY RUN] No JSON files modified.")
    else:
        print(f"\n[OK] Updated {updated} JSON files with 'has_kick' field.")
        print("\nNext step: re-run 08_finalize.py to build dataset.json with per-sample custom_tag.")

    if counts["has_kick"] < 50:
        print(f"\n[!] Only {counts['has_kick']} clips have kick. Training set will be small.")
        print(f"[!] Consider re-captioning if quality is poor after retrain.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
