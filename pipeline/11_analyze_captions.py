"""Diagnostic: analyze how captions describe kick drums.

Vinahouse should have driving 4-on-the-floor / off-beat kicks in DROP sections.
If dataset captions are dominated by 'absent kicks', model will learn:
    vinahouse => no kick (breakdown vibe)
which explains why generated audio has no kick.

Usage:
    .venv/bin/python datasets/vinahouse/scripts/11_analyze_captions.py \
        --input-dir datasets/vinahouse/phase_b/audio_clean
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "no_kick": [
        r"absent kick",
        r"without kick",
        r"no kick",
        r"kicks? absent",
        r"kicks? missing",
    ],
    "off_beat_kick": [
        r"off-?beat kick",
        r"offbeat kick",
    ],
    "four_on_floor": [
        r"four[- ]on[- ]the[- ]floor",
        r"4[- ]on[- ]the[- ]floor",
        r"steady kick",
        r"driving kick",
        r"pounding kick",
        r"pumping kick",
    ],
    "any_kick_mention": [
        r"kick",
    ],
}


def classify(text: str) -> set[str]:
    hits = set()
    for cat, patterns in CATEGORIES.items():
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                hits.add(cat)
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--show-samples", type=int, default=3,
                    help="Print N sample captions per category")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files in {args.input_dir}")
        return 1

    total = 0
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {c: [] for c in CATEGORIES}

    section_words = Counter()
    section_pat = re.compile(
        r"^\s*(breakdown|drop|verse|chorus|bridge|intro|outro|build[- ]?up)",
        re.IGNORECASE,
    )

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cap = (data.get("caption") or "").strip()
        if not cap:
            continue
        total += 1

        m = section_pat.match(cap)
        if m:
            section_words[m.group(1).lower().replace(" ", "").replace("-", "")] += 1

        hits = classify(cap)
        for h in hits:
            counts[h] += 1
            if len(samples[h]) < args.show_samples:
                samples[h].append(f"{jf.name}: {cap[:120]}...")

    print("=" * 70)
    print(f"CAPTION KICK-DRUM ANALYSIS  ({total} captions)")
    print("=" * 70)

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "0%"

    print(f"\nKick-drum descriptions:")
    print(f"  Any 'kick' mention:      {counts['any_kick_mention']:4d}  ({pct(counts['any_kick_mention'])})")
    print(f"  ABSENT kick (no drum):   {counts['no_kick']:4d}  ({pct(counts['no_kick'])})  <- these teach model 'vinahouse = no kick'")
    print(f"  OFF-BEAT kick:           {counts['off_beat_kick']:4d}  ({pct(counts['off_beat_kick'])})")
    print(f"  4-on-floor/driving:      {counts['four_on_floor']:4d}  ({pct(counts['four_on_floor'])})  <- Vinahouse drop signature")

    if counts["no_kick"] > 0 and counts["four_on_floor"] == 0:
        ratio = counts["no_kick"] / max(counts["off_beat_kick"] + counts["four_on_floor"], 1)
        print(f"\n[WARNING] Absent-kick captions dominate. Ratio no_kick / (off_beat + 4x4) = {ratio:.1f}x")
        print("[WARNING] Model may have learned: 'vinahouse trigger = suppress kick'")

    print(f"\nSection type distribution (first word):")
    for k, v in section_words.most_common():
        print(f"  {k:15s}: {v:4d}  ({pct(v)})")

    for cat in ["no_kick", "off_beat_kick", "four_on_floor"]:
        print(f"\n--- Samples ({cat}) ---")
        if samples[cat]:
            for s in samples[cat]:
                print(f"  {s}")
        else:
            print("  (no examples in dataset)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
