"""Generate English captions for Vinahouse clips using OpenAI Chat API.

Reads {stem}.json for each audio clip to get BPM, key, and language, then asks
GPT-4o-mini to produce a rich English caption suitable for ACE-Step LoRA training.

The caption is written back to two places:
    - Merged into {stem}.json (field: 'caption')
    - Saved as {stem}.caption.txt (ACE-Step preferred format)

Usage:
    export OPENAI_API_KEY="sk-proj-..."

    .venv/bin/python datasets/vinahouse/scripts/07_captions.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --model gpt-4o-mini \\
        --dry-run              # preview 3 captions without spending credits

    .venv/bin/python datasets/vinahouse/scripts/07_captions.py \\
        --input-dir datasets/vinahouse/audio_clean

Cost estimate: ~$0.05 total for 152 clips with gpt-4o-mini.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai SDK not installed. Run: .venv/bin/pip install openai", file=sys.stderr)
    raise


SYSTEM_PROMPT = """You are a professional music producer and audio annotator writing captions \
for a Vinahouse dataset used to train a music generation model. Vinahouse is a Vietnamese subgenre \
of hard bounce EDM, characterized by a fast punchy off-beat kick pattern, wobble/pluck bass, \
arpeggiated synth leads, and often Vietnamese vocal chops or full sung vocals.

Your captions describe what the listener hears during a 60-second clip. Focus on:
    - Kick/bass pattern (off-beat, driving, bouncy, sidechained)
    - Synth character (arpeggio, pluck, stab, saw lead, pad)
    - Vocal presence (Vietnamese female/male vocal, vocal chop, chopped hook, none/instrumental)
    - Energy level (energetic, euphoric, dark, uplifting)
    - Mix atmosphere (bright, gritty, festival, club, radio-friendly)

Write in fluent English. Keep captions 30-60 words. Do NOT include the raw BPM/key values in the caption; \
those live in separate metadata fields. Do NOT prefix with "This clip..." or "The track..."; start directly \
with descriptive adjectives.

Return ONLY the caption text, no quotes, no formatting, no explanations."""


USER_TEMPLATE = """Write a caption for a Vinahouse clip with these hints:

BPM: {bpm}
Key: {keyscale}
Filename slug: {slug}
Estimated content: {content_hint}

Produce one caption now."""


def infer_content_hint(slug: str) -> str:
    """Heuristic hint from filename patterns."""
    s = slug.lower()
    if "chunk" in s or "mix" in s:
        return "Segment from a DJ mix; likely full drop with sustained energy, possible vocal chops."
    return "Standalone remix of a Vietnamese pop song; likely contains sung vocals and Vinahouse remix elements."


def build_prompt(meta: dict, filename: str) -> tuple[str, str]:
    slug = Path(filename).stem
    user = USER_TEMPLATE.format(
        bpm=meta.get("bpm", "unknown"),
        keyscale=meta.get("keyscale", "unknown"),
        slug=slug,
        content_hint=infer_content_hint(slug),
    )
    return SYSTEM_PROMPT, user


def call_openai(client: OpenAI, model: str, system: str, user: str, temperature: float) -> str:
    """Call the API with retry."""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=180,
            )
            content = resp.choices[0].message.content or ""
            return content.strip().strip('"').strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[WARN] API error (attempt {attempt+1}): {e}. Retry in {wait}s.", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("OpenAI API failed after 4 retries")


def write_outputs(audio_path: Path, caption: str) -> None:
    """Merge caption into JSON and also save a .caption.txt file."""
    json_path = audio_path.with_suffix(".json")
    data: dict = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data["caption"] = caption

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_path = audio_path.with_suffix(".caption.txt")
    txt_path.write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg"],
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.9, help="Higher = more diverse")
    parser.add_argument("--dry-run", action="store_true", help="Print 3 sample captions, don't write files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have a caption")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY env var not set.", file=sys.stderr)
        print('        Run: export OPENAI_API_KEY="sk-..."', file=sys.stderr)
        return 1

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in args.exts])
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    if args.dry_run:
        print("[DRY RUN] Generating 3 sample captions...\n")
        sample = random.sample(files, min(3, len(files)))
        for path in sample:
            meta = {}
            json_path = path.with_suffix(".json")
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            system, user = build_prompt(meta, path.name)
            caption = call_openai(client, args.model, system, user, args.temperature)
            print("-" * 70)
            print(f"File:  {path.name}")
            print(f"BPM:   {meta.get('bpm', '?')}")
            print(f"Key:   {meta.get('keyscale', '?')}")
            print(f"Caption:\n  {caption}\n")
        print("Dry run complete. Remove --dry-run to process all files.")
        return 0

    print(f"Generating captions for {len(files)} files with {args.model}")
    print(f"Temperature: {args.temperature} (higher = more diverse)\n")

    t_start = time.time()
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    est_cost = 0.0

    for i, path in enumerate(files, start=1):
        meta = {}
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        if args.skip_existing and meta.get("caption"):
            print(f"[{i:3d}/{len(files)}] SKIP (existing)  {path.name}")
            n_skipped += 1
            continue

        try:
            system, user = build_prompt(meta, path.name)
            caption = call_openai(client, args.model, system, user, args.temperature)
            write_outputs(path, caption)
            n_ok += 1
            est_cost += 0.0003
            preview = caption[:80] + ("..." if len(caption) > 80 else "")
            print(f"[{i:3d}/{len(files)}] OK  {path.name[:35]:35s}  {preview}")
        except Exception as e:
            print(f"[{i:3d}/{len(files)}] FAIL {path.name}: {e}", file=sys.stderr)
            n_failed += 1

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:     {len(files)}")
    print(f"  Succeeded:     {n_ok}")
    print(f"  Skipped:       {n_skipped}")
    print(f"  Failed:        {n_failed}")
    print(f"Estimated cost:  ~${est_cost:.3f} USD")
    print(f"Total time:      {total_time:.1f}s")
    print()
    print("Next step: combine remaining metadata into final JSON (08_finalize.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
