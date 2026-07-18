"""Generate audio-grounded captions using OpenAI's gpt-audio-1.5.

Unlike the text-only variant (07_captions.py), this script sends the actual
audio waveform to the model so it can describe the specific sonic content of
each clip (instruments, vocals, energy dynamics, transitions).

Cost warning:
    Audio input tokens are billed at a substantially higher rate than text.
    Roughly $0.20-0.35 USD per clip depending on OpenAI's pricing tier.
    Use --limit and --dry-run to validate before spending big.

Pipeline per clip:
    1. Load audio, downsample to 22050 Hz mono (reduces payload ~4x)
    2. Encode WAV bytes as base64
    3. Send Chat Completions request with 'input_audio' content
    4. Extract text caption, write to {stem}.caption.txt and merge into JSON

Usage:
    export OPENAI_API_KEY="sk-proj-..."

    # Test 3 clips first
    .venv/bin/python datasets/vinahouse/scripts/07b_captions_audio.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --dry-run --limit 3

    # Full run (cost $30-50)
    .venv/bin/python datasets/vinahouse/scripts/07b_captions_audio.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --skip-existing

    # Resume from failure - just re-run with --skip-existing
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path


def normalize_caption(text: str) -> str:
    """Fix common whitespace artifacts from LLM output.

    - Add space after punctuation missing space (",word" -> ", word")
    - Collapse repeated whitespace to single space
    - Strip trailing/leading whitespace and stray quotes
    """
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)
    text = re.sub(r"\b([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


BANNED_PATTERNS = [
    r"\bpolished\b",
    r"\bwell[-\s]balanced\b",
    r"\bclub[-\s]ready\b",
    r"\bfestival[-\s]ready\b",
    r"\bradio[-\s]friendly\b",
    r"\bcatchy\b",
    r"\bdanceable\b",
    r"\benergetic vibe\b",
    r"\buplifting atmosphere\b",
    r"\bemotional atmosphere\b",
    r"\bmelodic atmosphere\b",
    r"\btypical(?:ly)? (?:of )?[Vv]inahouse\b",
    r"\bcrisp high[-\s]end\b",
    r"\bfull[-\s]bodied low end\b",
    r"\bsmooth and \w+",
    r"\bwarm and \w+",
    r"\bbright and \w+",
]


def count_banned_words(caption: str) -> tuple[int, list[str]]:
    """Return (count, list of banned matches) in caption text."""
    hits = []
    for pat in BANNED_PATTERNS:
        for m in re.finditer(pat, caption, re.IGNORECASE):
            hits.append(m.group(0))
    return len(hits), hits

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("[ERROR] librosa/soundfile not installed. Run: .venv/bin/pip install librosa soundfile", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai SDK not installed. Run: .venv/bin/pip install openai", file=sys.stderr)
    raise


DOWNSAMPLE_SR = 22050

SYSTEM_PROMPT = """You are annotating Vinahouse EDM clips (~135-142 BPM, Vietnamese, minor keys) for AI training.

============================================================
HARD BANNED WORDS — using ANY of these = REJECTED caption:
============================================================
"polished"          "smooth and X"        "warm and X"        "well-balanced"
"bright and X"      "club-ready"          "festival-ready"    "radio-friendly"
"catchy"            "danceable"           "energetic vibe"    "uplifting atmosphere"
"emotional atmosphere"  "melodic atmosphere"  "typical Vinahouse"
"crisp high-end"    "full-bodied low end" "shimmer" (unless describing specific FX)

You MUST use CONCRETE, MEASURABLE descriptors instead:
- Instead of "polished" → "dry", "wet with reverb", "compressed hard", "loud master", "saturated"
- Instead of "warm" → "analog-sounding", "low-passed", "vinyl-cracked", "muffled"
- Instead of "smooth" → "long attack", "no transients", "sidechained pumping", "legato"
- Instead of "atmospheric" → "reverbed pads", "reverse cymbal", "background drone", "delay tail"

============================================================
REQUIRED CONTENT (in order in caption):
============================================================
1. STRUCTURAL ROLE (first 2-3 words): verse | buildup | drop | breakdown | transition | outro
   If transition, add timestamp (e.g. "buildup transitions to drop at 0:24")

2. KICK: state pattern explicitly. Options:
   - "off-beat kick" (Vinahouse signature: kick on beats 2 and 4 or 1&3 with syncopation)
   - "four-on-floor" (kick on every beat)
   - "sidechained kick" (audible pump on other elements)
   - "absent kick" (breakdown/verse without kick)
   - "distorted kick" (clipped/saturated)

3. BASS: pluck | wobble | sub | reese | saw | reversed | absent

4. LEAD/SYNTH: name specific timbres — "arpeggio saw", "supersaw pluck", "festival stab", 
   "chopped vocal", "reverse cymbal", "riser", "detuned lead". NOT just "synth".

5. VOCAL — LISTEN VERY CAREFULLY, do NOT default to male:
   - No vocals → "instrumental"
   - Vocals present → describe pitch range:
       * Fundamental > 250 Hz consistently = FEMALE
       * Fundamental < 180 Hz consistently = MALE
       * Between → say "androgynous" or "high male / low female"
   - Style: "sung phrase (Vietnamese)", "chopped vocal hook", "pitched vocal sample", "ad-lib shouts"
   - If in doubt about gender → say "vocal" without gender label

6. TIMBRE — pick 1 or 2 concrete descriptors from:
   dry, wet, saturated, filtered, muted, distorted, clean, gritty, muddy, sharp, dark, lofi, over-compressed

============================================================
FORMAT RULES:
============================================================
- 40-70 words, ONE paragraph
- Start with structural role, e.g. "Drop with off-beat kicks and a wobble bass..."
- NEVER start with: "This clip", "The track", "Audio contains", "In this piece"
- NO BPM/key values (those are separate)
- Return ONLY the caption text, no quotes, no markdown, no labels, no JSON

============================================================
GOLD EXAMPLE (mimic this specificity):
============================================================
"Drop with off-beat sidechained kicks and a punchy pluck bass, layered under a chopped 
Vietnamese vocal hook pitched down a fourth. A detuned saw lead arpeggiates in the right 
channel while a reversed cymbal sweeps into every 8-bar phrase. Overall mix is saturated 
and dry, with the vocal chop sitting forward in the stereo field."

BAD EXAMPLE (DO NOT WRITE LIKE THIS):
"Drop section with driving four-on-floor kicks and a warm pluck bass. Male Vietnamese 
vocals sing over polished pads, creating a smooth and uplifting atmosphere. The mix is 
bright and well-balanced, with an emotional melodic feel typical of Vinahouse."
"""


USER_PROMPT = """Listen to this Vinahouse clip and describe it."""


def audio_to_wav_bytes(audio_path: Path, sr: int = DOWNSAMPLE_SR) -> bytes:
    """Load audio, downsample to mono at target SR, return WAV bytes."""
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def call_audio_api(
    client: OpenAI, model: str, audio_b64: str, temperature: float
) -> tuple[str, dict]:
    """Send audio + text prompt, return (caption, usage_stats)."""
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=250,
            )
            content = normalize_caption(resp.choices[0].message.content or "")
            usage = {}
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                }
            return content, usage
        except Exception as e:
            last_err = e
            wait = 2 ** attempt + 1
            print(f"[WARN] API error (attempt {attempt+1}): {e}. Retry in {wait}s.", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"OpenAI Audio API failed after 4 retries: {last_err}")


def write_outputs(audio_path: Path, caption: str) -> None:
    """Merge caption into JSON and save .caption.txt."""
    json_path = audio_path.with_suffix(".json")
    data: dict = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data["caption"] = caption
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_path = audio_path.with_suffix(".caption.txt")
    txt_path.write_text(caption + "\n", encoding="utf-8")


def has_caption(audio_path: Path) -> bool:
    json_path = audio_path.with_suffix(".json")
    if not json_path.exists():
        return False
    try:
        return bool(json.loads(json_path.read_text(encoding="utf-8")).get("caption"))
    except json.JSONDecodeError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg"],
    )
    parser.add_argument("--model", default="gpt-audio-1.5")
    parser.add_argument("--fallback-model", default="gpt-audio", help="Used if primary model unavailable")
    parser.add_argument("--temperature", type=float, default=0.6, help="Lower = more consistent")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; print captions only")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (for testing)")
    parser.add_argument("--sample", type=int, default=None, help="Random sample N files (variety test). Ignored if --limit set.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with existing caption")
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.3,
        help="Seconds to sleep between requests (rate-limit safety)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY env var not set.", file=sys.stderr)
        return 1

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in args.exts])
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir}", file=sys.stderr)
        return 1

    if args.skip_existing:
        before = len(files)
        files = [p for p in files if not has_caption(p)]
        print(f"[skip-existing] {before - len(files)} files already captioned; {len(files)} remaining")

    if args.limit:
        files = files[: args.limit]
    elif args.sample:
        rng = random.Random(args.seed)
        n = min(args.sample, len(files))
        files = rng.sample(files, n)
        files.sort()
        print(f"[sample] Randomly selected {n} files (seed={args.seed})")

    if not files:
        print("[INFO] Nothing to do.")
        return 0

    client = OpenAI(api_key=api_key)

    model = args.model
    print(f"Using model: {model}")
    print(f"Files to process: {len(files)}")
    print(f"Temperature: {args.temperature}")
    print(f"Sleep between: {args.sleep_between}s\n")

    if args.dry_run:
        print("[DRY RUN] Will print captions but NOT write JSON/txt files.\n")

    t_start = time.time()
    n_ok = 0
    n_failed = 0
    n_banned_total = 0
    n_captions_with_banned = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            wav_bytes = audio_to_wav_bytes(path)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            payload_kb = len(audio_b64) / 1024

            try:
                caption, usage = call_audio_api(client, model, audio_b64, args.temperature)
            except Exception as primary_err:
                if args.fallback_model and args.fallback_model != model:
                    print(
                        f"[{i:3d}/{len(files)}] Primary {model} failed, trying fallback {args.fallback_model}",
                        file=sys.stderr,
                    )
                    caption, usage = call_audio_api(client, args.fallback_model, audio_b64, args.temperature)
                else:
                    raise primary_err

            banned_count, banned_hits = count_banned_words(caption)
            if banned_count > 0:
                n_banned_total += banned_count
                n_captions_with_banned += 1

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            if not args.dry_run:
                write_outputs(path, caption)

            n_ok += 1
            elapsed = time.time() - t0
            preview = caption[:70] + ("..." if len(caption) > 70 else "")
            banned_flag = f" [!banned:{','.join(banned_hits)}]" if banned_hits else ""
            print(
                f"[{i:3d}/{len(files)}] OK  {elapsed:5.1f}s  "
                f"{payload_kb:6.1f}KB  "
                f"tok={usage.get('total_tokens', 0):5d}  "
                f"{path.name[:30]:30s}  {preview}{banned_flag}"
            )
            if args.dry_run:
                word_count = len(caption.split())
                print(f"      >>> FULL CAPTION ({word_count} words):")
                print(f"      {caption}")
                if banned_hits:
                    print(f"      [!] Banned words detected: {banned_hits}")
                print()

            time.sleep(args.sleep_between)

        except Exception as e:
            print(f"[{i:3d}/{len(files)}] FAIL {path.name}: {e}", file=sys.stderr)
            n_failed += 1

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model used:            {model}")
    print(f"Total files:           {len(files)}")
    print(f"  Succeeded:           {n_ok}")
    print(f"  Failed:              {n_failed}")
    print(f"Total time:            {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Prompt tokens:         {total_prompt_tokens:,}")
    print(f"Completion tokens:     {total_completion_tokens:,}")
    print()
    print("Estimated cost (rough, verify at https://openai.com/api/pricing/):")
    audio_in_price = 100.0 / 1_000_000
    text_out_price = 20.0 / 1_000_000
    est = total_prompt_tokens * audio_in_price + total_completion_tokens * text_out_price
    print(f"  input  ~{total_prompt_tokens:,} tok x $100/1M   = ${total_prompt_tokens * audio_in_price:.3f}")
    print(f"  output ~{total_completion_tokens:,} tok x $20/1M   = ${total_completion_tokens * text_out_price:.3f}")
    print(f"  TOTAL ESTIMATE:                              ~${est:.2f} USD")
    print()
    print(f"Banned-word audit:")
    print(f"  Captions with at least 1 banned word: {n_captions_with_banned} / {n_ok}"
          f"  ({100.0 * n_captions_with_banned / max(1, n_ok):.1f}%)")
    print(f"  Total banned word hits: {n_banned_total}")
    if n_captions_with_banned / max(1, n_ok) > 0.20:
        print(f"  [!] >20% of captions contain banned words. Consider re-tuning SYSTEM_PROMPT.")
    print()
    if n_failed > 0:
        print(f"[!] {n_failed} file(s) failed. Re-run with --skip-existing to retry only failures.")
    print()
    print("Next step: verify a few captions, then combine metadata (08_finalize.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
