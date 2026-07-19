"""Transcribe Vietnamese lyrics for Vinahouse clips using OpenAI Whisper API.

Detects vocal vs instrumental automatically and updates per-clip metadata JSON:
  - is_instrumental: True/False (auto-detected)
  - lyrics: Vietnamese transcript (with timestamps) or "[Instrumental]"
  - raw_lyrics: plain text without timestamps
  - vocal_language: "vi" or "en" or "unknown"

Cost estimate:
  Whisper API: $0.006 / minute of audio
  152 clips x ~1 min avg = ~$1 USD total

Usage:
    export OPENAI_API_KEY="sk-proj-..."

    # Test 3 clips first
    .venv/bin/python datasets/vinahouse/scripts/09_transcribe.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --dry-run --limit 3

    # Full run
    .venv/bin/python datasets/vinahouse/scripts/09_transcribe.py \\
        --input-dir datasets/vinahouse/audio_clean \\
        --skip-existing

Detection logic:
  - Send audio to Whisper API
  - If transcript has enough Vietnamese/English words (>= 5 words) -> VOCAL
  - Else -> INSTRUMENTAL (lyrics = "[Instrumental]")
  - Handles edge cases: chopped vocal samples might get treated as noise
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path


MIN_WORDS_FOR_VOCAL = 5  # If transcript has < N words, treat as instrumental
INSTRUMENTAL_MARKER = "[Instrumental]"


try:
    import librosa
    import soundfile as sf
except ImportError:
    print("[ERROR] librosa/soundfile not installed.", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai SDK not installed.", file=sys.stderr)
    raise


def is_transcript_meaningful(text: str) -> bool:
    """Return True if transcript contains real vocal content (not just noise/silence)."""
    text = text.strip()
    if not text:
        return False
    # Remove punctuation-only content
    words = re.findall(r"\w+", text, re.UNICODE)
    if len(words) < MIN_WORDS_FOR_VOCAL:
        return False
    # Filter out common Whisper noise-transcripts
    noise_patterns = [
        r"^\.*$",  # Just dots
        r"^\s*$",  # Just whitespace
        r"^(\W+)$",  # Just symbols
    ]
    for pat in noise_patterns:
        if re.match(pat, text):
            return False
    return True


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format Whisper segments with [MM:SS] timestamps."""
    lines = []
    for seg in segments:
        start = int(seg.get("start", 0))
        m, s = divmod(start, 60)
        ts = f"[{m:02d}:{s:02d}]"
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{ts}{text}")
    return "\n".join(lines)


def detect_language_from_segments(segments: list[dict]) -> str:
    """Detect dominant language from Whisper segments.
    Falls back to 'unknown' if can't determine.
    """
    # Whisper transcribe returns 'language' at top level, not per-segment.
    # This is called by main() with response.language.
    return "unknown"


def transcribe_clip(client: OpenAI, audio_path: Path, model: str = "whisper-1") -> dict:
    """Call Whisper API. Returns dict with keys: text, language, segments.

    Uses verbose_json to get timestamps + language detection.
    """
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            language="vi",  # bias toward Vietnamese but Whisper auto-fallbacks
            temperature=0.0,
        )
    return {
        "text": resp.text,
        "language": getattr(resp, "language", "unknown"),
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in (resp.segments or [])
        ] if getattr(resp, "segments", None) else [],
        "duration": getattr(resp, "duration", 0),
    }


def update_metadata_json(audio_path: Path, transcript: dict) -> dict:
    """Merge transcript into per-clip JSON metadata. Returns final dict."""
    json_path = audio_path.with_suffix(".json")
    data: dict = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

    raw_text = transcript["text"].strip()
    segments = transcript["segments"]
    is_vocal = is_transcript_meaningful(raw_text)

    if is_vocal:
        data["is_instrumental"] = False
        data["raw_lyrics"] = raw_text
        data["lyrics"] = format_transcript_with_timestamps(segments) if segments else raw_text
        data["vocal_language"] = transcript.get("language", "unknown")
    else:
        data["is_instrumental"] = True
        data["raw_lyrics"] = ""
        data["lyrics"] = INSTRUMENTAL_MARKER
        data["vocal_language"] = "unknown"

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def has_lyrics_processed(audio_path: Path) -> bool:
    """Return True if .json already has is_instrumental + lyrics fields set."""
    json_path = audio_path.with_suffix(".json")
    if not json_path.exists():
        return False
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
        # Consider done if both keys exist and not both empty
        return ("is_instrumental" in d) and ("lyrics" in d) and d.get("lyrics")
    except json.JSONDecodeError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--exts", nargs="+", default=[".wav", ".mp3", ".flac", ".ogg"])
    parser.add_argument("--model", default="whisper-1")
    parser.add_argument("--dry-run", action="store_true", help="Print transcripts, don't save")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files with lyrics already")
    parser.add_argument("--sleep-between", type=float, default=0.3, help="Rate-limit safety sleep")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set.", file=sys.stderr)
        return 1

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted([p for p in args.input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in args.exts])
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir}", file=sys.stderr)
        return 1

    if args.skip_existing:
        before = len(files)
        files = [p for p in files if not has_lyrics_processed(p)]
        print(f"[skip-existing] {before - len(files)} files already transcribed; {len(files)} remaining")

    if args.limit:
        files = files[: args.limit]

    if not files:
        print("[INFO] Nothing to do.")
        return 0

    client = OpenAI(api_key=api_key)

    print(f"Model: {args.model}")
    print(f"Files to process: {len(files)}")
    print(f"Sleep between: {args.sleep_between}s")
    if args.dry_run:
        print("[DRY RUN] No writes.\n")
    print()

    t_start = time.time()
    n_ok = 0
    n_failed = 0
    n_vocal = 0
    n_inst = 0
    total_audio_sec = 0

    for i, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            transcript = transcribe_clip(client, path, model=args.model)
            duration = transcript.get("duration", 0)
            total_audio_sec += duration

            raw_text = transcript["text"].strip()
            is_vocal = is_transcript_meaningful(raw_text)

            if not args.dry_run:
                final_meta = update_metadata_json(path, transcript)
            else:
                final_meta = {"is_instrumental": not is_vocal, "lyrics": raw_text[:80]}

            elapsed = time.time() - t0
            n_ok += 1
            if is_vocal:
                n_vocal += 1
            else:
                n_inst += 1

            status = "VOCAL   " if is_vocal else "INSTR   "
            preview = raw_text[:60].replace("\n", " ") + ("..." if len(raw_text) > 60 else "")
            print(f"[{i:3d}/{len(files)}] {status} {elapsed:5.1f}s  "
                  f"{duration:5.1f}s audio  {path.name[:35]:35s}  {preview}")

            if args.dry_run and is_vocal:
                print(f"      >>> FULL:")
                print(f"      {raw_text}")
                if transcript.get("segments"):
                    formatted = format_transcript_with_timestamps(transcript["segments"])
                    print(f"      TIMESTAMPED:")
                    for line in formatted.split("\n")[:5]:
                        print(f"      {line}")
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
    print(f"Total files:          {len(files)}")
    print(f"  Succeeded:          {n_ok}")
    print(f"    -> VOCAL:         {n_vocal}")
    print(f"    -> INSTRUMENTAL:  {n_inst}")
    print(f"  Failed:             {n_failed}")
    print(f"Total time:           {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total audio:          {total_audio_sec:.0f}s ({total_audio_sec/60:.1f} min)")
    est_cost = (total_audio_sec / 60.0) * 0.006
    print(f"Estimated Whisper cost: ~${est_cost:.2f}")
    print()
    if n_failed > 0:
        print(f"[!] {n_failed} failed. Re-run with --skip-existing to retry only failures.")
    print("Next step: re-run 08_finalize.py to rebuild dataset.json with mixed vocal/instrumental")
    return 0


if __name__ == "__main__":
    sys.exit(main())
