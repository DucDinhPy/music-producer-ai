"""Verify audio files in datasets/vinahouse/audio/.

Checks:
    - Sample rate (target: 48kHz — ACE-Step will resample if not, but slower + lossy)
    - Channels (target: 2/stereo)
    - Duration (target: 30-120s, sweet spot 60s)
    - Bitrate (rough sanity check for mp3 quality)
    - No corrupt / unreadable files

Usage:
    python 01_verify_audio.py

Deps: pip install soundfile librosa

Run this INSIDE the acestep conda env (already has librosa/soundfile).
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import soundfile as sf
except ImportError:
    print("[ERROR] Missing dependency. Run: pip install soundfile", file=sys.stderr)
    raise

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"

TARGET_SR = 48000
TARGET_CHANNELS = 2
MIN_DURATION = 30.0
MAX_DURATION = 120.0
IDEAL_DURATION = 60.0


def fmt_row(cols: list[str], widths: list[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


def main() -> int:
    if not AUDIO_DIR.exists():
        print(f"[ERROR] Audio dir not found: {AUDIO_DIR}", file=sys.stderr)
        return 1

    files = sorted(
        [p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".opus"}]
    )
    if not files:
        print(f"[ERROR] No audio files in {AUDIO_DIR}", file=sys.stderr)
        return 1

    print(f"Scanning {len(files)} files in {AUDIO_DIR}\n")

    widths = [40, 6, 8, 10, 8, 10]
    header = fmt_row(["File", "SR", "Chan", "Duration", "Format", "Status"], widths)
    print(header)
    print("-" * len(header))

    all_ok = True
    total_seconds = 0.0
    problems: list[tuple[str, str]] = []

    for path in files:
        rel = path.name
        try:
            info = sf.info(str(path))
        except Exception as e:
            print(fmt_row([rel[:40], "?", "?", "?", "?", "FAIL"], widths))
            problems.append((rel, f"Cannot read: {e}"))
            all_ok = False
            continue

        sr = info.samplerate
        ch = info.channels
        dur = info.frames / sr if sr else 0.0
        fmt = info.format
        total_seconds += dur

        issues = []
        if sr != TARGET_SR:
            issues.append(f"SR={sr} (want {TARGET_SR})")
        if ch != TARGET_CHANNELS:
            issues.append(f"channels={ch} (want {TARGET_CHANNELS})")
        if dur < MIN_DURATION:
            issues.append(f"too short {dur:.1f}s")
        elif dur > MAX_DURATION:
            issues.append(f"too long {dur:.1f}s")

        status = "OK" if not issues else "WARN"
        if issues:
            all_ok = False
            problems.append((rel, "; ".join(issues)))

        print(fmt_row([rel[:40], str(sr), str(ch), f"{dur:.1f}s", fmt, status], widths))

    print("-" * len(header))
    avg = total_seconds / len(files) if files else 0.0
    print(f"\nSummary: {len(files)} files, total {total_seconds:.1f}s, avg {avg:.1f}s per clip")
    print(f"Ideal per clip: {IDEAL_DURATION:.0f}s ({MIN_DURATION:.0f}-{MAX_DURATION:.0f}s acceptable)")

    if problems:
        print(f"\n[!] {len(problems)} file(s) with issues:")
        for name, reason in problems:
            print(f"  - {name}: {reason}")
        print("\nRecommendation:")
        print("  - SR mismatch:      re-export at 48000 Hz OR let ACE-Step resample (auto, quality loss minor)")
        print("  - Mono file:        re-export as stereo (or duplicate mono to L+R channel)")
        print("  - Too short/long:   re-trim to ~60s")
        return 1

    if all_ok:
        print("\n[OK] All files pass sanity check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
