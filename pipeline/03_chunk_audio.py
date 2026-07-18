"""Chunk long audio mixes into fixed-length WAV clips for training.

Splits every audio file in --input-dir into contiguous chunks of --chunk-seconds,
optionally skipping the intro and outro. Outputs are named:
    {prefix}_{run_index}_{chunk_index}.wav

Example — chunk all MP3s in Mix/ into 60s pieces, skip first/last 10 seconds:

    .venv/bin/python datasets/vinahouse/scripts/03_chunk_audio.py \\
        --input-dir  datasets/vinahouse/audio/Mix \\
        --output-dir datasets/vinahouse/audio_from_mix \\
        --chunk-seconds 60 \\
        --skip-head-seconds 10 \\
        --skip-tail-seconds 10 \\
        --prefix vinahouse \\
        --start-index 17

Behavior notes:
  - Chunks shorter than --min-chunk-seconds are dropped (avoid stubby endings).
  - Each source file's chunks share a run_index; chunks within share chunk_index.
  - Overwrites are gated by --overwrite; default skips existing files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    print("[ERROR] pydub not installed. Install: .venv/bin/pip install pydub", file=sys.stderr)
    raise


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}


def chunk_file(
    audio_path: Path,
    output_dir: Path,
    chunk_ms: int,
    skip_head_ms: int,
    skip_tail_ms: int,
    min_chunk_ms: int,
    prefix: str,
    run_index: int,
    overwrite: bool,
) -> int:
    """Chunk one audio file. Returns number of chunks written."""
    audio = AudioSegment.from_file(audio_path)
    audio_name = audio_path.stem
    artist_name = audio_name.split("-", 1)[0].strip()
    total_ms = len(audio)

    start = skip_head_ms
    end = total_ms - skip_tail_ms
    if end - start < min_chunk_ms:
        print(f"[SKIP] {audio_path.name}: only {(end-start)/1000:.1f}s after trim, need >= {min_chunk_ms/1000:.1f}s")
        return 0

    trimmed = audio[start:end]
    n_full = len(trimmed) // chunk_ms
    n_written = 0

    for i in range(n_full):
        chunk = trimmed[i * chunk_ms : (i + 1) * chunk_ms]
        name = f"{prefix}_{artist_name}_{run_index}_chunk_{i}.wav"
        dst = output_dir / name
        if dst.exists() and not overwrite:
            print(f"[SKIP] exists: {name}")
            continue
        chunk.export(dst, format="wav")
        print(f"  wrote {name}  ({len(chunk)/1000:.1f}s)")
        n_written += 1

    remainder = trimmed[n_full * chunk_ms :]
    if len(remainder) >= min_chunk_ms:
        name = f"{prefix}_{artist_name}_{run_index}_chunk_{n_full}.wav"
        dst = output_dir / name
        if not dst.exists() or overwrite:
            remainder.export(dst, format="wav")
            print(f"  wrote {name}  ({len(remainder)/1000:.1f}s tail)")
            n_written += 1

    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder containing source audio files (recursive: no)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination folder for chunk WAVs")
    parser.add_argument("--exts", nargs="+", default=[".mp3", ".wav", ".flac"],
                        help="File extensions to process (space-separated)")
    parser.add_argument("--chunk-seconds", type=int, default=60,
                        help="Length of each chunk in seconds (default 60)")
    parser.add_argument("--min-chunk-seconds", type=int, default=20,
                        help="Drop chunks shorter than this (default 20s)")
    parser.add_argument("--skip-head-seconds", type=int, default=10,
                        help="Skip N seconds at start of each file (default 10). "
                             "Use 0 for standalone songs without DJ intro.")
    parser.add_argument("--skip-tail-seconds", type=int, default=10,
                        help="Skip N seconds at end of each file (default 10). "
                             "Trims outro/fade-out from DJ mixes.")
    parser.add_argument("--prefix", default="vinahouse",
                        help="Filename prefix; final name = {prefix}_{index}_chunk_{i}.wav")
    parser.add_argument("--start-index", type=int, default=1,
                        help="First run_index for output filenames (used to avoid clashing with existing files)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing chunk files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be written; skip actual export")
    parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Only process the first N audio files"
)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.exts}
    files = sorted([p for p in args.input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts])
    files = sorted([
    p for p in args.input_dir.iterdir()
    if p.is_file() and p.suffix.lower() in exts
])

    
    if not files:
        print(f"[ERROR] No audio files in {args.input_dir} (looking for: {sorted(exts)})",
              file=sys.stderr)
        return 1
    
    if args.limit is not None:
        files = files[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_ms = args.chunk_seconds * 1000
    skip_head_ms = args.skip_head_seconds * 1000
    skip_tail_ms = args.skip_tail_seconds * 1000
    min_chunk_ms = args.min_chunk_seconds * 1000

    print("=" * 70)
    print(f"Chunking {len(files)} files")
    print(f"  input:  {args.input_dir}")
    print(f"  output: {args.output_dir}")
    print(f"  chunk:  {args.chunk_seconds}s (min accepted: {args.min_chunk_seconds}s)")
    print(f"  skip:   head {args.skip_head_seconds}s / tail {args.skip_tail_seconds}s")
    print(f"  prefix: {args.prefix}_ (index starts at {args.start_index})")
    if args.dry_run:
        print("  [DRY RUN]")
    print("=" * 70)

    run_index = args.start_index
    total_written = 0
    for f in files:
        print(f"\nProcessing {f.name}")
        if args.dry_run:
            audio = AudioSegment.from_file(f)
            total = len(audio)
            usable = total - skip_head_ms - skip_tail_ms
            n_chunks = usable // chunk_ms if usable > 0 else 0
            remainder = usable - n_chunks * chunk_ms if usable > 0 else 0
            print(f"  duration {total/1000:.1f}s, usable {usable/1000:.1f}s")
            print(f"  -> {n_chunks} full chunks + "
                  f"{'1 tail chunk' if remainder >= min_chunk_ms else 'no tail'}")
        else:
            n = chunk_file(
                audio_path=f,
                output_dir=args.output_dir,
                chunk_ms=chunk_ms,
                skip_head_ms=skip_head_ms,
                skip_tail_ms=skip_tail_ms,
                min_chunk_ms=min_chunk_ms,
                prefix=args.prefix,
                run_index=run_index,
                overwrite=args.overwrite,
            )
            total_written += n
        run_index += 1

    print()
    print("=" * 70)
    print(f"DONE. {'(dry-run)' if args.dry_run else f'{total_written} chunk files written'}")
    if not args.dry_run:
        print(f"Next step: audit chunks with 04_audit_audio.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
