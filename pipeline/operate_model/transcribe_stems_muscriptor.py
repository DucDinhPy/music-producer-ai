from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_EXCLUDE_SUFFIXES = {
    "instrumental",
}


def transcribe_stems(
    input_dir: Path,
    output_dir: Path,
    model: str = "large",
    skip_existing: bool = True,
    fail_fast: bool = False,
    jobs: int = 1,
    exclude_suffixes: set[str] | None = None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy input folder: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input phải là thư mục: {input_dir}")

    cli = shutil.which("muscriptor")
    if cli is None:
        raise RuntimeError(
            "Không tìm thấy CLI 'muscriptor'. "
            "Hãy activate venv rồi cài: "
            "python -m pip install git+https://github.com/muscriptor/muscriptor.git"
        )

    exclude_suffixes = exclude_suffixes or set()
    wav_files = [
        path
        for path in sorted(input_dir.rglob("*.wav"))
        if _stem_suffix(path) not in exclude_suffixes
    ]

    if not wav_files:
        raise RuntimeError(f"Không tìm thấy stem WAV hợp lệ trong: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "muscriptor_manifest.jsonl"

    print(f"Input:          {input_dir.resolve()}")
    print(f"Output:         {output_dir.resolve()}")
    print(f"Model:          {model}")
    print(f"Jobs:           {jobs}")
    print(f"Skip existing:  {skip_existing}")
    print(f"Fail fast:      {fail_fast}")
    print(f"Exclude suffix: {', '.join(sorted(exclude_suffixes)) or '(none)'}")
    print(f"Files:          {len(wav_files)}")
    print(f"Manifest:       {manifest_path.resolve()}")

    ok_count = 0
    skip_count = 0
    error_count = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        tasks = []
        for index, stem_path in enumerate(wav_files, start=1):
            relative = stem_path.relative_to(input_dir)
            midi_path = output_dir / relative.with_suffix(".mid")
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((index, stem_path, relative, midi_path))

        if jobs <= 1:
            for index, stem_path, relative, midi_path in tasks:
                print(f"\n[{index}/{len(wav_files)}] {relative}")
                result = _transcribe_one(cli, stem_path, midi_path, model, skip_existing)
                _write_manifest(
                    manifest,
                    stem_path,
                    midi_path,
                    model,
                    result["status"],
                    returncode=result.get("returncode"),
                    error=result.get("error"),
                )
                ok_count, skip_count, error_count = _update_counts(
                    result["status"],
                    ok_count,
                    skip_count,
                    error_count,
                )
                _print_result(result, midi_path)
                if fail_fast and result["status"] == "error":
                    raise RuntimeError(result.get("error") or "Muscriptor failed")
        else:
            print("")
            print(f"Running {jobs} Muscriptor workers in parallel.")
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_to_task = {
                    executor.submit(
                        _transcribe_one,
                        cli,
                        stem_path,
                        midi_path,
                        model,
                        skip_existing,
                    ): (index, stem_path, relative, midi_path)
                    for index, stem_path, relative, midi_path in tasks
                }

                for future in as_completed(future_to_task):
                    index, stem_path, relative, midi_path = future_to_task[future]
                    print(f"\n[{index}/{len(wav_files)}] {relative}")
                    result = future.result()
                    _write_manifest(
                        manifest,
                        stem_path,
                        midi_path,
                        model,
                        result["status"],
                        returncode=result.get("returncode"),
                        error=result.get("error"),
                    )
                    ok_count, skip_count, error_count = _update_counts(
                        result["status"],
                        ok_count,
                        skip_count,
                        error_count,
                    )
                    _print_result(result, midi_path)
                    if fail_fast and result["status"] == "error":
                        raise RuntimeError(result.get("error") or "Muscriptor failed")

    print("")
    print("Hoàn tất Muscriptor batch.")
    print(f"OK:      {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors:  {error_count}")


def _stem_suffix(path: Path) -> str:
    # BS-RoFormer outputs names like song_bass.wav, song_drums.wav.
    return path.stem.rsplit("_", 1)[-1].lower()


def _transcribe_one(
    cli: str,
    stem_path: Path,
    midi_path: Path,
    model: str,
    skip_existing: bool,
) -> dict[str, str | int]:
    if skip_existing and midi_path.exists() and midi_path.stat().st_size > 0:
        return {"status": "skipped"}

    tmp_path = midi_path.with_name(
        f".{midi_path.stem}.{uuid.uuid4().hex}.tmp.mid"
    )
    tmp_path.unlink(missing_ok=True)

    cmd = [
        cli,
        "transcribe",
        "--model",
        model,
        str(stem_path),
        "-o",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        return {
            "status": "error",
            "returncode": exc.returncode,
            "error": f"Muscriptor failed with code {exc.returncode}",
        }

    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        return {
            "status": "error",
            "returncode": 0,
            "error": "Muscriptor finished but did not write a non-empty MIDI file",
        }

    tmp_path.replace(midi_path)
    return {"status": "ok"}


def _update_counts(
    status: str | int,
    ok_count: int,
    skip_count: int,
    error_count: int,
) -> tuple[int, int, int]:
    if status == "ok":
        ok_count += 1
    elif status == "skipped":
        skip_count += 1
    elif status == "error":
        error_count += 1
    return ok_count, skip_count, error_count


def _print_result(result: dict[str, str | int], midi_path: Path) -> None:
    status = result["status"]
    if status == "ok":
        print(f"Wrote: {midi_path}")
    elif status == "skipped":
        print(f"Skip existing: {midi_path}")
    else:
        print(f"[error] {result.get('error')}")


def _write_manifest(
    manifest,
    stem_path: Path,
    midi_path: Path,
    model: str,
    status: str,
    returncode: int | None = None,
    error: str | int | None = None,
) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": model,
        "stem_path": str(stem_path),
        "midi_path": str(midi_path),
    }
    if returncode is not None:
        row["returncode"] = returncode
    if error is not None:
        row["error"] = str(error)
    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy Muscriptor large cho folder stem WAV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder chứa stems WAV từ BS-RoFormer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder lưu MIDI output.",
    )
    parser.add_argument(
        "--model",
        default="large",
        help="Muscriptor model: small, medium, large hoặc model path/URL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Transcribe lại cả file MIDI đã tồn tại.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Số process Muscriptor chạy song song. Thử 2 trước với model large.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Dừng ngay khi một file lỗi.",
    )
    parser.add_argument(
        "--include-instrumental",
        action="store_true",
        help="Transcribe cả *_instrumental.wav. Mặc định bỏ qua để tránh trùng data.",
    )

    args = parser.parse_args()

    exclude_suffixes = set()
    if not args.include_instrumental:
        exclude_suffixes.update(DEFAULT_EXCLUDE_SUFFIXES)

    transcribe_stems(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        skip_existing=not args.overwrite,
        fail_fast=args.fail_fast,
        jobs=args.jobs,
        exclude_suffixes=exclude_suffixes,
    )


if __name__ == "__main__":
    main()
