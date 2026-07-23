from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def separate_folder(
    input_dir: Path,
    output_dir: Path,
    device: str = "auto",
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy input folder: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input phải là thư mục: {input_dir}"
        )

    audio_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".ogg",
    }

    audio_files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        raise RuntimeError(
            f"Không có audio trong: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Device: {device}")
    print(f"Files:  {len(audio_files)}")

    cli = shutil.which("bs-roformer-infer")
    if cli is None:
        raise RuntimeError(
            "Không tìm thấy CLI 'bs-roformer-infer'. "
            "Hãy activate venv rồi cài: python -m pip install --upgrade bs-roformer-infer"
        )

    # Dùng CLI chính thức để tránh phụ thuộc Python API nội bộ của package.
    # bs-roformer-infer tự tải model mặc định ở lần chạy đầu tiên.
    cmd = [
        cli,
        "--input_folder",
        str(input_dir),
        "--store_dir",
        str(output_dir),
    ]

    if device not in {"auto", ""}:
        print(
            "[warn] --device được giữ để tương thích command hiện tại, "
            "nhưng bs-roformer-infer CLI sẽ tự chọn device nếu không hỗ trợ flag này."
        )

    subprocess.run(cmd, check=True)

    print("Hoàn tất tách stem.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tách audio bằng BS-RoFormer-SW."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda hoặc cuda:0",
    )

    args = parser.parse_args()

    separate_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()