from pydub import AudioSegment
from pathlib import Path
import regex as re

FOLDER_PATH = Path("datasets/vinahouse/audio/Mix")
OUTPUT_PATH = Path("datasets/vinahouse/audio/Mix")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

audio_files = list(FOLDER_PATH.glob("*.mp3"))

import unicodedata
import regex as re

def slugify_vietnamese(text: str) -> str:
    # Bỏ dấu tiếng Việt
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    # Đổi Đ/đ riêng vì normalize không xử lý thành D/d
    text = text.replace("Đ", "D").replace("đ", "d")

    # Chuyển lowercase
    text = text.lower()

    # Thay mọi ký tự không phải chữ/số bằng dấu _
    text = re.sub(r"[^a-z0-9]+", "_", text)

    # Xóa _ ở đầu/cuối
    text = text.strip("_")

    return text

count = 17
for audio_file in audio_files:
    print(f"Processing {audio_file}")

    audio = AudioSegment.from_file(audio_file)
    stop = False

    while not stop:
        length_keep = 60 * 1000  # 60 seconds

        cut_start = input("Enter the start time (in seconds): ")

        raw_name = input("Enter the name of the file: ")
        clean_name = slugify_vietnamese(raw_name)
        output_file = OUTPUT_PATH / f"vinahouse_{count}_{clean_name}.wav"

        cut_start = int(cut_start) * 1000
        cut_end = cut_start + length_keep

        newaudio = audio[cut_start:cut_end]
        newaudio.export(output_file, format="wav")

        print(f"Saved {output_file}")

        count += 1

        answer = input("Do you want to cut another part? (y/n): ")
        if answer.lower() == "n":
            stop = True

print("All audio files cut")