"""Minimal audio API test — isolate whether gpt-audio* models accept audio.

Sends ONE audio clip with the simplest possible prompts to test:
  1. Does the model actually receive audio and describe it?
  2. Does verbose Vinahouse prompt confuse the model?
  3. What models are actually working right now?

Usage:
    .venv/bin/python datasets/vinahouse/scripts/debug_audio_api.py \\
        --audio-file datasets/vinahouse/phase_b/audio_clean/SOMEFILE.wav
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import librosa
import soundfile as sf
from openai import OpenAI


TESTS = [
    ("Minimal", "Describe the audio in one English sentence."),
    ("Music focused", "This is a music clip. Describe the instruments, tempo feel, and vocals you hear in 2 sentences."),
    ("Structured", "Listen to the audio. What genre would you classify it as? What tempo? Male or female vocals or instrumental?"),
]

MODELS = ["gpt-audio-1.5", "gpt-audio", "gpt-audio-mini"]


def audio_to_wav_bytes(path: Path, sr: int = 22050) -> bytes:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def call(client: OpenAI, model: str, prompt: str, audio_b64: str) -> tuple[str, int]:
    resp = client.chat.completions.create(
        model=model,
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=200,
    )
    text = resp.choices[0].message.content or ""
    tokens = getattr(resp.usage, "total_tokens", 0)
    return text.strip(), tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    if not args.audio_file.exists():
        print(f"File not found: {args.audio_file}", file=sys.stderr)
        return 1

    print(f"Audio file: {args.audio_file.name}")
    print(f"Loading + encoding audio...")
    wav = audio_to_wav_bytes(args.audio_file)
    print(f"WAV bytes: {len(wav)/1024:.1f} KB")
    b64 = base64.b64encode(wav).decode("utf-8")

    client = OpenAI(api_key=key)

    for model in args.models:
        print()
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)
        for label, prompt in TESTS:
            print(f"\n--- {label} ---")
            print(f"Prompt: {prompt}")
            try:
                text, tokens = call(client, model, prompt, b64)
                print(f"Tokens: {tokens}")
                print(f"Response: {text}")
            except Exception as e:
                print(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
