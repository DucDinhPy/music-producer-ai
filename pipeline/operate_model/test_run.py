import json
import time
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8001"
OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Generation params mirrored from the working Gradio UI setup.
# DO NOT set "timesteps" — the UI Custom Timesteps field is a turbo
# placeholder; sending it would override steps=50 down to 8 and cause noise.
payload = {
    "prompt": (
        "Vietnamese vinahouse, powerful kick drum, "
        "deep rolling bass, energetic synth lead"
    ),
    "lyrics": "[Instrumental]",
    "bpm": 138,
    "audio_duration": 30,
    "batch_size": 1,
    # ---- DiT Diffusion (match working UI) ----
    "model": "acestep-v15-xl-sft",
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "infer_method": "ode",
    "shift": 3.0,
    "use_adg": False,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "use_random_seed": True,
    # ---- Remix (uncomment to cover an existing track) ----
    # "src_audio_path": "/workspace/input/my_song.wav",
    # "audio_cover_strength": 0.6,
}


# ---------------------------------------------------------
# 1. Submit task
# ---------------------------------------------------------

response = requests.post(
    f"{BASE_URL}/release_task",
    json=payload,
    timeout=120,
)

print("Release status:", response.status_code)
print("Release body:", response.text)

response.raise_for_status()

release_body = response.json()

if release_body.get("code") != 200:
    raise RuntimeError(f"Task submission failed: {release_body}")

release_data = release_body.get("data") or {}
task_id = release_data.get("task_id")

if not task_id:
    raise RuntimeError(
        f"API did not return task_id. Full response: {release_body}"
    )

print("Task ID:", task_id)


# ---------------------------------------------------------
# 2. Poll task status
# ---------------------------------------------------------

while True:
    response = requests.post(
        f"{BASE_URL}/query_result",
        json={"task_id_list": [task_id]},
        timeout=60,
    )

    print("Query status:", response.status_code)
    print("Query body:", response.text)

    response.raise_for_status()

    query_body = response.json()

    if query_body.get("code") != 200:
        raise RuntimeError(f"Query failed: {query_body}")

    task_list = query_body.get("data")

    if not task_list:
        print("Task has not appeared in the result store yet. Retrying...")
        time.sleep(2)
        continue

    task = task_list[0]
    status = task.get("status")

    if status == 0:
        print("Task is queued or running...")
        time.sleep(3)
        continue

    if status == 2:
        raise RuntimeError(f"Generation failed: {task}")

    if status != 1:
        raise RuntimeError(f"Unknown task status: {task}")

    print("Generation completed.")
    break


# ---------------------------------------------------------
# 3. Parse result
# ---------------------------------------------------------

raw_result = task.get("result")

if not raw_result:
    raise RuntimeError(f"Task succeeded but result is empty: {task}")

if isinstance(raw_result, str):
    generated_items = json.loads(raw_result)
else:
    generated_items = raw_result

if not generated_items:
    raise RuntimeError(f"No generated files returned: {task}")

for index, item in enumerate(generated_items, start=1):
    file_url = item.get("file")

    if not file_url:
        print(f"Result {index} contains no file URL:", item)
        continue

    if file_url.startswith("/"):
        download_url = f"{BASE_URL}{file_url}"
    else:
        download_url = file_url

    print("Downloading:", download_url)

    audio_response = requests.get(
        download_url,
        timeout=600,
    )
    audio_response.raise_for_status()

    output_path = OUTPUT_DIR / f"result_{index:02d}.wav"
    output_path.write_bytes(audio_response.content)

    print("Saved:", output_path)