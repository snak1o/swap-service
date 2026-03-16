"""Загрузка Wan2.2-Animate-14B на Network Volume (один раз)."""

import os
import subprocess
import sys


def main():
    models_dir = os.environ.get("MODELS_DIR", "/runpod-volume/models")
    model_name = os.environ.get("WAN_MODEL_NAME", "Wan-AI/Wan2.2-Animate-14B")
    local_dir = os.path.join(models_dir, "Wan2.2-Animate-14B")

    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[download] Already exists: {local_dir}")
        return

    os.makedirs(models_dir, exist_ok=True)
    print(f"[download] Downloading {model_name} → {local_dir}")
    print("[download] ~50GB, takes 30-60 min first time...")

    hf_token = os.environ.get("HF_TOKEN", "")
    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.download",
        "--repo-id", model_name,
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False",
    ]
    if hf_token:
        cmd.extend(["--token", hf_token])

    subprocess.run(cmd, check=True)
    print("[download] Done!")


if __name__ == "__main__":
    main()
