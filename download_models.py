"""
Утилита для загрузки всех моделей на RunPod Network Volume.

Запускается ОДИН РАЗ при первом деплое:
  python download_models.py

Модели сохраняются в /runpod-volume/models/ и переживают перезапуски.
"""

import os
import subprocess
import sys


def get_models_dir():
    """Получить директорию для моделей из env."""
    return os.environ.get("MODELS_DIR", "/runpod-volume/models")


def download_wan_model(models_dir):
    """Скачать Wan2.2-Animate-14B через huggingface-cli."""
    model_name = os.environ.get("WAN_MODEL_NAME", "Wan-AI/Wan2.2-Animate-14B")
    local_dir = os.path.join(models_dir, "Wan2.2-Animate-14B")

    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[download] {model_name} already exists at {local_dir}, skipping.")
        return

    print(f"[download] Downloading {model_name} to {local_dir}...")
    print("[download] This is ~50GB and will take 30-60 minutes on first run.")

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
    print(f"[download] {model_name} downloaded successfully!")


def download_sam2(models_dir):
    """Скачать SAM 2.1 Hiera Large."""
    dest = os.path.join(models_dir, "sam2.1_hiera_large.pt")
    if os.path.isfile(dest):
        print(f"[download] SAM2 already exists at {dest}, skipping.")
        return

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    print(f"[download] Downloading SAM 2.1 Hiera Large...")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)
    print(f"[download] SAM2 downloaded to {dest}")


def download_gfpgan(models_dir):
    """Скачать GFPGAN v1.4."""
    dest = os.path.join(models_dir, "GFPGANv1.4.pth")
    if os.path.isfile(dest):
        print(f"[download] GFPGAN already exists at {dest}, skipping.")
        return

    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    print(f"[download] Downloading GFPGAN v1.4...")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)
    print(f"[download] GFPGAN downloaded to {dest}")


def download_insightface(models_dir):
    """Скачать InsightFace buffalo_l через Python API."""
    insightface_dir = os.path.join(models_dir, "insightface")
    buffalo_dir = os.path.join(insightface_dir, "models", "buffalo_l")

    if os.path.isdir(buffalo_dir) and os.listdir(buffalo_dir):
        print(f"[download] InsightFace buffalo_l already exists, skipping.")
        return

    print("[download] Downloading InsightFace buffalo_l...")
    os.makedirs(insightface_dir, exist_ok=True)
    os.environ["INSIGHTFACE_HOME"] = insightface_dir

    try:
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(
            name="buffalo_l",
            root=insightface_dir,
            providers=["CPUExecutionProvider"],
        )
        fa.prepare(ctx_id=-1)
        print("[download] InsightFace buffalo_l downloaded successfully!")
    except Exception as e:
        print(f"[download] InsightFace download failed: {e}")


def main():
    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    print(f"=== Downloading all models to {models_dir} ===")
    print()

    download_wan_model(models_dir)
    download_sam2(models_dir)
    download_gfpgan(models_dir)
    download_insightface(models_dir)

    print()
    print("=== All models downloaded successfully! ===")
    print(f"Total size: check with 'du -sh {models_dir}'")


if __name__ == "__main__":
    main()
