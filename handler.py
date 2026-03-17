"""
Wan2.2 Replace — RunPod Serverless Handler.

Input (compatible with Wan Animate template format):
  {
    "image_base64": "...",   # Base64-encoded photo (or image_url / image_path)
    "video_base64": "...",   # Base64-encoded video (or video_url / video_path)
    "width": 720,            # Optional, auto-detect from video if omitted
    "height": 1280,          # Optional, auto-detect from video if omitted
    "prompt": "...",         # Ignored (kept for compatibility)
    "steps": 20,             # Optional, default 20
    "seed": 12345,           # Optional
  }

Output (compatible with Wan Animate template format):
  {
    "video": "data:video/mp4;base64,..."
  }
"""

import base64
import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")
WAN_CKPT_DIR = os.path.join(MODELS_DIR, "Wan2.2-Animate-14B")
WAN_REPO = os.environ.get("WAN_REPO", "/workspace/Wan2.2")

REPLACE_FLAG = os.environ.get("REPLACE_FLAG", "true").lower() == "true"
USE_RELIGHTING_LORA = os.environ.get("USE_RELIGHTING_LORA", "true").lower() == "true"
OFFLOAD_MODEL = os.environ.get("OFFLOAD_MODEL", "false").lower() == "true"


def download_file(url: str, dest: str) -> None:
    """Download a file from URL to local path."""
    logger.info(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f} MB")


def save_base64_file(data: str, dest: str) -> None:
    """Decode base64 (with optional data URL prefix) and save to file."""
    # Strip data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    # Fix padding
    pad = len(data) % 4
    if pad:
        data += "=" * (4 - pad)
    decoded = base64.b64decode(data)
    with open(dest, "wb") as f:
        f.write(decoded)
    size_mb = len(decoded) / (1024 * 1024)
    logger.info(f"Saved base64 file: {size_mb:.1f} MB -> {dest}")


def resolve_input(job_input: dict, name: str, dest: str) -> str:
    """Resolve input from base64, url, or path."""
    b64_key = f"{name}_base64"
    url_key = f"{name}_url"
    path_key = f"{name}_path"

    if b64_key in job_input:
        save_base64_file(job_input[b64_key], dest)
        return dest
    elif url_key in job_input:
        download_file(job_input[url_key], dest)
        return dest
    elif path_key in job_input:
        return job_input[path_key]
    else:
        raise ValueError(f"No {name} input provided. Use {b64_key}, {url_key}, or {path_key}")


def get_video_resolution(video_path: str) -> tuple:
    """Get video resolution using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path,
        ], capture_output=True, text=True, timeout=30)
        w, h = result.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 1280, 720


def handler(job):
    """RunPod Serverless handler for Wan2.2 Replace."""
    job_input = job["input"]

    # Check model exists
    if not os.path.isdir(WAN_CKPT_DIR) or not os.listdir(WAN_CKPT_DIR):
        return {"error": f"Model not found at {WAN_CKPT_DIR}. Run setup_model.sh first."}

    work_dir = tempfile.mkdtemp(prefix="wan_swap_")

    try:
        start = time.time()

        # Resolve inputs
        runpod.serverless.progress_update(job, {"message": "Preparing files...", "percent": 2})
        photo_path = resolve_input(job_input, "image", os.path.join(work_dir, "photo.jpg"))
        video_path = resolve_input(job_input, "video", os.path.join(work_dir, "source.mp4"))
        output_path = os.path.join(work_dir, "result.mp4")

        # Resolution: from input or auto-detect from video
        if "width" in job_input and "height" in job_input:
            res_w = int(job_input["width"])
            res_h = int(job_input["height"])
        else:
            res_w, res_h = get_video_resolution(video_path)

        # Round to nearest multiple of 8
        res_w = (res_w // 8) * 8
        res_h = (res_h // 8) * 8
        logger.info(f"Resolution: {res_w}x{res_h}")

        # Step 1: Preprocess
        runpod.serverless.progress_update(job, {"message": f"Preprocessing ({res_w}x{res_h})...", "percent": 5})
        preprocess_dir = os.path.join(work_dir, "preprocess")
        _preprocess(job, video_path, photo_path, preprocess_dir, res_w, res_h)

        # Step 2: Generate
        runpod.serverless.progress_update(job, {"message": "Generating with Wan2.2 (full BF16)...", "percent": 30})
        gen_dir = os.path.join(work_dir, "generated")
        _generate(job, preprocess_dir, gen_dir)

        # Step 3: Collect result + merge audio
        runpod.serverless.progress_update(job, {"message": "Merging audio...", "percent": 90})
        result_path = _collect_result(gen_dir, output_path, video_path)

        # Encode result as base64 data URL (template-compatible format)
        runpod.serverless.progress_update(job, {"message": "Encoding result...", "percent": 95})
        with open(result_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        elapsed = time.time() - start
        logger.info(f"Done in {elapsed:.1f}s")

        return {
            "video": f"data:video/mp4;base64,{video_b64}",
            "refresh_worker": True,
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e), "refresh_worker": True}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _cleanup_gpu()


# =============================================
# Wan2.2 Pipeline
# =============================================

def _preprocess(job, video_path, photo_path, save_path, res_w, res_h):
    """Wan2.2 preprocessing."""
    os.makedirs(save_path, exist_ok=True)

    script = os.path.join(WAN_REPO, "wan", "modules", "animate",
                          "preprocess", "preprocess_data.py")
    ckpt = os.path.join(WAN_CKPT_DIR, "process_checkpoint")

    cmd = [
        sys.executable, script,
        "--ckpt_path", ckpt,
        "--video_path", video_path,
        "--refer_path", photo_path,
        "--save_path", save_path,
        "--resolution_area", str(res_w), str(res_h),
        "--iterations", "3",
        "--k", "7",
        "--w_len", "1",
        "--h_len", "1",
    ]

    if REPLACE_FLAG:
        cmd.append("--replace_flag")

    logger.info("[preprocess] Running...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WAN_REPO)
    last_update = time.time()
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            if line:
                stripped = line.strip()
                logger.info(f"[preprocess] {stripped}")
                if time.time() - last_update > 5:
                    runpod.serverless.progress_update(job, {
                        "message": f"[preprocess] {stripped}",
                        "percent": 15
                    })
                    last_update = time.time()
        process.stdout.close()
    returncode = process.wait(timeout=600)

    if returncode != 0:
        raise RuntimeError(f"Preprocess failed with code {returncode}")
    logger.info("[preprocess] Done")


def _generate(job, src_root_path, output_dir):
    """Wan2.2 inference — Replace mode with full BF16 model."""
    os.makedirs(output_dir, exist_ok=True)

    script = os.path.join(WAN_REPO, "generate.py")

    cmd = [
        sys.executable, script,
        "--task", "animate-14B",
        "--ckpt_dir", WAN_CKPT_DIR,
        "--src_root_path", src_root_path,
        "--refert_num", "1",
        "--save_file", output_dir,
    ]

    if REPLACE_FLAG:
        cmd.append("--replace_flag")
    if USE_RELIGHTING_LORA:
        cmd.append("--use_relighting_lora")
    if OFFLOAD_MODEL:
        cmd.extend(["--offload_model", "True", "--convert_model_dtype"])

    logger.info("[generate] Running Wan2.2...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WAN_REPO)
    last_update = time.time()
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            if line:
                stripped = line.strip()
                logger.info(f"[generate] {stripped}")
                if time.time() - last_update > 5:
                    runpod.serverless.progress_update(job, {
                        "message": f"[generate] {stripped}",
                        "percent": 50
                    })
                    last_update = time.time()
        process.stdout.close()
    returncode = process.wait(timeout=7200)

    if returncode != 0:
        raise RuntimeError(f"Generate failed with code {returncode}")
    logger.info("[generate] Done")


def _collect_result(gen_dir, output_path, source_video):
    """Find Wan2.2 output and merge audio from original."""
    videos = sorted(Path(gen_dir).rglob("*.mp4"))
    if videos:
        _merge_audio(str(videos[0]), source_video, output_path)
        return output_path

    frames = sorted(Path(gen_dir).rglob("*.png"))
    if not frames:
        frames = sorted(Path(gen_dir).rglob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No output in {gen_dir}")

    frame_pattern = str(frames[0].parent / "%04d.png")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", frame_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ], capture_output=True, timeout=120)

    _merge_audio(output_path, source_video, output_path)
    return output_path


def _merge_audio(video_path, audio_source, output):
    """Merge audio from original video into result."""
    temp = output + ".tmp.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path, "-i", audio_source,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", temp,
        ], capture_output=True, timeout=120)

        if os.path.exists(temp) and os.path.getsize(temp) > 0:
            shutil.move(temp, output)
    except Exception:
        pass
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _cleanup_gpu():
    """Free GPU VRAM."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


runpod.serverless.start({"handler": handler})
