"""
Wan2.2 Replace — API сервер.

Endpoints:
  POST /swap   — фото + видео → видео с заменённым персонажем
  GET  /health — статус сервера
  GET  /       — info

Запуск:
  python server.py
"""

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
VOLUME = os.environ.get("VOLUME", "/runpod-volume")
MODELS_DIR = os.path.join(VOLUME, "models")
WAN_CKPT_DIR = os.path.join(MODELS_DIR, "Wan2.2-Animate-14B")
WAN_REPO = os.environ.get("WAN_REPO", "/workspace/Wan2.2")
PORT = int(os.environ.get("PORT", "8000"))

# Wan2.2 settings
RESOLUTION_W = int(os.environ.get("RESOLUTION_W", "1280"))
RESOLUTION_H = int(os.environ.get("RESOLUTION_H", "720"))
REPLACE_FLAG = os.environ.get("REPLACE_FLAG", "true").lower() == "true"
USE_RELIGHTING_LORA = os.environ.get("USE_RELIGHTING_LORA", "true").lower() == "true"
OFFLOAD_MODEL = os.environ.get("OFFLOAD_MODEL", "false").lower() == "true"


@app.route("/")
def index():
    return jsonify({
        "service": "Wan2.2 Replace",
        "model": "Wan2.2-Animate-14B",
        "endpoints": {
            "POST /swap": "photo + video → swapped video",
            "GET /health": "server status",
        },
    })


@app.route("/health")
def health():
    model_exists = os.path.isdir(WAN_CKPT_DIR) and bool(os.listdir(WAN_CKPT_DIR))
    return jsonify({
        "status": "ok",
        "model_loaded": model_exists,
        "gpu": _get_gpu_info(),
    })


@app.route("/swap", methods=["POST"])
def swap():
    """
    POST /swap
    Files: photo (jpg/png), video (mp4)
    Returns: mp4 video
    """
    start = time.time()

    if "photo" not in request.files:
        return jsonify({"error": "photo file required"}), 400
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    work_dir = tempfile.mkdtemp(prefix="wan_swap_")

    try:
        # Save uploads
        photo_path = os.path.join(work_dir, "photo.jpg")
        video_path = os.path.join(work_dir, "source.mp4")
        output_path = os.path.join(work_dir, "result.mp4")

        request.files["photo"].save(photo_path)
        request.files["video"].save(video_path)

        logger.info(f"[swap] Got photo + video, starting Wan2.2 Replace...")

        # Step 1: Preprocess
        preprocess_dir = os.path.join(work_dir, "preprocess")
        _preprocess(video_path, photo_path, preprocess_dir)

        # Step 2: Generate
        gen_dir = os.path.join(work_dir, "generated")
        _generate(preprocess_dir, gen_dir)

        # Step 3: Collect result + merge audio
        result = _collect_result(gen_dir, output_path, video_path)

        elapsed = time.time() - start
        logger.info(f"[swap] Done in {elapsed:.1f}s")

        return send_file(result, mimetype="video/mp4", as_attachment=True,
                         download_name="result.mp4")

    except Exception as e:
        logger.error(f"[swap] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _cleanup_gpu()


# =============================================
# Wan2.2 Pipeline
# =============================================

def _preprocess(video_path, photo_path, save_path):
    """Wan2.2 preprocessing: skeleton, face encoding, pose."""
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
        "--resolution_area", str(RESOLUTION_W), str(RESOLUTION_H),
        "--iterations", "3",
        "--k", "7",
        "--w_len", "1",
        "--h_len", "1",
    ]

    if REPLACE_FLAG:
        cmd.append("--replace_flag")

    logger.info("[preprocess] Running...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=WAN_REPO, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"Preprocess failed: {result.stderr[:500]}")
    logger.info("[preprocess] Done")


def _generate(src_root_path, output_dir):
    """Wan2.2 inference — Replace mode."""
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

    logger.info("[generate] Running Wan2.2 (this takes a while)...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=WAN_REPO, timeout=7200)

    if result.returncode != 0:
        raise RuntimeError(f"Generate failed: {result.stderr[:500]}")
    logger.info("[generate] Done")


def _collect_result(gen_dir, output_path, source_video):
    """Find Wan2.2 output and merge audio from original."""
    # Find output video
    videos = sorted(Path(gen_dir).rglob("*.mp4"))
    if videos:
        _merge_audio(str(videos[0]), source_video, output_path)
        return output_path

    # Try frames
    frames = sorted(Path(gen_dir).rglob("*.png"))
    if not frames:
        frames = sorted(Path(gen_dir).rglob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No output in {gen_dir}")

    # Frames → video
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


def _get_gpu_info():
    """Get GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {"name": parts[0], "vram_total_mb": parts[1], "vram_used_mb": parts[2]}
    except Exception:
        pass
    return None


if __name__ == "__main__":
    logger.info(f"Starting Wan2.2 Replace server on port {PORT}")
    logger.info(f"Model: {WAN_CKPT_DIR}")
    logger.info(f"Repo: {WAN_REPO}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
