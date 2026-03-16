"""
RunPod Serverless Handler — Wan2.2-Animate-14B Replace.

Простой API:
  Input: photo (base64) + video (base64)
  Output: video (base64) с заменённым персонажем
"""

import base64
import logging
import os
import tempfile
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(event: dict) -> dict:
    """
    RunPod serverless handler.

    Input:
        photo: base64-encoded фото нового персонажа
        video: base64-encoded исходное видео

    Output:
        video: base64-encoded результат
        time_sec: время обработки
    """
    job_input = event.get("input", {})
    start = time.time()

    try:
        # --- Validate input ---
        photo_b64 = job_input.get("photo")
        video_b64 = job_input.get("video")

        if not photo_b64:
            return {"error": "photo is required (base64)"}
        if not video_b64:
            return {"error": "video is required (base64)"}

        # --- Save to temp files ---
        work_dir = tempfile.mkdtemp(prefix="swap_job_")

        photo_path = os.path.join(work_dir, "photo.jpg")
        with open(photo_path, "wb") as f:
            f.write(base64.b64decode(photo_b64))

        video_path = os.path.join(work_dir, "source.mp4")
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        output_path = os.path.join(work_dir, "result.mp4")

        logger.info(f"[Handler] Got photo + video, starting Wan2.2 Replace...")

        # --- Run Wan2.2 Replace ---
        from app.pipeline.generate import WanReplace

        wan = WanReplace()
        result_path = wan.replace(
            photo_path=photo_path,
            video_path=video_path,
            output_path=output_path,
        )

        # --- Encode result ---
        with open(result_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        elapsed = time.time() - start
        logger.info(f"[Handler] Done in {elapsed:.1f}s")

        return {
            "video": result_b64,
            "time_sec": round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"[Handler] Error: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        # Cleanup
        import shutil
        if "work_dir" in locals():
            shutil.rmtree(work_dir, ignore_errors=True)


# RunPod entrypoint
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
