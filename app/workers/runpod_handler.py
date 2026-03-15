"""
RunPod Serverless Handler — GPU Worker для full-body swap.

Этот Docker-образ разворачивается на RunPod Serverless.
Содержит все AI-модели и выполняет тяжёлую GPU-обработку.

Принимает:
- reference_image (base64)
- video_url или video_base64
- options (fps, scene_detection, etc.)

Возвращает:
- result_video (base64 или URL)
"""

import base64
import os
import tempfile
import time

import cv2
import numpy as np


def decode_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (BGR)."""
    img_bytes = base64.b64decode(b64_string)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def encode_image(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode numpy array to base64 string."""
    _, buf = cv2.imencode(fmt, image)
    return base64.b64encode(buf).decode()


def handler(event: dict) -> dict:
    """
    RunPod serverless handler.
    
    Input:
        reference_image: base64-encoded photo
        pose_images: list of base64-encoded pose skeleton images
        mode: "single_frame" | "full_video" | "batch_frames"
        options: dict with processing options
        
    Output:
        frames: list of base64-encoded result frames
        status: "success" | "error"
    """
    job_input = event.get("input", {})
    mode = job_input.get("mode", "single_frame")

    try:
        # Load reference image
        ref_b64 = job_input.get("reference_image")
        if not ref_b64:
            return {"status": "error", "error": "reference_image is required"}

        reference = decode_image(ref_b64)

        if mode == "single_frame":
            return _process_single_frame(reference, job_input)
        elif mode == "batch_frames":
            return _process_batch_frames(reference, job_input)
        elif mode == "full_video":
            return _process_full_video(reference, job_input)
        else:
            return {"status": "error", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def _process_single_frame(reference: np.ndarray, job_input: dict) -> dict:
    """Обработка одного кадра: pose → body generation → face refinement."""
    from app.pipeline.generate import BodyGenerator
    from app.pipeline.face import FaceRefiner

    pose_b64 = job_input.get("pose_image")
    if not pose_b64:
        return {"status": "error", "error": "pose_image is required"}

    pose = decode_image(pose_b64)

    # Generate body
    generator = BodyGenerator(model_name="animate_anyone_2")
    generated = generator.generate_frame(reference, pose)

    # Refine face
    refiner = FaceRefiner()
    refined = refiner.refine_frame(generated, reference)

    return {
        "status": "success",
        "image": encode_image(refined),
    }


def _process_batch_frames(reference: np.ndarray, job_input: dict) -> dict:
    """Обработка батча кадров."""
    from app.pipeline.generate import BodyGenerator
    from app.pipeline.face import FaceRefiner

    pose_images_b64 = job_input.get("pose_images", [])
    if not pose_images_b64:
        return {"status": "error", "error": "pose_images list is required"}

    poses = [decode_image(b64) for b64 in pose_images_b64]

    generator = BodyGenerator(model_name="animate_anyone_2")
    generated = generator.generate_sequence(reference, poses)

    refiner = FaceRefiner()
    refined = refiner.refine_sequence(generated, reference)

    result_b64 = [encode_image(f) for f in refined]

    return {
        "status": "success",
        "frames": result_b64,
        "count": len(result_b64),
    }


def _process_full_video(reference: np.ndarray, job_input: dict) -> dict:
    """Полная обработка видео на GPU (все этапы)."""
    from app.pipeline.orchestrator import SwapOrchestrator

    video_b64 = job_input.get("video_base64")
    if not video_b64:
        return {"status": "error", "error": "video_base64 is required"}

    # Decode video to temp file
    video_bytes = base64.b64decode(video_b64)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    # Save reference to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, reference)
        photo_path = f.name

    output_path = tempfile.mktemp(suffix=".mp4")

    options = job_input.get("options", {})

    try:
        orchestrator = SwapOrchestrator()
        result = orchestrator.process(
            photo_path=photo_path,
            video_path=video_path,
            output_path=output_path,
            options=options,
        )

        # Read result and encode
        with open(output_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

        return {
            "status": "success",
            "video": result_b64,
            "frames_processed": result["frames_processed"],
            "duration_sec": result["duration_sec"],
        }
    finally:
        # Cleanup
        for path in [video_path, photo_path, output_path]:
            try:
                os.unlink(path)
            except OSError:
                pass


# RunPod entrypoint
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
