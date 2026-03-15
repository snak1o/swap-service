"""Celery worker — асинхронная обработка swap-задач."""

import json
from datetime import datetime
from pathlib import Path

import redis
from celery import Celery

from app.config import settings

# Celery setup
celery_app = Celery(
    "swap-worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Process one task at a time (GPU-bound)
)

# Redis client (sync) for job status updates
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _update_job(job_id: str, **updates):
    """Обновить статус задачи в Redis."""
    data = redis_client.get(_job_key(job_id))
    if data:
        job = json.loads(data)
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        redis_client.set(_job_key(job_id), json.dumps(job, default=str), ex=86400)


@celery_app.task(bind=True, name="process_swap", max_retries=2)
def process_swap_task(self, job_id: str):
    """
    Основная Celery-задача для обработки swap.
    
    Запускает полный пайплайн через SwapOrchestrator.
    """
    from app.pipeline.orchestrator import SwapOrchestrator

    print(f"[Worker] Starting job: {job_id}")

    # Get job data
    data = redis_client.get(_job_key(job_id))
    if not data:
        print(f"[Worker] Job {job_id} not found")
        return {"error": "Job not found"}

    job = json.loads(data)
    photo_path = job["photo_path"]
    video_path = job["video_path"]
    options = job.get("options", {})

    # Output path
    result_dir = settings.RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(result_dir / "result.mp4")

    # Progress callback
    def on_progress(status: str, progress: float, message: str):
        _update_job(
            job_id,
            status=status,
            progress=progress,
            current_step=status,
            message=message,
        )

    try:
        import base64
        import httpx
        import time

        _update_job(job_id, status="processing", progress=0.0, message="Sending video to RunPod...")

        # Encode files
        with open(photo_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode()
            
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "input": {
                "mode": "full_video",
                "reference_image": ref_b64,
                "video_base64": video_b64,
                "options": options,
            }
        }

        # Start job on RunPod
        run_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}/run"
        headers = {
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }

        print(f"[Worker] Submitting job {job_id} to RunPod...")
        response = httpx.post(run_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        runpod_job_id = response.json().get("id")
        if not runpod_job_id:
            raise Exception("No Job ID returned from RunPod")

        print(f"[Worker] RunPod Job ID: {runpod_job_id}")

        # Poll status
        status_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}"
        frames_processed = 0
        duration_sec = 0.0

        while True:
            status_resp = httpx.get(status_url, headers=headers, timeout=30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            
            rp_status = status_data.get("status")
            
            if rp_status == "COMPLETED":
                result_data = status_data.get("output", {})
                if result_data.get("status") == "error":
                    raise Exception(f"RunPod pipeline failed: {result_data.get('error')}")
                    
                result_b64 = result_data.get("video")
                if not result_b64:
                    raise Exception("RunPod returned empty video")
                    
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(result_b64))
                    
                frames_processed = result_data.get("frames_processed", 0)
                duration_sec = result_data.get("duration_sec", 0.0)
                _update_job(job_id, progress=0.9, message="Downloading result...")
                break
                
            elif rp_status in ["FAILED", "CANCELLED"]:
                error_msg = status_data.get("error", "No error details provided by RunPod")
                print(f"[Worker] RunPod FAILED error details: {error_msg}")
                raise Exception(f"RunPod failed with status {rp_status}: {error_msg}")
                
            elif rp_status == "IN_QUEUE":
                _update_job(job_id, progress=0.1, message="Waiting in RunPod queue...")
            elif rp_status == "IN_PROGRESS":
                _update_job(job_id, progress=0.5, message="Processing on RunPod GPU...")
                
            time.sleep(5)

        # Upload result to storage (optional)
        result_url = f"/api/download/{job_id}/result.mp4"
        try:
            from app.storage.s3 import storage
            object_name = f"results/{job_id}/result.mp4"
            storage.upload_file(output_path, object_name, "video/mp4")
            result_url = storage.get_presigned_url(object_name)
        except Exception as e:
            print(f"[Worker] Storage upload skipped: {e}")
            # Serve from local filesystem
            result_url = f"/api/download/{job_id}/result.mp4"

        # Mark as completed
        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"Completed: {frames_processed} frames in {duration_sec:.1f}s",
            result_url=result_url,
            frames_processed=frames_processed,
            completed_at=datetime.utcnow().isoformat(),
        )

        print(f"[Worker] Job {job_id} completed: {frames_processed} frames")
        return {"status": "completed", "result_url": result_url}

    except Exception as e:
        error_msg = str(e)
        print(f"[Worker] Job {job_id} failed: {error_msg}")

        _update_job(
            job_id,
            status="failed",
            progress=0.0,
            message=f"Error: {error_msg}",
            error=error_msg,
        )

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=30)

        return {"status": "failed", "error": error_msg}
