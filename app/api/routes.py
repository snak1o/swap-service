"""API маршруты для swap-service."""

import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis.asyncio as redis
import json

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.schemas import (
    JobStatus,
    SwapResponse,
    JobStatusResponse,
    JobResultResponse,
)

router = APIRouter(prefix="/api", tags=["swap"])

# Redis connection (async)
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


async def _get_job(job_id: str) -> dict:
    """Получить данные задачи из Redis."""
    data = await redis_client.get(_job_key(job_id))
    if not data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return json.loads(data)


async def _save_job(job_id: str, data: dict):
    """Сохранить данные задачи в Redis (TTL 24h)."""
    await redis_client.set(_job_key(job_id), json.dumps(data, default=str), ex=86400)


@router.post("/swap", response_model=SwapResponse)
async def create_swap(
    photo: UploadFile = File(..., description="Фото целевого человека (лицо + тело)"),
    video: UploadFile = File(..., description="Исходное видео"),
    target_fps: Optional[int] = Form(None),
    skip_frames: int = Form(0),
    scene_detection: bool = Form(True),
    temporal_smoothing: bool = Form(True),
    upscale: bool = Form(False),
):
    """Создать задачу на full-body swap."""

    # Validate file types
    photo_ext = Path(photo.filename or "photo.jpg").suffix.lower()
    video_ext = Path(video.filename or "video.mp4").suffix.lower()

    if photo_ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Photo must be JPG, PNG, or WebP")
    if video_ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "Video must be MP4, MOV, AVI, MKV, or WebM")

    # Generate job ID
    job_id = str(uuid.uuid4())
    job_dir = settings.UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    photo_path = job_dir / f"photo{photo_ext}"
    video_path = job_dir / f"video{video_ext}"

    with open(photo_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Job metadata
    now = datetime.utcnow()
    job_data = {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "progress": 0.0,
        "current_step": "",
        "message": "Job created, waiting for processing",
        "photo_path": str(photo_path),
        "video_path": str(video_path),
        "result_url": None,
        "error": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "completed_at": None,
        "frames_processed": 0,
        "options": {
            "target_fps": target_fps,
            "skip_frames": skip_frames,
            "scene_detection": scene_detection,
            "temporal_smoothing": temporal_smoothing,
            "upscale": upscale,
        },
    }
    await _save_job(job_id, job_data)

    # Dispatch Celery task
    try:
        from app.workers.celery_worker import process_swap_task

        process_swap_task.delay(job_id)
    except Exception as e:
        # If Celery is not available, log warning
        print(f"[API] Warning: Could not dispatch Celery task: {e}")
        job_data["message"] = "Job created (worker dispatch pending)"
        await _save_job(job_id, job_data)

    return SwapResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job created successfully. Processing will start shortly.",
        created_at=now,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Получить статус задачи."""
    job = await _get_job(job_id)

    return JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        progress=job["progress"],
        current_step=job.get("current_step", ""),
        message=job.get("message", ""),
        created_at=datetime.fromisoformat(job["created_at"]),
        updated_at=datetime.fromisoformat(job["updated_at"]),
        result_url=job.get("result_url"),
        error=job.get("error"),
    )


@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_result(job_id: str):
    """Получить результат задачи."""
    job = await _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job['status']}",
        )

    if not job.get("result_url"):
        raise HTTPException(status_code=404, detail="Result file not found")

    return JobResultResponse(
        job_id=job["job_id"],
        status=JobStatus.COMPLETED,
        result_url=job["result_url"],
        duration_sec=0.0,  # TODO: calculate from timestamps
        frames_processed=job.get("frames_processed", 0),
        created_at=datetime.fromisoformat(job["created_at"]),
        completed_at=datetime.fromisoformat(job["completed_at"])
        if job.get("completed_at")
        else datetime.utcnow(),
    )


@router.get("/jobs")
async def list_jobs():
    """Список последних задач (для UI)."""
    keys = []
    async for key in redis_client.scan_iter(match="job:*", count=100):
        keys.append(key)

    jobs = []
    for key in sorted(keys, reverse=True)[:20]:  # Last 20 jobs
        data = await redis_client.get(key)
        if data:
            job = json.loads(data)
            jobs.append({
                "job_id": job["job_id"],
                "status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"],
                "message": job.get("message", ""),
            })

    return {"jobs": jobs}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Удалить задачу и связанные файлы."""
    job = await _get_job(job_id)

    # Delete uploads
    job_dir = settings.UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    # Delete results
    result_dir = settings.RESULTS_DIR / job_id
    if result_dir.exists():
        shutil.rmtree(result_dir)

    # Delete from Redis
    await redis_client.delete(_job_key(job_id))

    return {"message": f"Job {job_id} deleted"}
