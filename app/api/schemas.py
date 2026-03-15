"""Pydantic-схемы для API запросов и ответов."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Статусы обработки задачи."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    POSE_EXTRACTION = "pose_extraction"
    SEGMENTATION = "segmentation"
    BODY_GENERATION = "body_generation"
    FACE_REFINEMENT = "face_refinement"
    COMPOSITING = "compositing"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SwapRequest(BaseModel):
    """Запрос на создание swap-задачи (метаданные для multipart upload)."""

    target_fps: Optional[int] = Field(None, description="Целевой FPS (null = как в исходном)")
    skip_frames: Optional[int] = Field(0, description="Обрабатывать каждый N-й кадр (0 = все)")
    scene_detection: bool = Field(True, description="Автоматическое разбиение на сцены")
    temporal_smoothing: bool = Field(True, description="Применять temporal smoothing")
    upscale: bool = Field(False, description="Повысить разрешение результата (2x)")


class SwapResponse(BaseModel):
    """Ответ при создании задачи."""

    job_id: str
    status: JobStatus
    message: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Ответ со статусом задачи."""

    job_id: str
    status: JobStatus
    progress: float = Field(0.0, description="Прогресс 0.0-1.0")
    current_step: str = ""
    message: str = ""
    created_at: datetime
    updated_at: datetime
    result_url: Optional[str] = None
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    """Ответ с результатом."""

    job_id: str
    status: JobStatus
    result_url: str
    duration_sec: float
    frames_processed: int
    created_at: datetime
    completed_at: datetime
