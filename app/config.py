"""Конфигурация приложения."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # --- App ---
    APP_NAME: str = "Full-Body Swap Service"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # --- API ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # --- Redis ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- Celery ---
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

    # --- MinIO / S3 ---
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "swap-media")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # --- RunPod ---
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")

    # --- Models (Network Volume) ---
    MODELS_DIR: str = os.getenv("MODELS_DIR", "/runpod-volume/models")

    # --- Wan2.2-Animate-14B ---
    WAN_MODEL_NAME: str = os.getenv("WAN_MODEL_NAME", "Wan-AI/Wan2.2-Animate-14B")
    WAN_CKPT_DIR: str = os.getenv(
        "WAN_CKPT_DIR",
        os.path.join(os.getenv("MODELS_DIR", "/runpod-volume/models"), "Wan2.2-Animate-14B"),
    )
    WAN_RESOLUTION_W: int = int(os.getenv("WAN_RESOLUTION_W", "1280"))
    WAN_RESOLUTION_H: int = int(os.getenv("WAN_RESOLUTION_H", "720"))
    WAN_REPLACE_FLAG: bool = os.getenv("WAN_REPLACE_FLAG", "true").lower() == "true"
    WAN_USE_RELIGHTING_LORA: bool = os.getenv("WAN_USE_RELIGHTING_LORA", "true").lower() == "true"
    WAN_OFFLOAD_MODEL: bool = os.getenv("WAN_OFFLOAD_MODEL", "false").lower() == "true"
    WAN_REFERT_NUM: int = int(os.getenv("WAN_REFERT_NUM", "1"))
    WAN_PREPROCESS_ITERATIONS: int = int(os.getenv("WAN_PREPROCESS_ITERATIONS", "3"))
    WAN_PREPROCESS_K: int = int(os.getenv("WAN_PREPROCESS_K", "7"))

    # --- HuggingFace ---
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "results"
    TEMP_DIR: Path = BASE_DIR / "temp"
    WAN_REPO_DIR: str = os.getenv("WAN_REPO_DIR", "/app/Wan2.2")

    # --- Processing ---
    MAX_VIDEO_DURATION_SEC: int = int(os.getenv("MAX_VIDEO_DURATION_SEC", "120"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))
    SKIP_FRAMES: int = int(os.getenv("SKIP_FRAMES", "0"))  # 0 = process all frames

    def __init__(self):
        # Create directories
        for d in [self.UPLOAD_DIR, self.RESULTS_DIR, self.TEMP_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
