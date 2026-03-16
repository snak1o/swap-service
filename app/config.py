"""Конфигурация приложения."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Settings from environment variables."""

    # --- Models (Network Volume) ---
    MODELS_DIR: str = os.getenv("MODELS_DIR", "/runpod-volume/models")

    # --- Wan2.2-Animate-14B ---
    WAN_MODEL_NAME: str = os.getenv("WAN_MODEL_NAME", "Wan-AI/Wan2.2-Animate-14B")
    WAN_CKPT_DIR: str = os.getenv(
        "WAN_CKPT_DIR",
        os.path.join(os.getenv("MODELS_DIR", "/runpod-volume/models"), "Wan2.2-Animate-14B"),
    )
    WAN_REPO_DIR: str = os.getenv("WAN_REPO_DIR", "/app/Wan2.2")
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

    # --- Processing ---
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))


settings = Settings()
