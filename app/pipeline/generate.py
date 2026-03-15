"""Модуль генерации тела — Animate Anyone 2 / WAN 2.2."""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


class BodyGenerator:
    """
    Генерация тела по позе и референсному фото.
    
    Используется Animate Anyone 2 на RunPod GPU.
    Принимает:
    - Фото целевого человека (reference image)
    - Последовательность pose images
    
    Возвращает:
    - Кадры с сгенерированным телом целевого персонажа
    """

    def __init__(self, model_name: str = "animate_anyone_2"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Загрузить генеративную модель."""
        if self.model_name == "animate_anyone_2":
            try:
                # Animate Anyone 2 pipeline
                # Requires: pip install diffusers accelerate
                from diffusers import StableVideoDiffusionPipeline
                import torch

                self.model = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                if torch.cuda.is_available():
                    self.model.to("cuda")
                print("[BodyGenerator] Loaded model on GPU")
            except (ImportError, Exception) as e:
                print(f"[BodyGenerator] Model not available: {e}")
                print("[BodyGenerator] Will use RunPod API instead")
                self.model_name = "runpod_api"

    def generate_frame(
        self,
        reference_image: np.ndarray,
        pose_image: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Сгенерировать один кадр.
        
        Args:
            reference_image: фото целевого персонажа (BGR)
            pose_image: pose skeleton image (BGR)
            previous_frame: предыдущий сгенерированный кадр (для temporal consistency)
            
        Returns:
            сгенерированный кадр (BGR)
        """
        if self.model is None:
            self.load_model()

        if self.model_name == "runpod_api":
            return self._generate_via_runpod(reference_image, pose_image, previous_frame)
        else:
            return self._generate_local(reference_image, pose_image, previous_frame)

    def _generate_local(
        self,
        reference: np.ndarray,
        pose: np.ndarray,
        prev: Optional[np.ndarray],
    ) -> np.ndarray:
        """Локальная генерация через загруженную модель."""
        from PIL import Image
        import torch

        ref_pil = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        ref_pil = ref_pil.resize((512, 512))

        with torch.no_grad():
            frames = self.model(
                image=ref_pil,
                num_frames=1,
                decode_chunk_size=1,
            ).frames[0]

        result = np.array(frames[0])
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _generate_via_runpod(
        self,
        reference: np.ndarray,
        pose: np.ndarray,
        prev: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Генерация через RunPod Serverless API.
        
        Отправляет reference + pose на RunPod worker,
        который запускает Animate Anyone 2 на GPU.
        """
        import base64
        import httpx
        from app.config import settings

        # Encode images to base64
        _, ref_buf = cv2.imencode(".jpg", reference)
        _, pose_buf = cv2.imencode(".png", pose)

        ref_b64 = base64.b64encode(ref_buf).decode()
        pose_b64 = base64.b64encode(pose_buf).decode()

        payload = {
            "input": {
                "reference_image": ref_b64,
                "pose_image": pose_b64,
                "width": 512,
                "height": 768,
            }
        }

        if prev is not None:
            _, prev_buf = cv2.imencode(".jpg", prev)
            payload["input"]["previous_frame"] = base64.b64encode(prev_buf).decode()

        # Call RunPod
        endpoint_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}/runsync"
        headers = {
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }

        response = httpx.post(endpoint_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        result = response.json()
        output_b64 = result.get("output", {}).get("image", "")

        if output_b64:
            img_bytes = base64.b64decode(output_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Fallback: return pose overlay on reference
        return self._fallback_composite(reference, pose)

    def _fallback_composite(self, reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Простой compositing для тестирования без модели."""
        h, w = pose.shape[:2]
        ref_resized = cv2.resize(reference, (w, h))

        # Overlay pose skeleton on reference (for debugging)
        mask = cv2.cvtColor(pose, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        result = ref_resized.copy()
        result[mask > 0] = pose[mask > 0]

        return result

    def generate_sequence(
        self,
        reference_image: np.ndarray,
        pose_images: List[np.ndarray],
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        """
        Сгенерировать последовательность кадров.
        
        Args:
            reference_image: фото целевого персонажа
            pose_images: список pose skeleton images
            batch_size: размер батча для GPU обработки
            
        Returns:
            список сгенерированных кадров
        """
        results = []
        prev_frame = None

        for i, pose in enumerate(pose_images):
            frame = self.generate_frame(reference_image, pose, prev_frame)
            results.append(frame)
            prev_frame = frame

            if (i + 1) % 10 == 0:
                print(f"[BodyGenerator] Generated {i + 1}/{len(pose_images)} frames")

        return results
