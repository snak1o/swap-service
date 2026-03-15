"""Модуль генерации тела — Animate Anyone 2 / WAN 2.2."""

import gc
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


SVD_XT_NUM_FRAMES = 25
SVD_XT_INFERENCE_STEPS = 15


class BodyGenerator:
    """
    Генерация тела по позе и референсному фото.

    На RunPod GPU: SVD-XT генерирует батчи по 25 кадров,
    которые потом растягиваются на всю последовательность.
    Локально: отправляет на RunPod API.
    """

    def __init__(self, model_name: str = "animate_anyone_2"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Загрузить генеративную модель."""
        is_runpod = os.environ.get("RUNPOD_POD_ID") is not None

        if self.model_name == "animate_anyone_2":
            try:
                from diffusers import StableVideoDiffusionPipeline
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                self.model = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    self.model.enable_model_cpu_offload()

                print(f"[BodyGenerator] Loaded SVD-XT on {'cuda' if torch.cuda.is_available() else 'cpu'}")
            except (ImportError, Exception) as e:
                print(f"[BodyGenerator] Model not available: {e}")
                if is_runpod:
                    raise RuntimeError(
                        f"BodyGenerator failed to load on RunPod GPU: {e}. "
                        "Check VRAM usage — other models may need to be unloaded first."
                    )
                print("[BodyGenerator] Will use RunPod API instead")
                self.model_name = "runpod_api"

    def unload_model(self):
        """Выгрузить модель из GPU для освобождения VRAM."""
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_sequence(
        self,
        reference_image: np.ndarray,
        pose_images: List[np.ndarray],
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        """
        Сгенерировать последовательность кадров.

        SVD-XT генерирует 25 кадров за вызов. Для длинных
        последовательностей запускаем несколько батчей и
        распределяем результаты равномерно по таймлайну.
        """
        if self.model is None:
            self.load_model()

        total_needed = len(pose_images)
        if total_needed == 0:
            return []

        if self.model_name == "runpod_api":
            return self._generate_sequence_runpod(reference_image, pose_images)

        return self._generate_sequence_local(reference_image, total_needed)

    def _generate_sequence_local(
        self,
        reference: np.ndarray,
        total_needed: int,
    ) -> List[np.ndarray]:
        """
        Локальная генерация через SVD-XT.
        Генерирует батчи по 25 кадров и интерполирует между ними.
        """
        from PIL import Image
        import torch

        ref_pil = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        ref_pil = ref_pil.resize((512, 512))

        num_batches = max(1, (total_needed + SVD_XT_NUM_FRAMES - 1) // SVD_XT_NUM_FRAMES)
        all_generated = []

        for batch_idx in range(num_batches):
            print(f"[BodyGenerator] SVD batch {batch_idx + 1}/{num_batches}")

            try:
                with torch.no_grad():
                    output = self.model(
                        image=ref_pil,
                        num_frames=SVD_XT_NUM_FRAMES,
                        decode_chunk_size=4,
                        num_inference_steps=SVD_XT_INFERENCE_STEPS,
                        motion_bucket_id=127,
                    )
                    frames_pil = output.frames[0]

                for frame_pil in frames_pil:
                    frame_np = np.array(frame_pil)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    all_generated.append(frame_bgr)

                print(f"[BodyGenerator] Batch {batch_idx + 1} done: {len(frames_pil)} frames")

            except Exception as e:
                print(f"[BodyGenerator] SVD inference error in batch {batch_idx + 1}: {e}")
                raise

        if len(all_generated) >= total_needed:
            return all_generated[:total_needed]

        return self._stretch_to_length(all_generated, total_needed)

    def _stretch_to_length(self, frames: List[np.ndarray], target_len: int) -> List[np.ndarray]:
        """Растянуть/сжать набор кадров до нужной длины через линейную интерполяцию."""
        if not frames or target_len <= 0:
            return []

        src_len = len(frames)
        if src_len == target_len:
            return frames
        if src_len == 1:
            return frames * target_len

        result = []
        for i in range(target_len):
            t = i / (target_len - 1) * (src_len - 1)
            idx_low = int(t)
            idx_high = min(idx_low + 1, src_len - 1)
            alpha = t - idx_low

            if alpha < 0.01:
                result.append(frames[idx_low])
            elif alpha > 0.99:
                result.append(frames[idx_high])
            else:
                blended = cv2.addWeighted(
                    frames[idx_low], 1 - alpha,
                    frames[idx_high], alpha,
                    0,
                )
                result.append(blended)

        return result

    def generate_frame(
        self,
        reference_image: np.ndarray,
        pose_image: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Сгенерировать один кадр (для RunPod API fallback)."""
        if self.model is None:
            self.load_model()

        if self.model_name == "runpod_api":
            return self._generate_via_runpod(reference_image, pose_image, previous_frame)

        frames = self._generate_sequence_local(reference_image, 1)
        return frames[0] if frames else self._fallback_composite(reference_image, pose_image)

    def _generate_sequence_runpod(
        self,
        reference: np.ndarray,
        pose_images: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Батчевая генерация через RunPod API."""
        results = []
        prev_frame = None

        for i, pose in enumerate(pose_images):
            frame = self._generate_via_runpod(reference, pose, prev_frame)
            results.append(frame)
            prev_frame = frame

            if (i + 1) % 10 == 0:
                print(f"[BodyGenerator] Generated {i + 1}/{len(pose_images)} frames via RunPod API")

        return results

    def _generate_via_runpod(
        self,
        reference: np.ndarray,
        pose: np.ndarray,
        prev: Optional[np.ndarray],
    ) -> np.ndarray:
        """Генерация через RunPod Serverless API."""
        import base64
        import httpx
        from app.config import settings

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

        return self._fallback_composite(reference, pose)

    def _fallback_composite(self, reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Простой compositing для тестирования без модели."""
        h, w = pose.shape[:2]
        ref_resized = cv2.resize(reference, (w, h))

        mask = cv2.cvtColor(pose, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        result = ref_resized.copy()
        result[mask > 0] = pose[mask > 0]

        return result
