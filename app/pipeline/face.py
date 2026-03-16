"""Модуль улучшения лица — GFPGAN + InsightFace swap."""

import os
from typing import Optional

import cv2
import numpy as np


class FaceRefiner:
    """
    Улучшение и замена лица:
    1. Face detection в сгенерированном кадре
    2. Face swap с высоким качеством (InsightFace)
    3. Face enhancement (GFPGAN / CodeFormer)
    """

    def __init__(self):
        self.face_enhancer = None
        self.face_swapper = None
        self.face_analyzer = None

    def _get_gfpgan_path(self) -> str:
        """Get GFPGAN model path from config (not hardcoded)."""
        from app.config import settings
        return os.path.join(settings.MODELS_DIR, "GFPGANv1.4.pth")

    def _get_insightface_root(self) -> str:
        """Get InsightFace root dir from config (not hardcoded)."""
        from app.config import settings
        return os.path.join(settings.MODELS_DIR, "insightface")

    def load_models(self):
        """Загрузить модели для face processing."""
        # GFPGAN for face enhancement
        try:
            from gfpgan import GFPGANer

            gfpgan_path = self._get_gfpgan_path()
            self.face_enhancer = GFPGANer(
                model_path=gfpgan_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
            print(f"[FaceRefiner] Loaded GFPGAN from {gfpgan_path}")
        except (ImportError, Exception) as e:
            print(f"[FaceRefiner] GFPGAN not available: {e}")

        # InsightFace for face analysis + swap
        try:
            import insightface
            from insightface.app import FaceAnalysis

            insightface_root = self._get_insightface_root()
            os.environ["INSIGHTFACE_HOME"] = insightface_root

            self.face_analyzer = FaceAnalysis(
                name="buffalo_l",
                root=insightface_root,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

            # Load swapper model
            self.face_swapper = insightface.model_zoo.get_model(
                "inswapper_128.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print("[FaceRefiner] Loaded InsightFace swapper")
        except (ImportError, Exception) as e:
            print(f"[FaceRefiner] InsightFace not available: {e}")

    def refine_frame(
        self,
        generated_frame: np.ndarray,
        reference_photo: np.ndarray,
    ) -> np.ndarray:
        """
        Улучшить лицо в сгенерированном кадре.

        1. Найти лицо в reference_photo -> извлечь face embedding
        2. Найти лицо в generated_frame
        3. Заменить лицо (swap) для точной идентичности
        4. Улучшить качество лица (enhance)

        Args:
            generated_frame: сгенерированный кадр тела
            reference_photo: оригинальное фото целевого человека

        Returns:
            кадр с улучшенным лицом
        """
        if self.face_analyzer is None:
            self.load_models()

        result = generated_frame.copy()

        # Step 1: Face swap (if InsightFace available)
        if self.face_swapper is not None and self.face_analyzer is not None:
            result = self._swap_face(result, reference_photo)

        # Step 2: Face enhancement (if GFPGAN available)
        if self.face_enhancer is not None:
            result = self._enhance_face(result)

        return result

    def _swap_face(self, frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Заменить лицо в кадре на лицо из reference."""
        # Detect faces
        ref_faces = self.face_analyzer.get(reference)
        frame_faces = self.face_analyzer.get(frame)

        if not ref_faces or not frame_faces:
            return frame  # No face detected, return as-is

        # Use first detected face
        ref_face = ref_faces[0]
        frame_face = frame_faces[0]

        # Swap face
        result = self.face_swapper.get(frame, frame_face, ref_face, paste_back=True)
        return result

    def _enhance_face(self, frame: np.ndarray) -> np.ndarray:
        """Улучшить качество лица через GFPGAN."""
        try:
            _, _, output = self.face_enhancer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            return output
        except Exception:
            return frame

    def refine_sequence(
        self,
        frames: list,
        reference_photo: np.ndarray,
    ) -> list:
        """Улучшить лица во всех кадрах."""
        results = []
        for i, frame in enumerate(frames):
            refined = self.refine_frame(frame, reference_photo)
            results.append(refined)

            if (i + 1) % 10 == 0:
                print(f"[FaceRefiner] Refined {i + 1}/{len(frames)} frames")

        return results
