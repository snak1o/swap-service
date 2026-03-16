"""Модуль сегментации — отделение человека от фона (SAM 2)."""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


class BackgroundSegmenter:
    """
    Сегментация человека и фона.

    - На RunPod: SAM 2 для точной сегментации
    - Локально: GrabCut / простая фоновая модель как fallback
    """

    def __init__(self, model_name: str = "sam2"):
        self.model_name = model_name
        self.model = None

    def _get_sam2_checkpoint(self) -> str:
        """Get SAM2 checkpoint path from config (not hardcoded)."""
        from app.config import settings
        return os.path.join(settings.MODELS_DIR, "sam2.1_hiera_large.pt")

    def load_model(self):
        """Загрузить модель сегментации."""
        if self.model_name == "sam2":
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                checkpoint = self._get_sam2_checkpoint()
                model_cfg = "configs/sam2.1/sam2.1_hiera_l"

                sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
                self.model = SAM2ImagePredictor(sam2_model)
                import logging
                logging.getLogger("sam2.sam2_image_predictor").setLevel(logging.WARNING)
                print(f"[Segmenter] Loaded SAM 2.1 Hiera Large from {checkpoint}")
            except Exception as e:
                print(f"[Segmenter] SAM 2 not available ({e}), using fallback")
                self.model_name = "fallback"
        else:
            self.model_name = "fallback"

    def segment_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментировать кадр: разделить на person_mask и clean_background.

        Args:
            frame: BGR кадр
            bbox: [x1, y1, x2, y2] bounding box человека (из pose)

        Returns:
            (person_mask, clean_background)
            - person_mask: uint8 маска (255=person, 0=background)
            - clean_background: кадр с inpainted областью человека
        """
        if self.model is None:
            self.load_model()

        if self.model_name == "sam2" and self.model is not None:
            mask = self._segment_sam(frame, bbox)
        else:
            mask = self._segment_fallback(frame, bbox)

        # Clean background (inpaint where person was)
        clean_bg = self._inpaint_background(frame, mask)

        return mask, clean_bg

    def _segment_sam(self, frame: np.ndarray, bbox: Optional[List[float]]) -> np.ndarray:
        """Сегментация через SAM 2."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.model.set_image(rgb)

        if bbox:
            box = np.array(bbox)
            masks, scores, _ = self.model.predict(box=box, multimask_output=False)
        else:
            # Center point prompt
            h, w = frame.shape[:2]
            point = np.array([[w // 2, h // 2]])
            label = np.array([1])
            masks, scores, _ = self.model.predict(
                point_coords=point, point_labels=label, multimask_output=False
            )

        if masks.ndim == 3:
            mask = (masks[0] * 255).astype(np.uint8)
        else:
            mask = (masks[0, 0] * 255).astype(np.uint8) if masks.ndim == 4 else (masks * 255).astype(np.uint8)
            while mask.ndim > 2: mask = mask[0]

        return mask

    def _segment_fallback(self, frame: np.ndarray, bbox: Optional[List[float]]) -> np.ndarray:
        """
        Fallback сегментация через GrabCut.
        Работает без GPU, но менее точная.
        """
        h, w = frame.shape[:2]

        if bbox:
            rect = (
                int(max(0, bbox[0])),
                int(max(0, bbox[1])),
                int(min(w, bbox[2]) - max(0, bbox[0])),
                int(min(h, bbox[3]) - max(0, bbox[1])),
            )
        else:
            margin = int(min(w, h) * 0.05)
            rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            # If GrabCut fails, use simple threshold
            mask2 = np.zeros((h, w), dtype=np.uint8)
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                mask2[y1:y2, x1:x2] = 255

        # Smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

        return mask2

    def _inpaint_background(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Заполнить область человека через inpainting."""
        # Dilate mask slightly for cleaner inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)

        # OpenCV inpainting
        clean = cv2.inpaint(frame, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return clean

    def unload_model(self):
        """Выгрузить модель из GPU для освобождения VRAM."""
        if self.model is not None:
            del self.model
            self.model = None

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def segment_video_frames(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[List[float]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Сегментировать все кадры видео."""
        masks = []
        backgrounds = []

        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            mask, bg = self.segment_frame(frame, bbox)
            masks.append(mask)
            backgrounds.append(bg)

        return masks, backgrounds
