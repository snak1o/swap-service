"""Модуль извлечения позы из видео (DWPose)."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


class PoseExtractor:
    """
    Извлечение скелетных ключевых точек из видеокадров.
    
    На RunPod используется DWPose / OpenPose.
    Локально — MediaPipe как fallback.
    """

    def __init__(self, model_name: str = "dwpose"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Загрузить модель pose estimation."""
        if self.model_name == "dwpose":
            import os
            is_runpod = os.environ.get("RUNPOD_POD_ID") is not None

            try:
                import torch
                from easy_dwpose import DWposeDetector

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = DWposeDetector(device=device)
                self.model_name = "dwpose_easy"
                print(f"[PoseExtractor] Loaded easy-dwpose on {device}")

            except Exception as e:
                print(f"[PoseExtractor] Failed to load DWPose: {e}")
                if is_runpod:
                    raise RuntimeError(f"DWPose failed to load on RunPod: {e}")
                else:
                    print("[PoseExtractor] Falling back to MediaPipe")
                    self._load_mediapipe()
        else:
            self._load_mediapipe()

    def _load_mediapipe(self):
        """MediaPipe fallback для локальной разработки."""
        try:
            import mediapipe as mp
            self.model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.model_name = "mediapipe"
            print("[PoseExtractor] Loaded MediaPipe Pose model")
        except ImportError:
            print("[PoseExtractor] Warning: No pose model available")

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Извлечь позу из одного кадра.
        
        Returns:
            dict с ключами:
            - keypoints: List[List[float]] — [[x, y, confidence], ...]
            - bbox: [x1, y1, x2, y2] — bounding box человека
            - pose_image: np.ndarray — визуализация скелета (для передачи в генератор)
        """
        if self.model is None:
            self.load_model()

        h, w = frame.shape[:2]

        if self.model_name in ["dwpose", "dwpose_easy"]:
            return self._extract_dwpose(frame)
        elif self.model_name == "mediapipe":
            return self._extract_mediapipe(frame, h, w)
        else:
            return self._empty_result(h, w)

    def _extract_dwpose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Извлечение через DWPose."""
        from PIL import Image
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_image = self.model(pil_image)
            pose_np = np.array(pose_image)
            return {
                "keypoints": [],
                "bbox": [0, 0, frame.shape[1], frame.shape[0]],
                "pose_image": cv2.cvtColor(pose_np, cv2.COLOR_RGB2BGR),
            }
        except Exception as e:
            print(f"[PoseExtractor] Error in DWPose: {e}")
            h, w = frame.shape[:2]
            return self._empty_result(h, w)

    def _extract_mediapipe(self, frame: np.ndarray, h: int, w: int) -> Dict[str, Any]:
        """Извлечение через MediaPipe."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)

        keypoints = []
        bbox = [0, 0, w, h]

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.x * w, lm.y * h, lm.visibility])

            # Calculate bbox from keypoints
            xs = [kp[0] for kp in keypoints if kp[2] > 0.3]
            ys = [kp[1] for kp in keypoints if kp[2] > 0.3]
            if xs and ys:
                margin = 30
                bbox = [
                    max(0, min(xs) - margin),
                    max(0, min(ys) - margin),
                    min(w, max(xs) + margin),
                    min(h, max(ys) + margin),
                ]

        # Draw pose skeleton on blank canvas
        pose_image = self._draw_skeleton(keypoints, h, w)

        return {
            "keypoints": keypoints,
            "bbox": bbox,
            "pose_image": pose_image,
        }

    def _draw_skeleton(self, keypoints: List[List[float]], h: int, w: int) -> np.ndarray:
        """Нарисовать скелет на чёрном фоне."""
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not keypoints:
            return canvas

        # MediaPipe connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                kp1 = keypoints[start_idx]
                kp2 = keypoints[end_idx]
                if kp1[2] > 0.3 and kp2[2] > 0.3:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.3:
                cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        return canvas

    def _empty_result(self, h: int, w: int) -> Dict[str, Any]:
        return {
            "keypoints": [],
            "bbox": [0, 0, w, h],
            "pose_image": np.zeros((h, w, 3), dtype=np.uint8),
        }

    def extract_from_video(
        self,
        video_path: str,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Извлечь позы из всех кадров видео.
        
        Args:
            video_path: путь к видео
            skip_frames: пропускать каждые N кадров (0 = все)
            max_frames: максимум кадров для обработки
            
        Returns:
            список dict с позами для каждого кадра
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        poses = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            if max_frames and len(poses) >= max_frames:
                break

            pose = self.extract_from_frame(frame)
            pose["frame_idx"] = frame_idx
            poses.append(pose)

            frame_idx += 1

        cap.release()
        return poses

    def unload_model(self):
        """Выгрузить модель из GPU для освобождения VRAM."""
        if self.model is not None:
            if self.model_name == "mediapipe":
                try:
                    self.model.close()
                except Exception:
                    pass
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

    def save_poses(self, poses: List[Dict[str, Any]], output_dir: str):
        """Сохранить pose images для передачи в генератор."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for i, pose in enumerate(poses):
            # Save pose image
            img_path = out_path / f"pose_{i:06d}.png"
            cv2.imwrite(str(img_path), pose["pose_image"])

            # Save keypoints as JSON
            kp_data = {
                "frame_idx": pose.get("frame_idx", i),
                "keypoints": pose["keypoints"],
                "bbox": pose["bbox"],
            }
            json_path = out_path / f"pose_{i:06d}.json"
            with open(json_path, "w") as f:
                json.dump(kp_data, f)

        return str(out_path)
