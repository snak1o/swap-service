"""Модуль композитинга — наложение сгенерированного персонажа на фон + финальная сборка."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class Compositor:
    """
    Финальная сборка видео:
    1. Наложение сгенерированного персонажа на чистый фон
    2. Edge blending (сглаживание краёв)
    3. Color matching (согласование цвета с фоном)
    4. Temporal smoothing (устранение мерцания)
    5. Сборка кадров в видео через FFmpeg
    """

    def __init__(self):
        pass

    def composite_frame(
        self,
        generated_person: np.ndarray,
        clean_background: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Наложить сгенерированного персонажа на фон.
        
        Args:
            generated_person: сгенерированный кадр с персонажем
            clean_background: чистый фон (без исходного человека)
            person_mask: маска персонажа (если None — генерируем)
            
        Returns:
            финальный кадр
        """
        h, w = clean_background.shape[:2]
        person_resized = cv2.resize(generated_person, (w, h))

        if person_mask is None:
            # Generate mask from person frame (non-black areas)
            gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
            _, person_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        mask_resized = cv2.resize(person_mask, (w, h))

        # Feather edges for smooth blending
        mask_float = self._feather_mask(mask_resized, feather_radius=7)

        # Color match person to background lighting
        person_matched = self._color_match(person_resized, clean_background, mask_resized)

        # Alpha blending
        mask_3ch = np.stack([mask_float] * 3, axis=-1)
        result = (person_matched * mask_3ch + clean_background * (1 - mask_3ch)).astype(np.uint8)

        return result

    def _feather_mask(self, mask: np.ndarray, feather_radius: int = 5) -> np.ndarray:
        """Размыть края маски для плавного перехода."""
        # Erode slightly to avoid halo
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, kernel, iterations=1)

        # Gaussian blur for feathering
        blurred = cv2.GaussianBlur(eroded, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

        return blurred.astype(np.float32) / 255.0

    def _color_match(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Согласовать цветовую палитру source с target.
        Использует простое histogram matching в LAB пространстве.
        """
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute stats of target (background area)
        bg_mask = cv2.bitwise_not(mask)
        for ch in range(3):
            src_ch = src_lab[:, :, ch]
            tgt_ch = tgt_lab[:, :, ch]

            # Mean and std of target background
            tgt_mean = np.mean(tgt_ch[bg_mask > 128]) if np.any(bg_mask > 128) else np.mean(tgt_ch)
            tgt_std = np.std(tgt_ch[bg_mask > 128]) if np.any(bg_mask > 128) else np.std(tgt_ch)

            src_mean = np.mean(src_ch[mask > 128]) if np.any(mask > 128) else np.mean(src_ch)
            src_std = np.std(src_ch[mask > 128]) if np.any(mask > 128) else np.std(src_ch)

            if src_std > 0:
                # Subtle adjustment (blend 30% towards target)
                blend = 0.3
                adjusted = (src_ch - src_mean) * (1 + blend * (tgt_std / src_std - 1)) + src_mean + blend * (tgt_mean - src_mean)
                src_lab[:, :, ch] = np.clip(adjusted, 0, 255)

        result = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def temporal_smooth(
        self,
        frames: List[np.ndarray],
        window_size: int = 3,
    ) -> List[np.ndarray]:
        """
        Временное сглаживание для устранения мерцания.
        Применяет скользящее среднее по кадрам.
        """
        if len(frames) < window_size:
            return frames

        smoothed = []
        half_w = window_size // 2

        for i in range(len(frames)):
            start = max(0, i - half_w)
            end = min(len(frames), i + half_w + 1)

            # Weighted average (center frame has more weight)
            weights = []
            window_frames = []
            for j in range(start, end):
                weight = 1.0 if j == i else 0.3
                weights.append(weight)
                window_frames.append(frames[j].astype(np.float32))

            total_weight = sum(weights)
            blended = sum(f * w for f, w in zip(window_frames, weights)) / total_weight
            smoothed.append(blended.astype(np.uint8))

        return smoothed

    def assemble_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        audio_source: Optional[str] = None,
    ) -> str:
        """
        Собрать кадры в видео через FFmpeg.
        
        Args:
            frames: список кадров BGR
            output_path: путь для выходного видео
            fps: частота кадров
            audio_source: путь к видео для извлечения аудио
            
        Returns:
            путь к финальному видео
        """
        if not frames:
            raise ValueError("No frames to assemble")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        temp_video = str(output.with_suffix(".temp.mp4"))

        # Write frames to temp video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)
        writer.release()

        # Use FFmpeg to add audio and proper encoding
        if audio_source and Path(audio_source).exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_source,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                str(output),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                str(output),
            ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            Path(temp_video).unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            # If FFmpeg fails, keep the OpenCV video
            import shutil
            shutil.move(temp_video, str(output))

        print(f"[Compositor] Video saved: {output} ({len(frames)} frames, {fps} fps)")
        return str(output)

    def get_video_info(self, video_path: str) -> dict:
        """Получить информацию о видео."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1, cap.get(cv2.CAP_PROP_FPS)),
        }
        cap.release()
        return info
