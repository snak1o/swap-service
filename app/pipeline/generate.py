"""
Wan2.2-Animate-14B Replace — полная замена персонажа в видео.

Простой pipeline:
  Фото (reference) + Видео (source) → Wan2.2 Replace → Готовое видео
"""

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WanReplace:
    """
    Wan2.2-Animate-14B Replace mode.

    Вход: фото целевого человека + видео с исходным человеком
    Выход: видео где исходный человек заменён на целевого
    """

    def __init__(self):
        from app.config import settings
        self.settings = settings
        self.wan_repo_dir = settings.WAN_REPO_DIR
        self.ckpt_dir = settings.WAN_CKPT_DIR

    def replace(
        self,
        photo_path: str,
        video_path: str,
        output_path: str,
    ) -> str:
        """
        Заменить персонажа в видео.

        Args:
            photo_path: путь к фото нового персонажа
            video_path: путь к исходному видео
            output_path: путь для результата

        Returns:
            путь к готовому видео
        """
        work_dir = tempfile.mkdtemp(prefix="wan_replace_")

        try:
            # Step 1: Preprocessing (skeleton, face encoding, pose)
            preprocess_dir = os.path.join(work_dir, "preprocess")
            self._preprocess(video_path, photo_path, preprocess_dir)

            # Step 2: Wan2.2 Inference (Replace mode)
            gen_dir = os.path.join(work_dir, "generated")
            self._generate(preprocess_dir, gen_dir)

            # Step 3: Собрать результат
            result = self._collect_result(gen_dir, output_path, video_path)

            logger.info(f"[WanReplace] Done! Result: {result}")
            return result

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._cleanup_gpu()

    def _preprocess(self, video_path: str, photo_path: str, save_path: str):
        """Wan2.2 preprocessing: скелет, face encoding, позы."""
        os.makedirs(save_path, exist_ok=True)
        s = self.settings

        script = os.path.join(
            self.wan_repo_dir, "wan", "modules", "animate",
            "preprocess", "preprocess_data.py",
        )
        ckpt = os.path.join(self.ckpt_dir, "process_checkpoint")

        cmd = [
            sys.executable, script,
            "--ckpt_path", ckpt,
            "--video_path", video_path,
            "--refer_path", photo_path,
            "--save_path", save_path,
            "--resolution_area", str(s.WAN_RESOLUTION_W), str(s.WAN_RESOLUTION_H),
            "--iterations", str(s.WAN_PREPROCESS_ITERATIONS),
            "--k", str(s.WAN_PREPROCESS_K),
            "--w_len", "1",
            "--h_len", "1",
        ]

        if s.WAN_REPLACE_FLAG:
            cmd.append("--replace_flag")

        logger.info(f"[WanReplace] Preprocessing...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=self.wan_repo_dir, timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Preprocessing failed: {result.stderr[:500]}")
        logger.info("[WanReplace] Preprocessing done")

    def _generate(self, src_root_path: str, output_dir: str):
        """Wan2.2 inference — Replace mode."""
        os.makedirs(output_dir, exist_ok=True)
        s = self.settings

        script = os.path.join(self.wan_repo_dir, "generate.py")

        cmd = [
            sys.executable, script,
            "--task", "animate-14B",
            "--ckpt_dir", self.ckpt_dir,
            "--src_root_path", src_root_path,
            "--refert_num", str(s.WAN_REFERT_NUM),
            "--save_file", output_dir,
        ]

        if s.WAN_REPLACE_FLAG:
            cmd.append("--replace_flag")
        if s.WAN_USE_RELIGHTING_LORA:
            cmd.append("--use_relighting_lora")
        if s.WAN_OFFLOAD_MODEL:
            cmd.extend(["--offload_model", "True", "--convert_model_dtype"])

        logger.info("[WanReplace] Generating (this takes a while)...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=self.wan_repo_dir, timeout=7200,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Generation failed: {result.stderr[:500]}")
        logger.info("[WanReplace] Generation done")

    def _collect_result(
        self, gen_dir: str, output_path: str, source_video: str,
    ) -> str:
        """Найти результат Wan2.2 и сохранить как финальное видео."""
        # Wan2.2 выводит .mp4 файлы
        videos = sorted(Path(gen_dir).rglob("*.mp4"))

        if videos:
            # Копируем первый результат + добавляем аудио из оригинала
            self._merge_audio(str(videos[0]), source_video, output_path)
            return output_path

        # Если нет видео — собираем из кадров
        frames = sorted(Path(gen_dir).rglob("*.png"))
        if not frames:
            frames = sorted(Path(gen_dir).rglob("*.jpg"))

        if not frames:
            raise RuntimeError(f"No output found in {gen_dir}")

        # Собираем видео из кадров через ffmpeg
        from app.config import settings
        fps = settings.TARGET_FPS
        frame_pattern = str(frames[0].parent / "%04d.png")

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
        ], capture_output=True, timeout=120)

        # Добавить аудио
        self._merge_audio(output_path, source_video, output_path)
        return output_path

    def _merge_audio(self, video_path: str, audio_source: str, output: str):
        """Перенести аудио из оригинального видео в результат."""
        temp_out = output + ".tmp.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_source,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                temp_out,
            ], capture_output=True, timeout=120)

            if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
                shutil.move(temp_out, output)
            # Если ffmpeg не смог добавить аудио — оставляем видео без звука
        except Exception:
            pass
        finally:
            if os.path.exists(temp_out):
                os.unlink(temp_out)

    def _cleanup_gpu(self):
        """Освободить GPU VRAM."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
