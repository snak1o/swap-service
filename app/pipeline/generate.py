"""Body generation using Wan2.2-Animate-14B — Replace mode for full-body swap."""

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BodyGenerator:
    """
    Generates video with replaced person using Wan2.2-Animate-14B (Replace mode).

    Pipeline:
    1. Preprocessing: extract skeleton, face encoding, poses from source video
    2. Inference: generate video with replaced character via Wan2.2 Animate
    3. Output: BGR frames with replaced person

    All paths and settings come from config (not hardcoded).
    """

    def __init__(self):
        from app.config import settings

        self.settings = settings
        self.wan_repo_dir = settings.WAN_REPO_DIR
        self.ckpt_dir = settings.WAN_CKPT_DIR
        self.resolution_w = settings.WAN_RESOLUTION_W
        self.resolution_h = settings.WAN_RESOLUTION_H
        self.replace_flag = settings.WAN_REPLACE_FLAG
        self.use_relighting_lora = settings.WAN_USE_RELIGHTING_LORA
        self.offload_model = settings.WAN_OFFLOAD_MODEL
        self.refert_num = settings.WAN_REFERT_NUM
        self.preprocess_iterations = settings.WAN_PREPROCESS_ITERATIONS
        self.preprocess_k = settings.WAN_PREPROCESS_K

    def generate_sequence(
        self,
        reference_image: np.ndarray,
        video_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Generate video with replaced person using Wan2.2 Animate Replace mode.

        Args:
            reference_image: BGR photo of the target person (new character)
            video_frames: list of BGR frames from the source video

        Returns:
            list of BGR frames with replaced person
        """
        if not video_frames:
            return []

        work_dir = tempfile.mkdtemp(prefix="wan_swap_")

        try:
            # Save reference image
            ref_path = os.path.join(work_dir, "reference.jpg")
            cv2.imwrite(ref_path, reference_image)

            # Save source video from frames
            video_path = os.path.join(work_dir, "source.mp4")
            self._save_frames_to_video(video_frames, video_path)

            # Step 1: Preprocessing
            preprocess_dir = os.path.join(work_dir, "process_results")
            self._run_preprocessing(video_path, ref_path, preprocess_dir)

            # Step 2: Wan2.2 Inference (Replace mode)
            output_dir = os.path.join(work_dir, "output")
            self._run_inference(preprocess_dir, output_dir)

            # Step 3: Read generated frames
            output_frames = self._read_output_frames(output_dir, len(video_frames))

            logger.info(f"[BodyGenerator] Generated {len(output_frames)} frames via Wan2.2 Replace")
            return output_frames

        finally:
            # Cleanup work directory
            shutil.rmtree(work_dir, ignore_errors=True)

    def _save_frames_to_video(self, frames: List[np.ndarray], output_path: str):
        """Save BGR frames to an MP4 video file."""
        from app.config import settings

        h, w = frames[0].shape[:2]
        fps = float(settings.TARGET_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)
        writer.release()

        logger.info(f"[BodyGenerator] Saved {len(frames)} frames to {output_path} ({w}x{h}, {fps}fps)")

    def _run_preprocessing(self, video_path: str, ref_path: str, save_path: str):
        """
        Run Wan2.2-Animate preprocessing: skeleton extraction, face encoding, pose retargeting.

        Calls: wan/modules/animate/preprocess/preprocess_data.py
        """
        os.makedirs(save_path, exist_ok=True)

        preprocess_script = os.path.join(
            self.wan_repo_dir, "wan", "modules", "animate", "preprocess", "preprocess_data.py"
        )
        process_ckpt = os.path.join(self.ckpt_dir, "process_checkpoint")

        cmd = [
            sys.executable, preprocess_script,
            "--ckpt_path", process_ckpt,
            "--video_path", video_path,
            "--refer_path", ref_path,
            "--save_path", save_path,
            "--resolution_area", str(self.resolution_w), str(self.resolution_h),
            "--iterations", str(self.preprocess_iterations),
            "--k", str(self.preprocess_k),
            "--w_len", "1",
            "--h_len", "1",
        ]

        if self.replace_flag:
            cmd.append("--replace_flag")

        logger.info(f"[BodyGenerator] Running preprocessing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.wan_repo_dir,
            timeout=600,  # 10 min max for preprocessing
        )

        if result.returncode != 0:
            logger.error(f"[BodyGenerator] Preprocessing failed:\n{result.stderr}")
            raise RuntimeError(f"Wan2.2 preprocessing failed: {result.stderr[:500]}")

        logger.info(f"[BodyGenerator] Preprocessing complete: {save_path}")

    def _run_inference(self, src_root_path: str, output_dir: str):
        """
        Run Wan2.2-Animate-14B inference in Replace mode.

        Calls: generate.py --task animate-14B --replace_flag --use_relighting_lora
        """
        os.makedirs(output_dir, exist_ok=True)

        generate_script = os.path.join(self.wan_repo_dir, "generate.py")

        cmd = [
            sys.executable, generate_script,
            "--task", "animate-14B",
            "--ckpt_dir", self.ckpt_dir,
            "--src_root_path", src_root_path,
            "--refert_num", str(self.refert_num),
            "--save_file", output_dir,
        ]

        if self.replace_flag:
            cmd.append("--replace_flag")

        if self.use_relighting_lora:
            cmd.append("--use_relighting_lora")

        if self.offload_model:
            cmd.extend(["--offload_model", "True", "--convert_model_dtype"])

        logger.info(f"[BodyGenerator] Running Wan2.2 inference: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.wan_repo_dir,
            timeout=7200,  # 2 hours max for inference
        )

        if result.returncode != 0:
            logger.error(f"[BodyGenerator] Inference failed:\n{result.stderr}")
            raise RuntimeError(f"Wan2.2 inference failed: {result.stderr[:500]}")

        logger.info(f"[BodyGenerator] Inference complete. Output: {output_dir}")

    def _read_output_frames(self, output_dir: str, expected_count: int) -> List[np.ndarray]:
        """Read generated frames from Wan2.2 output directory."""
        output_frames = []

        # Wan2.2 outputs video files — find them
        video_files = sorted(Path(output_dir).rglob("*.mp4"))

        if video_files:
            # Read frames from output video
            for video_file in video_files:
                cap = cv2.VideoCapture(str(video_file))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    output_frames.append(frame)
                cap.release()
        else:
            # Try reading individual frame images
            image_files = sorted(
                Path(output_dir).rglob("*.png"),
                key=lambda p: p.stem,
            )
            if not image_files:
                image_files = sorted(
                    Path(output_dir).rglob("*.jpg"),
                    key=lambda p: p.stem,
                )

            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    output_frames.append(frame)

        if not output_frames:
            raise RuntimeError(f"No output frames found in {output_dir}")

        # Pad or trim to expected count
        if len(output_frames) > expected_count:
            output_frames = output_frames[:expected_count]
        elif len(output_frames) < expected_count:
            pad_count = expected_count - len(output_frames)
            output_frames.extend([output_frames[-1]] * pad_count)

        logger.info(
            f"[BodyGenerator] Read {len(output_frames)} output frames "
            f"(expected {expected_count})"
        )
        return output_frames

    def unload_model(self):
        """Free GPU VRAM (Wan2.2 runs as subprocess, so cleanup is automatic)."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_frame(
        self,
        reference_image: np.ndarray,
        pose_image: np.ndarray,
    ) -> np.ndarray:
        """Single frame generation (wraps generate_sequence for compatibility)."""
        result = self.generate_sequence(reference_image, [pose_image])
        return result[0] if result else pose_image
