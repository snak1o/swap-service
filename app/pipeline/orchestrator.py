"""Оркестратор пайплайна — координация всех этапов обработки."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from app.pipeline.pose import PoseExtractor
from app.pipeline.segment import BackgroundSegmenter
from app.pipeline.generate import BodyGenerator
from app.pipeline.face import FaceRefiner
from app.pipeline.composite import Compositor
from app.pipeline.camera_motion import CameraProfiler
from app.pipeline.frame_interpolation import FrameInterpolator


class SwapOrchestrator:
    """
    Координирует полный пайплайн full-body swap:
    
    1. Scene detection (разбиение на сцены)
    2. Pose extraction (извлечение скелета)
    3. Background segmentation (отделение фона)
    4. Body generation (генерация тела по позе + фото)
    5. Face refinement (улучшение лица)
    6. Compositing (наложение на фон)
    7. Post-processing (temporal smoothing, сборка видео)
    """

    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.segmenter = BackgroundSegmenter()
        self.generator = BodyGenerator()
        self.face_refiner = FaceRefiner()
        self.compositor = Compositor()
        self.camera_profiler = CameraProfiler()
        self.frame_interpolator = FrameInterpolator()

    def process(
        self,
        photo_path: str,
        video_path: str,
        output_path: str,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Запустить полный пайplайн обработки.
        
        Args:
            photo_path: путь к фото целевого персонажа
            video_path: путь к исходному видео
            output_path: путь для результата
            options: опции обработки
            progress_callback: функция для отчёта о прогрессе
                              callback(status, progress, message)
        
        Returns:
            dict с результатами
        """
        opts = options or {}
        start_time = time.time()

        def report(status: str, progress: float, message: str):
            if progress_callback:
                progress_callback(status, progress, message)
            print(f"[Orchestrator] [{progress:.0%}] {status}: {message}")

        # -----------------------------------------------------------
        # Step 0: Load inputs
        # -----------------------------------------------------------
        report("processing", 0.0, "Loading inputs...")

        reference = cv2.imread(photo_path)
        if reference is None:
            raise ValueError(f"Cannot load photo: {photo_path}")

        video_info = self.compositor.get_video_info(video_path)
        fps = opts.get("target_fps") or video_info["fps"]
        total_frames = video_info["frame_count"]

        report("processing", 0.02, f"Video: {video_info['width']}x{video_info['height']}, "
               f"{total_frames} frames, {video_info['fps']:.1f} fps")

        # -----------------------------------------------------------
        # Step 1: Scene detection
        # -----------------------------------------------------------
        report("processing", 0.05, "Detecting scenes...")
        segments = self._detect_scenes(video_path, opts.get("scene_detection", True))

        report("processing", 0.08, f"Found {len(segments)} scene segment(s)")

        # -----------------------------------------------------------
        # Step 2-6: Process each segment
        # -----------------------------------------------------------
        all_result_frames = []
        total_segments = len(segments)

        for seg_idx, (start_frame, end_frame) in enumerate(segments):
            seg_frames = self._read_frames(video_path, start_frame, end_frame)
            seg_count = len(seg_frames)

            base_progress = 0.1 + (seg_idx / total_segments) * 0.8
            seg_weight = 0.8 / total_segments

            report("pose_extraction", base_progress,
                   f"Segment {seg_idx + 1}/{total_segments}: Extracting poses ({seg_count} frames)")

            # 1.5. Evaluate camera motion for this segment
            report("pose_extraction", base_progress + seg_weight * 0.05,
                   f"Segment {seg_idx + 1}: Estimating camera motion")
                   
            rel_transforms = self.camera_profiler.estimate_motion(seg_frames)
            abs_transforms = self.camera_profiler.get_absolute_transforms(rel_transforms)

            # 2. Pose extraction
            skip = opts.get("skip_frames", 0)
            poses = []
            for i, frame in enumerate(seg_frames):
                if skip > 0 and i % (skip + 1) != 0:
                    poses.append(None)  # Will be interpolated
                    continue
                pose = self.pose_extractor.extract_from_frame(frame)
                poses.append(pose)

            # Interpolate skipped poses
            poses = self._interpolate_poses(poses)
            
            # Apply camera stabilization to poses
            keypoints_list = [p["keypoints"] if p else [] for p in poses]
            stabilized_kps = self.camera_profiler.stabilize_poses(keypoints_list, abs_transforms)
            
            # Update poses with stabilized keypoints
            for p, skp in zip(poses, stabilized_kps):
                if p:
                    p["keypoints"] = skp
                    # Also redraw pose_image with stabilized keypoints for generator
                    h, w = p["pose_image"].shape[:2]
                    p["pose_image"] = self.pose_extractor._draw_skeleton(skp, h, w)

            report("segmentation", base_progress + seg_weight * 0.2,
                   f"Segment {seg_idx + 1}: Segmenting background")

            # 3. Background segmentation
            bboxes = [p["bbox"] if p else None for p in poses]
            masks, backgrounds = self.segmenter.segment_video_frames(seg_frames, bboxes)

            report("body_generation", base_progress + seg_weight * 0.35,
                   f"Segment {seg_idx + 1}: Generating body frames")

            # 4. Body generation (Only generated for non-skipped frames)
            pose_images = [p["pose_image"] for p in poses if p is not None]
            
            # Get only frames that we actually need to generate
            if skip > 0:
                pose_images = []
                for i, p in enumerate(poses):
                    # In our simple _interpolate_poses, actual generated poses are the ones at interval
                    if i % (skip + 1) == 0:
                        pose_images.append(p["pose_image"])
            
            generated = self.generator.generate_sequence(reference, pose_images)
            
            # Frame Interpolation (RIFE) - fill the gaps
            if skip > 0:
                report("body_generation", base_progress + seg_weight * 0.55,
                       f"Segment {seg_idx + 1}: Interpolating missed frames (RIFE)")
                generated = self.frame_interpolator.interpolate_sequence(generated, skip)
                
                # Make sure lengths match (trim if interpolator added too many)
                if len(generated) > len(seg_frames):
                    generated = generated[:len(seg_frames)]
                elif len(generated) < len(seg_frames):
                    # Pad if too few (should not happen with good interpolator)
                    generated.extend([generated[-1]] * (len(seg_frames) - len(generated)))

            report("face_refinement", base_progress + seg_weight * 0.65,
                   f"Segment {seg_idx + 1}: Refining faces")

            # 5. Face refinement
            refined = self.face_refiner.refine_sequence(generated, reference)
            
            # 5.5 Re-apply camera motion to generated characters before compositing
            refined = self.camera_profiler.apply_motion_to_generated(refined, abs_transforms)

            report("compositing", base_progress + seg_weight * 0.8,
                   f"Segment {seg_idx + 1}: Compositing")

            # 6. Compositing
            composited = []
            for i in range(len(refined)):
                comp = self.compositor.composite_frame(
                    refined[i], backgrounds[i], masks[i]
                )
                composited.append(comp)

            all_result_frames.extend(composited)

        # -----------------------------------------------------------
        # Step 7: Post-processing
        # -----------------------------------------------------------
        report("post_processing", 0.92, "Temporal smoothing...")

        if opts.get("temporal_smoothing", True):
            all_result_frames = self.compositor.temporal_smooth(all_result_frames, window_size=3)

        report("post_processing", 0.95, "Assembling video...")

        # Assemble final video
        output = self.compositor.assemble_video(
            frames=all_result_frames,
            output_path=output_path,
            fps=fps,
            audio_source=video_path,
        )

        elapsed = time.time() - start_time

        report("completed", 1.0, f"Done in {elapsed:.1f}s — {len(all_result_frames)} frames")

        return {
            "output_path": output,
            "frames_processed": len(all_result_frames),
            "duration_sec": elapsed,
            "segments": len(segments),
            "fps": fps,
        }

    def _detect_scenes(self, video_path: str, enabled: bool) -> list:
        """Разбить видео на сцены через PySceneDetect."""
        if not enabled:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0, total)]

        try:
            from scenedetect import detect, ContentDetector

            scene_list = detect(video_path, ContentDetector(threshold=27.0))

            if not scene_list:
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return [(0, total)]

            segments = []
            for scene in scene_list:
                start = scene[0].get_frames()
                end = scene[1].get_frames()
                segments.append((start, end))

            return segments
        except ImportError:
            print("[Orchestrator] PySceneDetect not available, processing as single scene")
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0, total)]

    def _read_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> list:
        """Прочитать кадры из видео."""
        cap = cv2.VideoCapture(video_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        current = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if end_frame and current >= end_frame:
                break
            frames.append(frame)
            current += 1

        cap.release()
        return frames

    def _interpolate_poses(self, poses: list) -> list:
        """Интерполировать пропущенные позы (при skip_frames > 0)."""
        result = list(poses)

        # Find first valid pose
        first_valid = None
        for p in result:
            if p is not None:
                first_valid = p
                break

        if first_valid is None:
            # Fallback to a square resolution if nothing found
            return [self.pose_extractor._empty_result(512, 512)] * len(result)

        # Fill None gaps with nearest valid pose
        last_valid = first_valid
        for i in range(len(result)):
            if result[i] is None:
                result[i] = last_valid
            else:
                last_valid = result[i]

        return result
