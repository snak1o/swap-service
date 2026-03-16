"""Pipeline orchestrator — coordinates all stages of full-body swap."""

import time
from typing import Optional, Callable

import cv2
import numpy as np

from app.pipeline.segment import BackgroundSegmenter
from app.pipeline.generate import BodyGenerator
from app.pipeline.face import FaceRefiner
from app.pipeline.composite import Compositor


class SwapOrchestrator:
    """
    Full-body swap pipeline (Wan2.2-Animate-14B Replace mode):

    1. Scene detection (split video by hard cuts)
    2. For each scene independently:
       a. Segment original frames -> clean backgrounds + person masks
       b. Wan2.2-Animate-14B Replace: generate video with replaced person
       c. Resize generated frames to original resolution
       d. Face refinement (InsightFace swap + GFPGAN)
       e. Composite generated person onto original background
    3. Temporal smoothing + assemble final video with original audio
    """

    def __init__(self):
        self.segmenter = BackgroundSegmenter()
        self.generator = BodyGenerator()
        self.face_refiner = FaceRefiner()
        self.compositor = Compositor()

    def process(
        self,
        photo_path: str,
        video_path: str,
        output_path: str,
        options: Optional[dict] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        opts = options or {}
        start_time = time.time()

        def report(status: str, progress: float, message: str):
            if progress_callback:
                progress_callback(status, progress, message)
            print(f"[Orchestrator] [{progress:.0%}] {status}: {message}")

        report("processing", 0.0, "Loading inputs...")

        reference = cv2.imread(photo_path)
        if reference is None:
            raise ValueError(f"Cannot load photo: {photo_path}")

        video_info = self.compositor.get_video_info(video_path)
        fps = opts.get("target_fps") or video_info["fps"]
        total_frames = video_info["frame_count"]

        report("processing", 0.02,
               f"Video: {video_info['width']}x{video_info['height']}, "
               f"{total_frames} frames, {video_info['fps']:.1f} fps")

        # --- Scene detection ---
        report("processing", 0.05, "Detecting scenes...")
        segments = self._detect_scenes(video_path, opts.get("scene_detection", True))
        report("processing", 0.08, f"Found {len(segments)} scene segment(s)")

        # --- Process each scene independently ---
        all_result_frames = []
        total_segments = len(segments)

        for seg_idx, (start_frame, end_frame) in enumerate(segments):
            seg_frames = self._read_frames(video_path, start_frame, end_frame)
            seg_count = len(seg_frames)
            orig_h, orig_w = seg_frames[0].shape[:2]

            base_progress = 0.1 + (seg_idx / total_segments) * 0.8
            seg_weight = 0.8 / total_segments

            # --- Step 1: Segment original frames ---
            report("segmentation", base_progress,
                   f"Segment {seg_idx + 1}/{total_segments}: Segmenting background ({seg_count} frames)")

            masks, backgrounds = self.segmenter.segment_video_frames(seg_frames)
            self.segmenter.unload_model()

            # --- Step 2: Generate with Wan2.2-Animate-14B Replace ---
            report("body_generation", base_progress + seg_weight * 0.2,
                   f"Segment {seg_idx + 1}/{total_segments}: Wan2.2 Replace generating ({seg_count} frames)")

            generated = self.generator.generate_sequence(reference, seg_frames)
            self.generator.unload_model()

            # --- Step 3: Resize to original resolution ---
            gen_h, gen_w = generated[0].shape[:2] if generated else (orig_h, orig_w)
            if gen_h != orig_h or gen_w != orig_w:
                report("body_generation", base_progress + seg_weight * 0.55,
                       f"Segment {seg_idx + 1}: Resizing {gen_w}x{gen_h} -> {orig_w}x{orig_h}")
                generated = [
                    cv2.resize(f, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                    for f in generated
                ]

            # --- Step 4: Face refinement ---
            report("face_refinement", base_progress + seg_weight * 0.6,
                   f"Segment {seg_idx + 1}: Refining faces")

            refined = self.face_refiner.refine_sequence(generated, reference)

            # --- Step 5: Composite onto original backgrounds ---
            report("compositing", base_progress + seg_weight * 0.8,
                   f"Segment {seg_idx + 1}: Compositing onto original background")

            composited = []
            for i in range(min(len(refined), len(backgrounds))):
                comp = self.compositor.composite_frame(
                    refined[i], backgrounds[i], masks[i]
                )
                composited.append(comp)

            all_result_frames.extend(composited)

        # --- Post-processing ---
        report("post_processing", 0.92, "Temporal smoothing...")

        if opts.get("temporal_smoothing", True):
            all_result_frames = self.compositor.temporal_smooth(
                all_result_frames, window_size=3
            )

        report("post_processing", 0.95, "Assembling video...")

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
        """Split video into scenes using PySceneDetect."""
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
        """Read frames from video."""
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
