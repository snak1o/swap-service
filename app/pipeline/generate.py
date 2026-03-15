"""Body generation using MimicMotion — pose-guided video synthesis (Tencent, ICML 2025)."""

import gc
import logging
import math
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

MIMIC_BASE_MODEL = "/app/models/svd_xt_1_1"
MIMIC_CHECKPOINT = "/app/models/MimicMotion_1-1.pth"
ASPECT_RATIO = 9 / 16


class BodyGenerator:
    """
    Generates video of a reference person performing poses from a source video.

    Uses MimicMotion (Tencent, ICML 2025):
    - SVD-XT base with PoseNet + Temporal UNet
    - Confidence-aware pose guidance
    - Progressive latent fusion for arbitrary-length video
    - 576x1024 resolution, up to 72 frames per chunk
    """

    def __init__(self):
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tile_size = 72
        self.tile_overlap = 6
        self.num_inference_steps = 25
        self.guidance_scale = 2.0
        self.noise_aug_strength = 0.02
        self.seed = 42
        self.resolution = 576
        self.sample_stride = 1

    def load_model(self):
        """Load MimicMotion pipeline."""
        from mimicmotion.utils.geglu_patch import patch_geglu_inplace
        patch_geglu_inplace()

        from mimicmotion.utils.loader import MimicMotionModel
        from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)

        try:
            logger.info("[BodyGenerator] Loading MimicMotion pipeline...")

            model = MimicMotionModel(MIMIC_BASE_MODEL)
            if hasattr(torch.serialization, "safe_globals"):
                checkpoint = torch.load(MIMIC_CHECKPOINT, map_location="cpu", weights_only=True)
            else:
                checkpoint = torch.load(MIMIC_CHECKPOINT, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            del checkpoint
            gc.collect()

            self.pipeline = MimicMotionPipeline(
                vae=model.vae,
                image_encoder=model.image_encoder,
                unet=model.unet,
                scheduler=model.noise_scheduler,
                feature_extractor=model.feature_extractor,
                pose_net=model.pose_net,
            )
            logger.info("[BodyGenerator] MimicMotion loaded successfully")
        finally:
            torch.set_default_dtype(prev_dtype)

    def unload_model(self):
        """Free GPU VRAM."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_sequence(
        self,
        reference_image: np.ndarray,
        video_frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Generate video of reference person in poses extracted from video_frames.

        MimicMotion handles DWPose extraction, pose rescaling, and generation internally.
        Progressive latent fusion ensures smooth output for long sequences.

        Args:
            reference_image: BGR photo of the target person
            video_frames: list of BGR frames from the source video (a single scene)

        Returns:
            list of BGR generated frames (same count as video_frames)
        """
        if self.pipeline is None:
            self.load_model()

        if not video_frames:
            return []

        ref_path = None
        video_path = None

        try:
            ref_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(ref_path, reference_image)

            video_path = tempfile.mktemp(suffix=".mp4")
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            for frame in video_frames:
                writer.write(frame)
            writer.release()

            pose_pixels, image_pixels = self._preprocess(video_path, ref_path)

            output_frames = self._run_pipeline(image_pixels, pose_pixels)

            expected = len(video_frames)
            if len(output_frames) > expected:
                output_frames = output_frames[:expected]
            elif len(output_frames) < expected:
                pad = expected - len(output_frames)
                output_frames.extend([output_frames[-1]] * pad)

            logger.info(f"[BodyGenerator] Generated {len(output_frames)} frames")
            return output_frames

        finally:
            for p in [ref_path, video_path]:
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _preprocess(self, video_path: str, image_path: str):
        """
        Preprocess reference image and video for MimicMotion.
        Extracts DWPose skeletons with rescaling to match reference proportions.
        """
        from torchvision.datasets.folder import pil_loader
        from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
        from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

        image_pixels = pil_loader(image_path)
        image_pixels = pil_to_tensor(image_pixels)
        h, w = image_pixels.shape[-2:]

        if h > w:
            w_target = self.resolution
            h_target = int(self.resolution / ASPECT_RATIO // 64) * 64
        else:
            w_target = int(self.resolution / ASPECT_RATIO // 64) * 64
            h_target = self.resolution

        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize = h_target
            w_resize = math.ceil(h_target / h_w_ratio)
        else:
            h_resize = math.ceil(w_target * h_w_ratio)
            w_resize = w_target

        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()

        logger.info(f"[BodyGenerator] Extracting poses from video (stride={self.sample_stride})...")
        image_pose = get_image_pose(image_pixels)
        video_pose = get_video_pose(video_path, image_pixels, sample_stride=self.sample_stride)

        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        pose_pixels = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1

        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
        image_pixels = torch.from_numpy(image_pixels) / 127.5 - 1

        logger.info(f"[BodyGenerator] Preprocessed: {pose_pixels.shape[0]} pose frames, "
                     f"resolution {pose_pixels.shape[-2]}x{pose_pixels.shape[-1]}")
        return pose_pixels, image_pixels

    @torch.no_grad()
    def _run_pipeline(self, image_pixels, pose_pixels) -> List[np.ndarray]:
        """Run MimicMotion inference with progressive latent fusion."""
        from torchvision.transforms.functional import to_pil_image

        image_pil = [
            to_pil_image(img.to(torch.uint8))
            for img in (image_pixels + 1.0) * 127.5
        ]

        num_frames = pose_pixels.size(0)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        logger.info(f"[BodyGenerator] Running MimicMotion: {num_frames} frames, "
                     f"tile_size={self.tile_size}, steps={self.num_inference_steps}")

        frames = self.pipeline(
            image_pil,
            image_pose=pose_pixels,
            num_frames=num_frames,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=7,
            noise_aug_strength=self.noise_aug_strength,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            min_guidance_scale=self.guidance_scale,
            max_guidance_scale=self.guidance_scale,
            decode_chunk_size=8,
            output_type="pt",
            device=self.device,
        ).frames.cpu()

        video_tensor = (frames * 255.0).to(torch.uint8)
        result_tensor = video_tensor[0, 1:]

        output = []
        for i in range(result_tensor.shape[0]):
            frame_rgb = result_tensor[i].permute(1, 2, 0).numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            output.append(frame_bgr)

        return output
