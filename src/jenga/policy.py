"""Online policy adapter: tower-infer-v1 observation -> OpenPI model input -> absolute wire actions.

Mirrors the training chain in jenga.dataset.TowerDataset field for field (camera mapping, letterbox resize,
16-D state layout, gripper closure, quantile normalization, discrete-state tokens, 32-D padding).
tests/test_policy.py checks bitwise equality against TowerDataset on dataset frames.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openpi_client.image_tools import resize_with_pad
from PIL import Image

from jenga.hdf5_dataset import CAMERA_MAP, MODEL_CAMERAS
from jenga.instructions import OFFICIAL_INSTRUCTIONS, TRAIN_INSTRUCTIONS
from jenga.dataset import ACTION_DIM, MAX_TOKEN_LEN
from jenga.transforms import GRIPPER_FULL_WIDTH_M, from_model_actions, to_model_state

# Training images are the dataset's JPEGs (quality 95: the quantization tables equal the IJG tables scaled for 95).
# The simulator delivers raw RGB, so an optional round trip reproduces the training input distribution.
JPEG_QUALITY = 95

# The dataset JPEGs of the two external cameras are rotated by 180 degrees relative to the raw render; the official
# worker sends the raw render, except in trials where the views already arrive turned (jenga.orient,
# docs/method.md 7.2). The wrist cameras already match.
WIRE_ROTATED_CAMERAS = ("left_back_camera", "right_front_camera")


def to_dataset_orientation(observation: dict) -> dict:
    """Wire observation -> same observation with every image in the dataset orientation. Shallow copy."""
    images = dict(observation["images"])
    for camera in WIRE_ROTATED_CAMERAS:
        if camera in images:
            images[camera] = np.ascontiguousarray(np.asarray(images[camera])[::-1, ::-1])
    return {**observation, "images": images}


def variant_key(task_variant: str) -> str:
    """'task2-3_cross_tower_transfer_3' -> 'task2-3'."""
    key = task_variant.split("_")[0]
    if key not in TRAIN_INSTRUCTIONS:
        raise ValueError("invalid_task_variant")
    return key


def state16(observation: dict) -> np.ndarray:
    """Wire state -> dataset qpos layout: left 7 joints, left width, back 7 joints, back width."""
    q = np.asarray(observation["joint_positions"], dtype=np.float32)
    w = np.asarray(observation["gripper_widths_m"], dtype=np.float32)
    if q.shape != (2, 7) or w.shape != (2,) or not (np.isfinite(q).all() and np.isfinite(w).all()):
        raise ValueError("invalid_observation_state")
    return np.concatenate([q[0], w[:1], q[1], w[1:]])


def jpeg_round_trip(image: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


_IMAGE_POOL = ThreadPoolExecutor(len(MODEL_CAMERAS), thread_name_prefix="adapter-img")


class ObservationAdapter:
    """Per-request transform; stateless apart from the lazily built tokenizer, so replicas can share one."""

    def __init__(self, norm_stats: dict, *, prompt: str = "train", jpeg_quality: int | None = JPEG_QUALITY,
                 image_size: int = 224, max_token_len: int = MAX_TOKEN_LEN):
        if prompt not in ("train", "official"):
            raise ValueError("prompt must be 'train' (the variant's training instruction) or 'official'")
        self.norm_stats = norm_stats
        self.prompt = prompt
        self.jpeg_quality = jpeg_quality
        self.image_size = image_size
        self.max_token_len = max_token_len
        self._tokenizer = None

    def _norm(self, x: np.ndarray, key: str) -> np.ndarray:  # identical to TowerDataset._norm
        s = self.norm_stats[key]
        return ((x - s.q01) / (s.q99 - s.q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)

    def prompt_text(self, observation: dict) -> str:
        """The exact training string, chosen by task_variant. With 'official' the wire instruction is only
        compared against our copy of the published text: any drift (whitespace, punctuation, a doc revision)
        would otherwise reach the model as a prompt it never saw."""
        key = variant_key(observation["task_variant"])
        if self.prompt == "train":
            return TRAIN_INSTRUCTIONS[key]
        text = OFFICIAL_INSTRUCTIONS[key[:5]]
        if observation.get("instruction") != text:
            logging.getLogger(__name__).warning("wire instruction differs from the published text for %s: %r",
                                                key, observation.get("instruction"))
        return text

    def image(self, observation: dict, camera: str) -> np.ndarray:
        image = np.asarray(observation["images"][camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"invalid_image_{camera}")
        if self.jpeg_quality is not None:
            image = jpeg_round_trip(image, self.jpeg_quality)
        return resize_with_pad(image, self.image_size, self.image_size)

    def __call__(self, observation: dict) -> dict:
        if self._tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer

            self._tokenizer = PaligemmaTokenizer(self.max_token_len)
        state = self._norm(to_model_state(state16(observation), GRIPPER_FULL_WIDTH_M), "state")
        tokens, mask = self._tokenizer.tokenize(self.prompt_text(observation), state)
        if mask.all():
            raise ValueError("prompt_too_long")
        # PIL releases the GIL in JPEG encode/decode and resize, so the four cameras overlap (24 -> 12 ms, identical output)
        images = dict(zip(MODEL_CAMERAS, _IMAGE_POOL.map(lambda key: self.image(observation, CAMERA_MAP[key]), MODEL_CAMERAS)))
        return {
            "image": images,
            "image_mask": {key: np.True_ for key in MODEL_CAMERAS},
            "state": np.pad(state, (0, ACTION_DIM - state.shape[-1])),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
        }

    def decode(self, model_actions: np.ndarray, observation_state16: np.ndarray) -> np.ndarray:
        """Model output [H, 32] (normalized model space) -> absolute [H, 16] joint radians and widths in metres.
        Joint deltas are relative to `observation_state16`, the state the model was given."""
        s = self.norm_stats["actions"]
        model_space = (np.asarray(model_actions, np.float32)[:, :16] + 1.0) / 2.0 * (s.q99 - s.q01 + 1e-6) + s.q01
        return from_model_actions(model_space, observation_state16, GRIPPER_FULL_WIDTH_M)


def clip_to_bounds(actions: np.ndarray, lower, upper) -> np.ndarray:
    """Release bounds from track.json; gripper widths are additionally kept within [0, full width]."""
    out = np.clip(actions, np.asarray(lower, np.float32), np.asarray(upper, np.float32))
    out[:, [7, 15]] = np.clip(out[:, [7, 15]], 0.0, GRIPPER_FULL_WIDTH_M)
    if not np.isfinite(out).all():
        raise ValueError("non_finite_actions")
    return out.astype("<f4")
