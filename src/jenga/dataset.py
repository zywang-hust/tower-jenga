"""Training batches: TOWER HDF5 -> OpenPI pi05 model inputs (docs/recipe.md).

Per sample: four uint8 cameras (letterboxed to 224), model-space state and action window (joint delta,
gripper closure, tail hold labels), quantile normalization from the train split, discrete-state prompt
tokens, zero-pad to 32 dims. Optional lighting augmentation (docs/recipe.md §6), shared across the four views.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jenga.hdf5_dataset import MODEL_CAMERAS, TowerFrames
from jenga.instructions import OFFICIAL_INSTRUCTIONS
from jenga.recipe import TASK_SCORE_WEIGHTS, emphasize, sample_weights
from jenga.transforms import GRIPPER_FULL_WIDTH_M, JOINTS, to_model_actions, to_model_state

ACTION_DIM = 32
# Prompt+state token budget. Worst case over all 10 TOWER instructions and every state bin is 101 tokens
# (OpenPI default 200); padding is masked, so 128 computes the same model 4-7% faster (docs/throughput.md).
MAX_TOKEN_LEN = 128
CONTACT_RADIUS = 45  # frames: samples within 1.5 s at 30 Hz of a grasp or release count as contact phase


def kelvin_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody RGB (Tanner Helland fit), normalized so 6500 K is neutral."""

    def raw(k):
        t = k / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
        return np.clip([r, g, b], 1.0, 255.0)

    return raw(kelvin) / raw(6500.0)


def lighting(images: dict[str, np.ndarray], rng: np.random.Generator, p: float = 0.5) -> dict[str, np.ndarray]:
    """docs/recipe.md §6: gain log-uniform in [0.3, 2.0], white balance 3000–8500 K, one draw for all views."""
    if rng.random() >= p:
        return images
    gain = np.exp(rng.uniform(np.log(0.3), np.log(2.0)))
    scale = (gain * kelvin_rgb(rng.uniform(3000.0, 8500.0))).astype(np.float32)
    return {k: np.clip(v.astype(np.float32) * scale, 0, 255).astype(np.uint8) for k, v in images.items()}


class TowerDataset:
    def __init__(self, root: Path, files: list[str], norm_stats: dict, *, horizon: int = 32,
                 max_token_len: int = MAX_TOKEN_LEN, augment_lighting: bool = False, prompt: str = "train",
                 state_noise_bins: int = 0, state_noise_p: float = 0.5, synthetic_files: Sequence[Path] = (),
                 contact_radius: int = 0):
        """files: dataset-relative demonstration paths. synthetic_files: absolute paths of extra episodes in the
        same layout (e.g. DART copies); sampler_weights mixes them in as a separate set. Ones without sim event
        fields take contact phases from their `source` demo under `root`.
        contact_radius > 0 marks samples near grasp/release for sampler_weights(contact_weight=...).
        prompt: 'train' (episode instruction) or 'official' (competition text, identical for X-2/X-3).
        state_noise_bins = K > 0: DART-style proprioceptive perturbation. With probability state_noise_p each joint
        of the state moves by k * bin width, k uniform in [-K, K], and the joint-delta labels are taken against the
        perturbed state, so the targets stay the demonstrated absolute commands and the model learns to correct
        small offsets. The state token then moves by exactly k and the label by exactly k bins, so the correction
        is a deterministic function of what the model sees. Joints where the token would saturate at q01/q99
        instead of moving by k are left unperturbed. Grippers untouched."""
        if prompt not in ("train", "official"):
            raise ValueError("prompt must be 'train' or 'official'")
        # Random sampling spans ~475 files; 64 handles per worker keeps reopen cost (~11 ms on GPFS) rare.
        paths = [Path(root) / f for f in files] + [Path(f) for f in synthetic_files]
        self.frames = TowerFrames(paths, horizon=horizon, tail_hold=True, max_open_files=64,
                                  contact_radius=contact_radius, source_root=Path(root))
        self.variants = [p.parent.name.split("_")[0] for p in paths]  # e.g. "task4-2"
        self.synthetic = np.arange(len(paths)) >= len(files)
        self.norm_stats = norm_stats
        self.max_token_len = max_token_len
        self.augment_lighting = augment_lighting
        self.prompt = prompt
        self.state_noise_p = state_noise_p
        self.state_noise_bins = state_noise_bins
        s = norm_stats["state"]  # [q01, q99] maps to [-1, 1], cut into 256 token bins
        self.state_bin_rad = ((np.asarray(s.q99) - np.asarray(s.q01) + 1e-6) / 256.0)[JOINTS].astype(np.float32)
        self._tokenizer = None

    def __len__(self) -> int:
        return len(self.frames)

    def sample_variants(self) -> list[str]:
        return [self.variants[i] for i in self.frames.index[:, 0]]

    def sampler_weights(self, synthetic_frac: float = 0.0, contact_weight: float = 1.0) -> np.ndarray:
        """docs/recipe.md §5 mix: variant probability proportional to sqrt(frame share x score share) of the
        demonstrations. Within a variant that has synthetic samples, synthetic_frac of its share goes to them (so
        variants not yet generated keep their full share on demonstrations). contact_weight: samples near
        grasp/release are drawn that many times as often as the rest of their variant and set, whose share is
        unchanged."""
        variants = np.asarray(self.sample_variants())
        synthetic = self.synthetic[self.frames.index[:, 0]]
        if synthetic_frac > 0 and not synthetic.any() or synthetic_frac == 0 and synthetic.any():
            raise ValueError("synthetic_frac > 0 exactly when synthetic files are given")
        score = {v: TASK_SCORE_WEIGHTS[int(v[4])] for v in set(variants.tolist())}
        weights = np.zeros(len(variants))
        weights[~synthetic] = sample_weights(variants[~synthetic].tolist(), score)
        for v in np.unique(variants[synthetic]):
            real, syn = (variants == v) & ~synthetic, (variants == v) & synthetic
            share = weights[real].sum()
            weights[real] *= 1.0 - synthetic_frac
            # a sparse-render sample stands for `span` frames, so every episode keeps its full-length share
            weights[syn] = synthetic_frac * share * self.frames.span[syn] / self.frames.span[syn].sum()
        if contact_weight != 1.0:
            if self.frames.contact is None:
                raise ValueError("contact_weight needs contact_radius > 0")
            groups = np.unique(variants, return_inverse=True)[1] * 2 + synthetic
            weights = emphasize(weights, groups, self.frames.contact, contact_weight)
        return weights

    def _state_tokens(self, measured: np.ndarray) -> np.ndarray:
        """Joint state tokens as OpenPI's PaligemmaTokenizer computes them (256 bins over [-1, 1])."""
        x = self._norm(to_model_state(measured, GRIPPER_FULL_WIDTH_M), "state")[JOINTS]
        return np.digitize(x, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    def _norm(self, x: np.ndarray, key: str) -> np.ndarray:
        s = self.norm_stats[key]
        return ((x - s.q01) / (s.q99 - s.q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        raw = self.frames[i]
        images = raw["images"]
        noisy = self.state_noise_bins > 0
        rng = np.random.default_rng() if self.augment_lighting or noisy else None
        if self.augment_lighting:
            images = lighting(images, rng)  # fresh entropy per call, independent per worker
        measured = raw["state"]
        if noisy and rng.random() < self.state_noise_p:
            k = rng.integers(-self.state_noise_bins, self.state_noise_bins + 1, len(JOINTS))
            shifted = measured.copy()
            shifted[JOINTS] += (k * self.state_bin_rad).astype(np.float32)
            # The tokenizer saturates outside [q01, q99] (tokens -1 and 255): a shift into or within saturation
            # would move the label without moving the token, so those joints stay unperturbed.
            k = np.where(self._state_tokens(shifted) - self._state_tokens(measured) == k, k, 0)
            measured = measured.copy()
            measured[JOINTS] += (k * self.state_bin_rad).astype(np.float32)
        prompt = raw["prompt"] if self.prompt == "train" else OFFICIAL_INSTRUCTIONS[self.variants[raw["file_id"]][:5]]
        return self.encode(images, measured, raw["actions"], prompt)

    def encode(self, images: dict, measured: np.ndarray, actions: np.ndarray, prompt: str) -> dict:
        """Model inputs from 224 px images, the measured 16-D state, absolute [H, 16] commands and a prompt."""
        if self._tokenizer is None:  # created lazily inside each DataLoader worker
            from openpi.models.tokenizer import PaligemmaTokenizer

            self._tokenizer = PaligemmaTokenizer(self.max_token_len)
        state = self._norm(to_model_state(measured, GRIPPER_FULL_WIDTH_M), "state")
        actions = self._norm(to_model_actions(actions, measured, GRIPPER_FULL_WIDTH_M), "actions")
        tokens, mask = self._tokenizer.tokenize(prompt, state)
        if mask.all():  # the tokenizer truncates silently past max_len (it only logs a warning)
            raise ValueError(f"prompt+state fills all {self.max_token_len} tokens; raise MAX_TOKEN_LEN")
        return {
            "image": {k: images[k] for k in MODEL_CAMERAS},
            "image_mask": {k: np.True_ for k in MODEL_CAMERAS},
            "state": np.pad(state, (0, ACTION_DIM - state.shape[-1])),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "actions": np.pad(actions, ((0, 0), (0, ACTION_DIM - actions.shape[-1]))),
        }


def collate(items: list[dict]) -> dict:
    def stack(xs):
        if isinstance(xs[0], dict):
            return {k: stack([x[k] for x in xs]) for k in xs[0]}
        return np.stack([np.asarray(x) for x in xs])

    return stack(items)
