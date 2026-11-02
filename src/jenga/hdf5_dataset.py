"""Random-access TOWER HDF5 frames: the I/O and decode hot path of training.

Returns raw physical arrays (uint8 images, 16-D state, 32x16 action window); action representation and
normalization live in jenga.transforms. `tail_hold` implements the end-of-episode hold labels
(docs/recipe.md §3). `contact_radius` marks samples near grasp/release events (docs/recipe.md §5).
Works with torch.utils.data.DataLoader; HDF5 handles are opened lazily per worker.
"""
from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from openpi_client.image_tools import resize_with_pad
from PIL import Image

CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")
MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "second_base_rgb")
CAMERA_MAP = dict(zip(MODEL_CAMERAS, (CAMERAS[0], CAMERAS[2], CAMERAS[3], CAMERAS[1]), strict=True))
GRIPPER_COLUMNS = (7, 15)  # jenga.transforms.GRIPPERS


def contact_events(active_block: np.ndarray, released: np.ndarray, gripper_cmd: np.ndarray,
                   tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (grasp, release) event flags of one demo. Grasp: a block becomes the active one, or a commanded
    gripper width closes. Release: the active block is cleared, a block is flagged released, or a width opens.
    Inputs: sim/active_block [T] (-1 = none), sim/block_released [T, B], leader/qpos gripper columns [T, 2]."""
    grasp, release = np.zeros(len(active_block), bool), np.zeros(len(active_block), bool)
    changed = active_block[1:] != active_block[:-1]
    width = np.diff(gripper_cmd, axis=0)
    grasp[1:] = (changed & (active_block[1:] >= 0)) | (width < -tol).any(1)
    release[1:] = ((changed & (active_block[:-1] >= 0)) | (released[1:] & ~released[:-1]).any(1)
                   | (width > tol).any(1))
    return grasp, release


def near(events: np.ndarray, radius: int) -> np.ndarray:
    """Frames within `radius` frames of any event."""
    count = np.concatenate([[0], np.cumsum(events, dtype=np.int64)])
    t = np.arange(len(events))
    return count[np.minimum(t + radius + 1, len(events))] > count[np.maximum(t - radius, 0)]


def contact_frames(f: h5py.File, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """(near grasp, near release) masks over the frames of an open episode file."""
    grasp, release = contact_events(f["sim/active_block"][:], f["sim/block_released"][:],
                                    f["leader/qpos"][:, list(GRIPPER_COLUMNS)])
    return near(grasp, radius), near(release, radius)


def window_starts(valid: np.ndarray, horizon: int) -> np.ndarray:
    """Frame indices whose next `horizon` actions are all valid (no padding, no cross-episode windows)."""
    if len(valid) < horizon:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.lib.stride_tricks.sliding_window_view(valid, horizon).all(-1))


def hold_window(actions, last_valid: int, start: int, horizon: int) -> np.ndarray:
    """actions[start:start+horizon], rows past `last_valid` repeat the last valid command ("hold still").

    Reads one contiguous slice (h5py datasets reject repeated fancy indices) and pads in memory.
    """
    window = np.asarray(actions[start : min(start + horizon, last_valid + 1)])
    return np.concatenate([window, np.repeat(window[-1:], horizon - len(window), axis=0)])


class TowerFrames:
    def __init__(
        self,
        files: list[Path],
        *,
        horizon: int = 32,
        image_size: int = 224,
        max_open_files: int = 16,
        tail_hold: bool = False,
        contact_radius: int = 0,
        source_root: Path | None = None,
    ):
        """contact_radius > 0: `self.contact[i]` is True when sample i starts within that many frames of a
        grasp or release event (contact_frames; needs sim/active_block and sim/block_released). A derived episode
        without them (e.g. a DART copy) takes the events of the demo named by its `source` attribute, a path
        relative to `source_root`, which must have the same frame count."""
        self.files = [Path(f) for f in files]
        self.horizon = horizon
        self.tail_hold = tail_hold
        self.image_size = image_size
        self.max_open_files = max_open_files
        index = []
        contact = []
        span = []  # frames of its episode each sample stands for (the render stride; 1 for dense files)
        self.last_valid = []
        self.instructions = []
        for file_id, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                valid = f["sim/action_valid"][:]
                if contact_radius > 0 and "sim/active_block" in f:
                    contact.append(np.logical_or(*contact_frames(f, contact_radius)))
                elif contact_radius > 0:
                    with h5py.File(Path(source_root) / f.attrs["source"], "r") as src:
                        contact.append(np.logical_or(*contact_frames(src, contact_radius)))
                    if len(contact[-1]) != len(valid):
                        raise ValueError(f"{path}: frame count differs from its source episode")
                instruction = f.attrs.get("instruction", "")
                image_valid = f["sim/image_valid"][:] if "sim/image_valid" in f else None
            self.instructions.append(instruction.decode() if isinstance(instruction, bytes) else str(instruction))
            valid_rows = np.flatnonzero(valid)
            # Valid rows must be one contiguous prefix for hold labels to mean "after the last command".
            if tail_hold and len(valid_rows) and not valid[: valid_rows[-1] + 1].all():
                raise ValueError(f"{path}: action_valid has interior gaps; hold labels undefined")
            self.last_valid.append(int(valid_rows[-1]) if len(valid_rows) else -1)
            starts = valid_rows if tail_hold else window_starts(valid, horizon)
            n_all = len(starts)
            if image_valid is not None:  # sparse renders (sim/expert_write.py --frame-stride): frames with images
                starts = starts[image_valid[starts]]
            index.append(np.stack([np.full_like(starts, file_id), starts], axis=1))
            span.append(np.full(len(starts), n_all / max(len(starts), 1)))
            if contact_radius > 0:
                contact[-1] = contact[-1][starts]
        self.index = np.concatenate(index) if index else np.zeros((0, 2), dtype=np.int64)
        self.contact = np.concatenate(contact) if contact else None
        self.span = np.concatenate(span) if span else np.zeros(0)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self):
        # Handles never cross process boundaries; each DataLoader worker opens its own.
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def _file(self, file_id: int) -> h5py.File:
        handle = self._handles.pop(file_id, None)
        if handle is None:
            handle = h5py.File(self.files[file_id], "r")
            while len(self._handles) >= self.max_open_files:
                self._handles.popitem(last=False)[1].close()
        self._handles[file_id] = handle
        return handle

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        image = Image.open(io.BytesIO(raw.tobytes()))
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        # Same PIL letterbox as OpenPI serving (openpi_client.image_tools), so train and serve inputs match.
        return resize_with_pad(image, self.image_size, self.image_size)

    def __getitem__(self, i: int) -> dict:
        file_id, start = (int(x) for x in self.index[i])
        f = self._file(file_id)
        return {
            "images": {key: self._decode(f[f"observations/images/{cam}"][start]) for key, cam in CAMERA_MAP.items()},
            "state": f["observations/qpos"][start].astype(np.float32),
            "actions": (
                hold_window(f["leader/qpos"], self.last_valid[file_id], start, self.horizon)
                if self.tail_hold
                else f["leader/qpos"][start : start + self.horizon]
            ).astype(np.float32),
            "prompt": self.instructions[file_id],
            "file_id": file_id,
            "frame": start,
        }
