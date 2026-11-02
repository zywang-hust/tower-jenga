"""Deterministic HDF5 test fixtures in the TOWER layout. NOT robot demonstrations."""
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from jenga.hdf5_dataset import CAMERAS


def generate(directory: Path, episodes=4, frames=96, size=224, seed=42):
    if episodes < 2 or frames < 34 or size < 32:
        raise ValueError("Need >=2 episodes, >=34 frames, image size >=32")
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    records = []
    for episode in range(episodes):
        phase = np.arange(frames, dtype=np.float32) / 30 + episode * 0.3
        state = (0.3 * np.sin(phase[:, None] + np.arange(16)[None] * 0.2)).astype(np.float32)
        state[:, [7, 15]] = 0.03 + 0.01 * np.sin(phase[:, None] + [0, 0.4])
        actions = state.copy()
        joints = [i for i in range(16) if i not in (7, 15)]
        actions[:, joints] += (0.02 * np.cos(phase[:, None])).astype(np.float32)
        path = directory / f"synthetic_{episode:03}.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs.update(synthetic=True, provenance="NOT physical demonstrations; software fixtures only",
                           fps=30.0, recording_complete=True, image_encoding="jpeg",
                           arm_order=json.dumps(["left", "back"]))
            f.create_dataset("observations/qpos", data=state)
            f.create_dataset("leader/qpos", data=actions)
            for key in ("observations/lifter/position_mm", "lifter/action_mm"):
                f.create_dataset(key, data=np.zeros((frames, 2), dtype=np.float32))
            f.create_dataset("sim/time", data=np.arange(frames, dtype=np.float64) / 30)
            f.create_dataset("sim/action_valid", data=np.arange(frames) < frames - 1)
            f.create_dataset("sim/terminal", data=np.arange(frames) == frames - 1)
            for camera_index, camera in enumerate(CAMERAS):
                ds = f.create_dataset(f"observations/images/{camera}", (frames,),
                                      dtype=h5py.vlen_dtype(np.dtype("uint8")))
                texture = rng.integers(70, 110, (size, size, 3), dtype=np.uint8)
                for t in range(frames):
                    image = Image.fromarray(texture)
                    draw = ImageDraw.Draw(image)
                    shift = int(12 * np.sin(phase[t] + camera_index))
                    for level in range(3):
                        x = size // 3 + shift + level * 3
                        y = size * 3 // 4 - level * 25
                        draw.rectangle((x, y, x + 65, y + 18), fill=(170 + level * 20, 120, 50))
                    stream = io.BytesIO()
                    image.save(stream, format="JPEG", quality=90)
                    ds[t] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        records.append({"file": path.name, "split": "validation" if episode == episodes - 1 else "train"})
    manifest = {"synthetic": True, "purpose": "correctness/throughput only, not task-policy training",
                "seed": seed, "frames_per_episode": frames, "image_size": size, "episodes": records}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
