"""Locations of the artifacts that live outside the repository. Each one has an environment override; the
defaults are gitignored directories of this checkout (data/, models/) or the user cache. Stdlib only, so entry
points can import it before OpenPI or JAX.

  JENGA_DATA_ROOT   TOWER dataset release (MANIFEST.jsonl, data/<variant>/*.hdf5)   default <repo>/data/tower
  JENGA_MODELS      model store; the OpenPI base checkpoint is <JENGA_MODELS>/pi05_base default <repo>/models
  JENGA_CACHE       built helpers (sim/nosmi.c -> libnosmi.so)                       default ~/.cache/jenga
  CUDA_HOME         CUDA toolkit (nsys, libdevice) of the profiling and renderer tools; default the toolkit of the
                    nvcc on PATH, else /usr/local/cuda
  OPENPI_DATA_HOME  read by OpenPI itself (PaliGemma tokenizer cache)                default ~/.cache/openpi
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_ROOT = env_path("JENGA_DATA_ROOT", ROOT / "data/tower")
MODELS = env_path("JENGA_MODELS", ROOT / "models")
PI05_BASE = MODELS / "pi05_base"
CACHE = env_path("JENGA_CACHE", Path.home() / ".cache/jenga")
_nvcc = shutil.which("nvcc")
CUDA_HOME = env_path("CUDA_HOME", Path(_nvcc).resolve().parents[1] if _nvcc else Path("/usr/local/cuda"))
OPENPI_DATA_HOME = env_path("OPENPI_DATA_HOME", Path.home() / ".cache/openpi")
