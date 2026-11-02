#!/usr/bin/env python3
"""Verify imports and optional GPU math without a dataset or a model download."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Run a small matmul on every visible JAX GPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    versions = {}
    for module in ("numpy", "h5py", "jax", "jaxlib", "flax", "orbax.checkpoint", "torch", "lerobot", "openpi"):
        importlib.import_module(module)
        dist = module.replace(".", "-")
        versions[dist] = importlib.metadata.version(dist)
    import jax
    import jax.numpy as jnp
    import numpy as np

    devices = jax.devices()
    results = []
    if args.gpu:
        assert all(d.platform == "gpu" for d in devices), f"GPU backend not available: {devices}"
        for device in devices:
            with jax.default_device(device):
                x = jnp.ones((128, 128), dtype=jnp.float32)
                y = jax.jit(lambda a: a @ a)(x).block_until_ready()
                np.testing.assert_allclose(np.asarray(y), 128.0)
                results.append({"device": str(device), "matmul": "passed"})
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": platform.python_version(),
        "versions": versions,
        "devices": [str(d) for d in devices],
        "gpu_checks": results,
        "openpi_revision": subprocess.check_output(
            ["git", "-C", str(root / "external/openpi"), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
