"""One switch for the measured, capability-preserving OpenPI fast path (see docs/throughput.md).

Call `enable_fast_path()` before JAX initializes its GPU backend (i.e. before the first jax.devices()
or jit) in both training and serving processes. Every setting computes the same model; differences
versus upstream are at the level of upstream's own run-to-run non-determinism.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

# Communication/compute overlap. Measured with CUDA_DEVICE_MAX_CONNECTIONS=1.
XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_pipelined_all_gather=true",
    "--xla_gpu_enable_pipelined_reduce_scatter=true",
    "--xla_gpu_enable_pipelined_all_reduce=true",
    "--xla_gpu_enable_while_loop_double_buffering=true",
)


def _preload_nvrtc() -> None:
    # cuDNN's runtime-compiled attention kernels dlopen("libnvrtc.so.12") by soname. The venv ships it
    # outside the loader path; loading it globally first lets that later dlopen resolve.
    import nvidia.cuda_nvrtc

    lib = Path(nvidia.cuda_nvrtc.__path__[0]) / "lib" / "libnvrtc.so.12"
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def enable_fast_path(*, training: bool = True, per_device_batch: int = 8) -> dict:
    """Apply all adopted switches; returns what was set for logging into run metadata."""
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = " ".join([existing, *[f for f in XLA_FLAGS if f.split("=")[0] not in existing]]).strip()
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _preload_nvrtc()

    import openpi.models.gemma as gemma
    import openpi.models.pi0 as pi0
    import openpi.models.siglip as siglip

    pi0.BATCH_IMAGE_VIEWS = True
    gemma.ATTENTION_NATURAL_LAYOUT = True
    siglip.ATTENTION_IMPL = "cudnn"
    if training:
        # Saved matmul outputs need ~55 GiB/GPU at global batch 32 (token budget 128; 57 at 200) on 4xH800 (fsdp=4); unrolled layer scans.
        # At 16 samples/GPU (b64 on 4 GPUs) Gemma dnb crashes XLA at compile time, so Gemma keeps the
        # upstream policy there (88.6 samples/s, 60.5 GiB at token budget 128, runs/bench/train/tok128_b64.json, with XLA_PYTHON_CLIENT_MEM_FRACTION=0.94).
        gemma.REMAT_POLICY = "dots_with_no_batch_dims_saveable" if per_device_batch <= 8 else "nothing_saveable"
        siglip.REMAT_POLICY = "dots_with_no_batch_dims_saveable"
        gemma.SCAN_UNROLL = 18
        siglip.SCAN_UNROLL = 27
    return {
        "XLA_FLAGS": os.environ["XLA_FLAGS"],
        "CUDA_DEVICE_MAX_CONNECTIONS": os.environ["CUDA_DEVICE_MAX_CONNECTIONS"],
        "BATCH_IMAGE_VIEWS": pi0.BATCH_IMAGE_VIEWS,
        "ATTENTION_NATURAL_LAYOUT": gemma.ATTENTION_NATURAL_LAYOUT,
        "SIGLIP_ATTENTION_IMPL": siglip.ATTENTION_IMPL,
        "GEMMA_REMAT_POLICY": gemma.REMAT_POLICY,
        "SIGLIP_REMAT_POLICY": siglip.REMAT_POLICY,
        "GEMMA_SCAN_UNROLL": gemma.SCAN_UNROLL,
        "SIGLIP_SCAN_UNROLL": siglip.SCAN_UNROLL,
    }
