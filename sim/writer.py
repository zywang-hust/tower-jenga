"""Shared by the writers of rendered training episodes (sim/perturb.py, sim/expert_write.py, sim/expert_dagger.py
write): the four cameras rendered in batches and JPEG-encoded into the demonstration layout, files that appear by
rename, and a shard of work items pulled by renderer processes on one GPU. Renderer venv.
"""
from __future__ import annotations

import ctypes
import io
import json
import multiprocessing as mp
import os
import queue as queue_module
import signal
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from render import CAMERAS, STATION, episode_blocks, episode_dims  # noqa: F401  episode_dims for the writers

BATCH, JPEG_QUALITY = 8, 95  # envs per render call: GPU time per frame is flat from 2 to 16, 8 peaks at 10 GiB


@lru_cache
def block_names(fixed: tuple) -> list[str]:
    """Station block names of an episode's sim/block_poses columns, from its block_fixed_on_table flags."""
    return episode_blocks(STATION, fixed)


def frame_poses(renderer, links, blocks, names, moved: dict, dims: dict | None = None) -> np.ndarray:
    """Owner pose stack [O, 4, 4]; `moved` maps block index -> new 4x4; `dims` from episode_dims (task5 side
    blocks are 32 mm tall)."""
    return renderer.poses(links, {n: moved.get(k, b) for k, (n, b) in enumerate(zip(names, blocks))}, dims)


def encode(image: np.ndarray) -> np.ndarray:
    """JPEG of an RGB [H, W, 3] or RGBX [H, W, 4] (C-contiguous) uint8 image. RGBX is wrapped without a copy and
    packed to RGB inside the encoder, which releases the GIL; Image.fromarray of RGB unpacks with the GIL held.
    Both give the same bytes."""
    with _nvtx("jpeg"):
        buf = io.BytesIO()
        h, w, c = image.shape
        im = Image.frombuffer("RGBX", (w, h), image, "raw", "RGBX", 0, 1) if c == 4 else Image.fromarray(image)
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return np.frombuffer(buf.getvalue(), np.uint8)


@contextmanager
def _nvtx(name: str):
    """NVTX range for the nsys timeline (docs/throughput.md §8.2)."""
    import torch
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def atomic(dst: Path):
    """A temporary path next to dst that becomes dst only if the block completes."""
    tmp = dst.with_name(f"{dst.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(dst)


def write_images(g, renderer, n: int, poses, jpeg_dtype, stride: int = 1) -> float:
    """observations/images/<camera> [n] of the open HDF5 `g`: frames 0, stride, 2 stride, ... < n rendered BATCH at a
    time in dataset orientation (`poses(t)`: owner pose stack of frame t); the other entries stay empty (zero-length)
    and sim/image_valid [n] marks the rendered ones. Three-stage pipeline over rotating pinned host buffers:
    while the GPU renders batch k, the host waits for batch k-1's copy, submits its JPEG encodes to 8 threads and
    writes batch k-2's JPEGs, then prepares the poses of batch k+1. -> seconds the host spent blocked on the GPU."""
    import torch
    images = [g.create_dataset(f"observations/images/{c}", (n,), dtype=jpeg_dtype, chunks=(64,)) for c in CAMERAS]
    C = len(CAMERAS)
    ticks = np.arange(0, n, stride)
    g["sim/image_valid"] = np.isin(np.arange(n), ticks)
    bufs = [torch.empty((BATCH, C, renderer.h, renderer.w, 4), dtype=torch.uint8).pin_memory() for _ in range(3)]
    inflight, encoding, t_wait = None, None, 0.0  # (start, frames, buffer, event); (start, futures)

    def flush(job):
        s, futures = job
        for i, fu in enumerate(futures):  # frame-major, one by one: h5py reads a 1-element vlen slice as 2-D
            with _nvtx("jpeg_wait"):
                b = fu.result()
            with _nvtx("h5py"):
                images[i % C][ticks[s + i // C]] = b

    with ThreadPoolExecutor(8) as pool:
        m_all = len(ticks)
        for k, s in enumerate(range(0, m_all + BATCH, BATCH)):
            nxt = None
            if s < m_all:
                with _nvtx("poses"):
                    P = np.stack([poses(t) for t in ticks[s:s + BATCH]])
                with _nvtx("render_batch"):  # buffer k % 3 was last read by batch k-3's encodes, flushed at k-1
                    nxt = s, len(P), bufs[k % 3], renderer.render_rgbx_async(P, bufs[k % 3])
            if inflight:
                s0, m, buf, done = inflight
                r0 = time.perf_counter()
                with _nvtx("gpu_wait"):
                    done.synchronize()
                t_wait += time.perf_counter() - r0
                if encoding:
                    flush(encoding)
                rgbx = buf.numpy()
                encoding = s0, [pool.submit(encode, rgbx[e, c]) for e in range(m) for c in range(C)]
            inflight = nxt
        if encoding:
            flush(encoding)
    return t_wait


def _worker(queue, done, setup, setup_args: tuple, key: str):
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: do not outlive the shard process
    job = setup(*setup_args)
    while (item := queue.get()) is not None:
        try:
            done.put(job(item))
        except Exception as e:  # noqa: BLE001  report and go on; the item stays undone for the next run
            done.put({key: item, "frames": 0, "error": repr(e)})


def run_shard(items: list, workers: int, setup, setup_args: tuple = (), key: str = "name") -> int:
    """`items` through `workers` spawned processes, each of which calls setup(*setup_args) once (a module-level
    function: it loads the renderer) for a job, then job(item) -> result dict with "frames" per item it pulls. One
    JSON line per result; an item whose job raised reports {key: item, "error": ...}. -> frames written."""
    ctx = mp.get_context("spawn")
    queue, done = ctx.Queue(), ctx.Queue()
    for x in items:
        queue.put(x)
    procs = [ctx.Process(target=_worker, args=(queue, done, setup, setup_args, key)) for _ in range(workers)]
    for p in procs:
        queue.put(None)
        p.start()
    t0, frames = time.perf_counter(), 0
    for i in range(len(items)):
        while True:
            try:
                r = done.get(timeout=60)
                break
            except queue_module.Empty:
                assert any(p.is_alive() for p in procs), "all workers exited"
        frames += r["frames"]
        print(json.dumps({**r, "done": i + 1, "of": len(items), "shard_fps": round(frames / (time.perf_counter() - t0), 1)}),
              flush=True)
    for p in procs:
        p.join()
    return frames
