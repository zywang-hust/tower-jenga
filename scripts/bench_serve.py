"""Model-side latency of S concurrent closed-loop sessions on one GPU replica, with and without batching.

Question: with 8 episodes + 1 check session spread over 4 GPUs (2-3 sessions per GPU), what request
latency does a session see when requests queue on a shared replica, and does dynamic batching help?
Each session thread sends a request, waits for its chunk, and immediately sends the next (zero sim time:
worst case). Per-session noise is passed explicitly, so batching keeps each session's own sample seed.
Excludes tokenizer, transforms and network. Writes runs/bench/serve/<tag>.json.
"""
import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import queue
import threading
import time
from pathlib import Path

from jenga.paths import PI05_BASE

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PI05_BASE / "params"  # env JENGA_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--requests", type=int, default=60, help="per session, after warmup")
    parser.add_argument("--mode", choices=["serial", "batch"], required=True)
    parser.add_argument("--buckets", default="1,2,4", help="padded batch sizes compiled up front")
    parser.add_argument("--barrier-dir", type=Path, help="co-located replicas: start timing only once all are warm")
    parser.add_argument("--barrier-n", type=int, default=1)
    parser.add_argument("--window-ms", type=float, default=3.0, help="batch mode: wait this long for more requests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/bench/serve")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from jenga.perf import enable_fast_path

    fast = enable_fast_path(training=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    from jenga.model import model_config
    from openpi.shared import nnx_utils

    cfg = model_config()  # the served configuration (token budget 128)
    params = model_lib.restore_params(CHECKPOINT, restore_type=jax.Array, dtype=jnp.bfloat16)
    policy = cfg.load(params)
    policy.eval()
    del params
    sample = nnx_utils.module_jit(policy.sample_actions, static_argnames=("num_steps",))
    buckets = sorted(int(x) for x in args.buckets.split(","))
    if args.mode == "batch" and buckets[-1] < args.sessions:
        raise ValueError("largest bucket must hold every session")

    # One fixed observation and noise per session (host numpy, as a server would hold after decode).
    rng = np.random.default_rng(0)
    template = cfg.fake_obs(batch_size=1)
    sessions = []
    for _ in range(args.sessions):
        sessions.append({
            "images": {k: rng.uniform(-1, 1, v.shape).astype(np.float32) for k, v in template.images.items()},
            "image_masks": {k: np.ones(v.shape, bool) for k, v in template.image_masks.items()},
            "state": rng.uniform(-1, 1, template.state.shape).astype(np.float32),
            "tokenized_prompt": rng.integers(0, 2048, template.tokenized_prompt.shape).astype(np.int32),
            "tokenized_prompt_mask": np.ones(template.tokenized_prompt_mask.shape, bool),
            "noise": rng.standard_normal((1, cfg.action_horizon, cfg.action_dim)).astype(np.float32),
        })

    def run(rows):
        n = len(rows)
        pad = next(b for b in buckets if b >= n) if args.mode == "batch" else n
        rows = rows + [rows[-1]] * (pad - n)
        cat = lambda key: np.concatenate([r[key] for r in rows])  # noqa: E731
        obs = dataclasses.replace(
            template,
            images={k: jnp.asarray(np.concatenate([r["images"][k] for r in rows]), jnp.float32) for k in template.images},
            image_masks={k: jnp.asarray(np.concatenate([r["image_masks"][k] for r in rows])) for k in template.image_masks},
            state=jnp.asarray(cat("state")),
            tokenized_prompt=jnp.asarray(cat("tokenized_prompt")),
            tokenized_prompt_mask=jnp.asarray(cat("tokenized_prompt_mask")),
        )
        out = sample(jax.random.key(0), obs, num_steps=10, noise=jnp.asarray(cat("noise")))
        return np.asarray(out[:n])

    t0 = time.perf_counter()
    for b in buckets if args.mode == "batch" else [1]:
        run(sessions[:1] * b)
    compile_s = time.perf_counter() - t0

    # Batch invariance: a session's chunk in a padded batch vs the same request alone.
    alone = run(sessions[:1]) if args.mode == "serial" else None
    invariance = None
    if args.mode == "batch":
        alone = np.concatenate([run([s]) for s in sessions])
        together = run(sessions)
        invariance = float(np.abs(alone - together).max())

    work: queue.Queue = queue.Queue()

    def server():
        while True:
            item = work.get()
            if item is None:
                return
            batch = [item]
            if args.mode == "batch":
                deadline = time.perf_counter() + args.window_ms / 1e3
                while len(batch) < args.sessions:
                    try:
                        nxt = work.get(timeout=max(0.0, deadline - time.perf_counter()))
                    except queue.Empty:
                        break
                    if nxt is None:
                        work.put(None)
                        break
                    batch.append(nxt)
            batch_sizes.append(len(batch))
            outs = run([s for s, _ in batch])
            for (_, fut), o in zip(batch, outs, strict=True):
                fut.set_result(o)

    latencies = [[] for _ in sessions]
    batch_sizes = []  # requests per served batch (server thread only)

    def client(i):
        for k in range(args.requests + 5):
            fut = cf.Future()
            t = time.perf_counter()
            work.put((sessions[i], fut))
            fut.result()
            if k >= 5:
                latencies[i].append((time.perf_counter() - t) * 1e3)

    if args.barrier_dir:
        # Compiling neighbours steal host CPU; measure only the warm steady state a deployment would see.
        args.barrier_dir.mkdir(parents=True, exist_ok=True)
        (args.barrier_dir / args.tag).touch()
        deadline = time.time() + 600
        while len(list(args.barrier_dir.iterdir())) < args.barrier_n:
            if time.time() > deadline:
                raise TimeoutError("replica barrier not reached in 600 s")
            time.sleep(0.2)
    srv = threading.Thread(target=server, daemon=True)
    srv.start()
    t_start = time.perf_counter()
    threads = [threading.Thread(target=client, args=(i,)) for i in range(args.sessions)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t_start
    work.put(None)
    lat = np.concatenate([np.array(x) for x in latencies])
    result = {
        "question": "per-request latency of concurrent closed-loop sessions on one GPU replica",
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(jax.devices()[0]),
        "fast_path": fast,
        "compile_s": compile_s,
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(lat.max()),
        "chunks_per_s": args.sessions * (args.requests + 5) / wall,
        "mean_batch": float(np.mean(batch_sizes)),
        "batch_invariance_max_abs": invariance,
        "latencies_ms": [x.tolist() for x in map(np.array, latencies)],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "latencies_ms"}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_DONE", args.tag)


if __name__ == "__main__":
    main()
