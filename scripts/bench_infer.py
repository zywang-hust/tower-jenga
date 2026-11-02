"""Model-side inference latency of pi05 sample_actions (4 cams, H=32, 10 steps) on one GPU.

Uses real bf16 weights, as OpenPI serving does. Excludes tokenizer, transforms and network.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

from jenga.paths import PI05_BASE

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PI05_BASE / "params"  # env JENGA_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--fast", action="store_true", help="jenga.perf.enable_fast_path(training=False)")
    parser.add_argument("--gemma-unroll", type=int, default=1)
    parser.add_argument("--siglip-unroll", type=int, default=1)
    parser.add_argument("--max-token-len", type=int, default=200, help="prompt+state token budget (TOWER max 101)")
    parser.add_argument("--prompt-len", type=int, default=0,
                        help="real tokens per prompt (rest masked padding); 0 = all tokens real")
    parser.add_argument("--save-actions", type=Path, help="write batch-1 actions for equivalence checks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/bench/infer")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    fast = None
    if args.fast:
        from jenga.perf import enable_fast_path

        fast = enable_fast_path(training=False)
    import openpi.models.gemma as gemma_module
    import openpi.models.siglip as siglip_module

    gemma_module.SCAN_UNROLL = args.gemma_unroll
    siglip_module.SCAN_UNROLL = args.siglip_unroll
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    from jenga.model import model_config
    from openpi.shared import nnx_utils

    cfg = dataclasses.replace(model_config(), max_token_len=args.max_token_len)
    params = model_lib.restore_params(CHECKPOINT, restore_type=jax.Array, dtype=jnp.bfloat16)
    policy = cfg.load(params)
    policy.eval()
    del params
    # Static num_steps matches serving, where Policy leaves it at the Python default.
    sample = nnx_utils.module_jit(policy.sample_actions, static_argnames=("num_steps",))
    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        rng = np.random.default_rng(b)
        obs = cfg.fake_obs(batch_size=b)
        obs = dataclasses.replace(
            obs,
            images={k: jnp.asarray(rng.uniform(-1, 1, v.shape), v.dtype) for k, v in obs.images.items()},
            state=jnp.asarray(rng.uniform(-1, 1, obs.state.shape), jnp.float32),
            tokenized_prompt=jnp.asarray(rng.integers(0, 2048, obs.tokenized_prompt.shape), jnp.int32),
        )
        if args.prompt_len:
            # Same real tokens regardless of budget: draw 200 then truncate, so 128 vs 200 see identical prompts.
            tokens = np.random.default_rng(1000 + b).integers(0, 2048, (b, 200))[:, : args.max_token_len]
            mask = np.arange(args.max_token_len)[None, :].repeat(b, 0) < args.prompt_len
            obs = dataclasses.replace(obs, tokenized_prompt=jnp.asarray(np.where(mask, tokens, 0), jnp.int32),
                                      tokenized_prompt_mask=jnp.asarray(mask))
        key = jax.random.key(0)
        t0 = time.perf_counter()
        out = sample(key, obs, num_steps=args.denoise_steps).block_until_ready()
        compile_s = time.perf_counter() - t0
        if b == 1 and args.save_actions:
            args.save_actions.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_actions, np.asarray(out, np.float32))
        lat = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sample(key, obs, num_steps=args.denoise_steps).block_until_ready()
            lat.append((time.perf_counter() - t0) * 1e3)
        lat = np.array(lat)
        rows.append({"batch": b, "compile_s": compile_s, "p50_ms": float(np.percentile(lat, 50)),
                     "p99_ms": float(np.percentile(lat, 99)), "mean_ms": float(lat.mean()),
                     "chunks_per_s": b / (np.percentile(lat, 50) / 1e3)})
        print(rows[-1], flush=True)
    result = {"question": "model-side sample_actions latency vs batch on one GPU",
              "args": {k: str(v) for k, v in vars(args).items()},
              "device": str(jax.devices()[0]), "xla_flags": os.environ.get("XLA_FLAGS", ""), "fast_path": fast, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_DONE", args.tag)


if __name__ == "__main__":
    main()
