"""Model-side training throughput for the TOWER pi05 config, using OpenPI's own train_step.

The batch is a fixed device-resident fake batch, so the number isolates model compute and
communication from data loading. It says nothing about task quality.
"""
import argparse
import dataclasses
import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/openpi/scripts"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import openpi.models.gemma as gemma_module  # noqa: E402
import openpi.models.pi0 as pi0_module  # noqa: E402
import openpi.models.siglip as siglip_module  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402
import train as openpi_train  # noqa: E402

from jenga.model import model_config  # noqa: E402
from jenga.paths import PI05_BASE  # noqa: E402

CHECKPOINT = str(PI05_BASE / "params")  # env JENGA_MODELS


def make_config(args) -> _config.TrainConfig:
    model = dataclasses.replace(model_config(), max_token_len=args.max_token_len)
    loader = weight_loaders.CheckpointWeightLoader(CHECKPOINT) if args.load_weights else weight_loaders.NoOpWeightLoader()
    return _config.TrainConfig(
        name="tower_bench", exp_name=args.tag, model=model, weight_loader=loader,
        batch_size=args.batch, fsdp_devices=args.fsdp,
        ema_decay=None if args.no_ema else 0.99, wandb_enabled=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fsdp", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--load-weights", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/bench/train")
    parser.add_argument("--gemma-remat", default="nothing_saveable", help="jax.checkpoint_policies name")
    parser.add_argument("--siglip-remat", default="nothing_saveable", help="jax.checkpoint_policies name")
    parser.add_argument("--per-view-siglip", action="store_true", help="upstream per-camera SigLIP calls")
    parser.add_argument("--siglip-attention", choices=["flax", "cudnn"], default="flax")
    parser.add_argument("--gemma-natural-layout", action="store_true", help="gemma.ATTENTION_NATURAL_LAYOUT")
    parser.add_argument("--gemma-unroll", type=int, default=1)
    parser.add_argument("--siglip-unroll", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1, help="train.GRAD_ACCUM_STEPS")
    parser.add_argument("--max-token-len", type=int, default=200, help="prompt+state token budget (TOWER max 101)")
    parser.add_argument("--fast-path", action="store_true", help="jenga.perf.enable_fast_path; ignores model flags")
    args = parser.parse_args()
    openpi_train.GRAD_ACCUM_STEPS = args.grad_accum
    gemma_module.SCAN_UNROLL = args.gemma_unroll
    siglip_module.SCAN_UNROLL = args.siglip_unroll
    if args.gemma_natural_layout:
        gemma_module.ATTENTION_NATURAL_LAYOUT = True
    if args.siglip_attention != "flax":
        siglip_module.ATTENTION_IMPL = args.siglip_attention
    if args.per_view_siglip:
        pi0_module.BATCH_IMAGE_VIEWS = False
    gemma_module.REMAT_POLICY = args.gemma_remat
    siglip_module.REMAT_POLICY = args.siglip_remat
    fast_path = None
    if args.fast_path:
        from jenga.perf import enable_fast_path

        fast_path = enable_fast_path(training=True, per_device_batch=args.batch // args.fsdp)  # benches use fsdp == GPUs; no backend init yet
    config = make_config(args)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    t0 = time.perf_counter()
    state, state_sharding = openpi_train.init_train_state(config, jax.random.key(1), mesh, resume=False)
    jax.block_until_ready(state)
    init_s = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    obs_spec, act_spec = config.model.inputs_spec(batch_size=config.batch_size)

    def fake(spec):
        if spec.dtype == np.float32:
            return rng.uniform(-1, 1, spec.shape).astype(np.float32)
        if spec.dtype == np.int32:
            return rng.integers(0, 2048, spec.shape, dtype=np.int32)
        return np.ones(spec.shape, dtype=spec.dtype)

    batch = jax.device_put((jax.tree.map(fake, obs_spec), jax.tree.map(fake, act_spec)), data_sharding)

    step_fn = jax.jit(
        functools.partial(openpi_train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    train_rng = jax.random.key(2)
    times, losses, grad_norms, param_norms = [], [], [], []
    with sharding.set_mesh(mesh):
        t0 = time.perf_counter()
        state, info = step_fn(train_rng, state, batch)
        losses.append(float(info["loss"]))
        compile_s = time.perf_counter() - t0
        for i in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            state, info = step_fn(train_rng, state, batch)
            loss = float(info["loss"])  # blocks
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
            losses.append(loss)
            grad_norms.append(float(info["grad_norm"]))
            param_norms.append(float(info["param_norm"]))

    peak = [d.memory_stats().get("peak_bytes_in_use", 0) / 2**30 for d in jax.local_devices()]
    times_ms = np.array(times) * 1e3
    result = {
        "question": "model-side train step time of OpenPI pi05 (4 cams, H=32) on local GPUs, fixed fake batch",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "devices": [str(d) for d in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
        "fast_path": fast_path,
        "init_seconds": init_s,
        "first_step_seconds_incl_compile": compile_s,
        "step_ms_median": float(np.median(times_ms)),
        "step_ms_mean": float(times_ms.mean()),
        "step_ms_min": float(times_ms.min()),
        "step_ms_max": float(times_ms.max()),
        "samples_per_s": args.batch / (np.median(times_ms) / 1e3),
        "peak_gib_per_device": peak,
        "losses": losses,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "finite": bool(np.isfinite(losses).all()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("norms") and k != "losses"}, indent=2))
    print("BENCH_DONE", args.tag)


if __name__ == "__main__":
    main()
