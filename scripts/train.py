"""Policy training on local TOWER HDF5 with the OpenPI pi05 train step (docs/recipe.md).

Data: runs/prep/split.json + runs/prep/norm_stats (scripts/prepare.py), variant mix, tail hold labels,
joint-delta/gripper-closure actions. Optimizer: AdamW, warmup then constant LR, EMA. Fast path from
jenga.perf. Run directory runs/<tag>: run.json, train.jsonl (one JSON line per logged step) and bf16
params_<step> exports; full train states go to runs/checkpoints/tower/<tag>/<step>. Writes the TRAIN_DONE
marker to stdout.

Fixed since r2 (docs/recipe.md §7): competition instructions as prompts, lighting augmentation, no geometric augmentation
of the external views. A follow-up run continues another run's train state (`--init-from`: params, EMA, optimizer
moments and step), e.g. with DART-style state noise (`--state-noise-bins K`) and a cosine cooldown from `--lr` at the
restored step to `--final-lr` at `--steps`.

Sampling options (docs/recipe.md §5): `--contact-weight F` draws frames within 1.5 s of a grasp or release F times as
often as the rest of their variant; `--synthetic-root` / `--synthetic-frac` mix in extra episodes (e.g. DART copies
of the train split): each variant with copies draws that share of its samples from them. Both off by default.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/openpi/scripts"))
from jenga.paths import DATA_ROOT, PI05_BASE  # noqa: E402

CHECKPOINT = str(PI05_BASE / "params")  # base weights (env JENGA_MODELS)
PROMPT = "official"  # competition instructions (jenga.instructions); run.json records it for serving and evaluation


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--dataset-root", type=Path, default=DATA_ROOT, help="TOWER dataset root (env JENGA_DATA_ROOT)")
    p.add_argument("--prep-dir", type=Path, default=ROOT / "runs/prep")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--save-interval", type=int, default=0,
                   help="n > 0: full train state every n steps and at the end (needed for --resume / --init-from)")
    p.add_argument("--export-interval", type=int, default=0,
                   help="every n steps write bf16 EMA params (OpenPI restore_params layout) to <run>/params_<step>")
    p.add_argument("--resume", action="store_true",
                   help="continue this tag's own checkpoints; repeat the original flags, including --seed as given (run.json "
                        "base_seed). With --init-from, the start step recorded in run.json is reused")
    p.add_argument("--init-from", type=Path, help="checkpoint dir of another run; start from its latest train state")
    p.add_argument("--final-lr", type=float, help="with --init-from: cosine from --lr to this over the remaining steps")
    p.add_argument("--state-noise-bins", type=int, default=0,
                   help="K > 0: DART state perturbation, each joint shifted by k whole state-token bins (256 over "
                        "[q01, q99], 2.6-7.2 mrad), k uniform in [-K, K], on half the samples")
    p.add_argument("--synthetic-root", type=Path,
                   help="extra episodes in the dataset layout (<root>/data/<variant_dir>/*.hdf5, e.g. DART copies of "
                        "the train split); a variant's synthetic copies take --synthetic-frac of its share")
    p.add_argument("--synthetic-frac", type=float, default=0.0,
                   help="within each variant that has synthetic copies, share of its samples drawn from them")
    p.add_argument("--contact-weight", type=float, default=1.0,
                   help="sample frames within 1.5 s of a grasp or release this many times as often as the rest of "
                        "their variant (docs/recipe.md §5)")
    p.add_argument("--seed", type=int, default=42, help="with --init-from, offset by the restored step")
    return p.parse_args()


def wait_for_free_gpus(timeout_s: float = 120.0) -> int:
    """JAX preallocates a fraction of *currently free* memory; a GPU still held by a finishing process
    gives a smaller pool and an OOM minutes later. Fail fast instead."""
    import subprocess

    def query(*args):
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader"], capture_output=True, text=True,
                              check=True).stdout.strip().splitlines()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    uuids = {u for i, u in (line.split(", ") for line in query("--query-gpu=index,uuid"))
             if visible is None or i in visible.split(",")}
    deadline = time.time() + timeout_s
    while True:
        out = "\n".join(line for line in query("--query-compute-apps=gpu_uuid,pid,used_memory")
                        if line.split(", ")[0] in uuids)
        if not out:
            return len(uuids)
        if time.time() > deadline:
            raise RuntimeError(f"GPUs still in use by other processes:\n{out}")
        time.sleep(2)


def main():
    args = parse()
    # The b64 step needs one ~51.5 GiB temp buffer (51.47 GiB in the OOM message); JAX's default 75% pool (60 GiB) cannot hold it next to
    # the 12.5 GiB state. Set before jax is imported; an explicit environment value still wins.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.94")
    n_gpus = wait_for_free_gpus()  # every visible GPU joins one FSDP mesh (4 on a half node, 8 on a full one)
    from jenga.perf import enable_fast_path

    fast = enable_fast_path(training=True, per_device_batch=args.batch // n_gpus)
    import jax
    import numpy as np
    import openpi.models.model as _model
    import openpi.shared.normalize as normalize
    import openpi.training.checkpoints as _checkpoints
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.sharding as sharding
    import openpi.training.weight_loaders as weight_loaders
    import torch
    import train as openpi_train

    from jenga.dataset import CONTACT_RADIUS, TowerDataset, collate
    from jenga.model import model_config, norm_stats_digest

    def latest_step(directory: Path) -> int | None:
        steps = [int(d.name) for d in directory.iterdir() if d.name.isdigit()] if directory.is_dir() else []
        return max(steps, default=None)

    base_seed, init_step = args.seed, None
    first_run_json = ROOT / "runs" / args.tag / "run.json"
    if args.init_from and args.resume and first_run_json.exists() and "init_step" in json.loads(first_run_json.read_text()):
        # The source run may since have been pruned or have saved later steps: take the step this run started from.
        init_step = json.loads(first_run_json.read_text())["init_step"]
    elif args.init_from:
        init_step = latest_step(args.init_from)
        if init_step is None or not init_step < args.steps:
            raise ValueError("--init-from needs a checkpoint below --steps (--steps is absolute)")
    if init_step is not None:
        args.seed = base_seed + init_step  # otherwise the sampler replays the source run's sample order from its step 0
    # Checkpoint directories are named by the optimizer step they hold (state.step). On --resume the sampler
    # continues from there instead of replaying this run's sample order from its first step.
    resume_step = latest_step(ROOT / "runs/checkpoints/tower" / args.tag) if args.resume else None
    first_step = resume_step if resume_step is not None else init_step or 0

    split = json.loads((args.prep_dir / "split.json").read_text())
    stats = normalize.load(args.prep_dir / "norm_stats")
    files = [p for t in split["tasks"].values() for p in t["train"]]
    run_dir = ROOT / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    synthetic = []
    if args.synthetic_root:
        # The synthetic set grows while it is generated: an existing list (from this run's start, or copied
        # from the run being compared against) pins the set.
        listing = run_dir / "synthetic_files.txt"
        if listing.exists():  # relative entries (configs/synthetic/*.txt) resolve against --synthetic-root
            synthetic = [p if Path(p).is_absolute() else str(args.synthetic_root / p) for p in listing.read_text().split()]
        else:
            synthetic = sorted(str(p) for p in args.synthetic_root.glob("data/*/*.hdf5"))
            listing.write_text("".join(f"{p}\n" for p in synthetic))
        val = [Path(p).stem for t in split["tasks"].values() for p in t["val"]]
        leaked = [p for p in synthetic if any(v in Path(p).name for v in val)]
        if leaked:
            raise ValueError(f"synthetic episodes derived from val episodes: {leaked[:3]}")
    dataset = TowerDataset(args.dataset_root, files, stats, augment_lighting=True, prompt=PROMPT,
                           state_noise_bins=args.state_noise_bins, synthetic_files=synthetic,
                           contact_radius=CONTACT_RADIUS if args.contact_weight != 1.0 else 0)
    weights = dataset.sampler_weights(args.synthetic_frac, args.contact_weight)
    mix = {}  # recorded in run.json
    if synthetic:
        mix["synthetic_files"] = len(synthetic)
        is_synthetic = dataset.synthetic[dataset.frames.index[:, 0]]
        mix["synthetic_samples"] = int(is_synthetic.sum())
        mix["synthetic_draw_share"] = float(weights[is_synthetic].sum() / weights.sum())
        mix["synthetic_variants"] = sorted(set(np.asarray(dataset.sample_variants())[is_synthetic].tolist()))
    if args.contact_weight != 1.0:
        mix["contact_samples"] = int(dataset.frames.contact.sum())
        mix["contact_draw_share"] = float(weights[dataset.frames.contact].sum() / weights.sum())
    weights = torch.as_tensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=(args.steps - first_step + 1) * args.batch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed + (resume_step or 0)),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler, num_workers=args.workers, collate_fn=collate,
        drop_last=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        multiprocessing_context="spawn" if args.workers else None,
    )

    @dataclasses.dataclass(frozen=True)
    class Cooldown(_optimizer.LRScheduleConfig):  # the schedule sees the optimizer count, which continues from start
        start: int
        end: int
        peak: float
        final: float

        def create(self):
            import jax.numpy as jnp

            def lr(count):
                frac = jnp.clip((count - self.start) / (self.end - self.start), 0.0, 1.0)
                return self.final + (self.peak - self.final) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
            return lr

    if init_step is not None and args.final_lr is not None:
        schedule = Cooldown(init_step, args.steps, args.lr, args.final_lr)
    else:
        schedule = _optimizer.CosineDecaySchedule(  # decay_lr == peak: constant after warmup (pi05_libero)
            warmup_steps=args.warmup, peak_lr=args.lr, decay_steps=max(args.steps, args.warmup + 1), decay_lr=args.lr)

    model_cfg = model_config()

    @dataclasses.dataclass(frozen=True)
    class BaseWeights(weight_loaders.CheckpointWeightLoader):
        def load(self, params):
            loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
            return weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*")

    config = _config.TrainConfig(
        name="tower", exp_name=args.tag, model=model_cfg,
        # --init-from overwrites every leaf, so the base weights would only cost a load
        weight_loader=weight_loaders.NoOpWeightLoader() if init_step is not None else BaseWeights(CHECKPOINT),
        lr_schedule=schedule,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=args.ema, batch_size=args.batch, fsdp_devices=n_gpus, num_train_steps=args.steps,
        seed=args.seed, wandb_enabled=False, checkpoint_base_dir=str(ROOT / "runs/checkpoints"),
    )
    (run_dir / ("run.json" if resume_step is None else f"run_resume_{resume_step}.json")).write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "prompt": PROMPT, "gpus": n_gpus, "fast_path": fast, "train_files": len(files),
        "samples": len(dataset), "split_seed": split["seed"], "base_seed": base_seed, "init_step": init_step,
        "norm_stats_sha256": norm_stats_digest(args.prep_dir / "norm_stats"),
    } | mix, indent=2) + "\n")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    class _Assets:  # what checkpoints.save_state needs from a data loader
        def data_config(self):
            return type("C", (), {"norm_stats": stats, "asset_id": "tower"})()

    mngr = None
    resuming = False
    if args.save_interval or args.resume:
        mngr, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=None, overwrite=False, resume=args.resume)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = openpi_train.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = _checkpoints.restore_state(mngr, state, None)
    elif init_step is not None:
        src, _ = _checkpoints.initialize_checkpoint_dir(args.init_from, keep_period=None, overwrite=False, resume=True)
        # openpi's restore_state reuses the saved shardings, which fails when the source ran on devices this
        # mesh lacks (r2 on 8 GPUs onto 4: "sharding ... Got None"). Restore with explicit target shardings.
        import orbax.checkpoint as ocp
        from openpi.shared import array_typing as at

        target = jax.tree.map(lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
                              state, state_sharding)
        del state
        with at.disable_typechecking():
            train_state, params = _checkpoints._split_params(target)
            items = {"train_state": train_state, "params": {"params": params}}
            restored = src.restore(init_step, args=ocp.args.Composite(**{
                k: ocp.args.PyTreeRestore(item=v, restore_args=ocp.checkpoint_utils.construct_restore_args(v))
                for k, v in items.items()}))
        state = _checkpoints._merge_params(restored["train_state"], restored["params"])
        del restored
        print(json.dumps({"init_from": str(args.init_from), "step": int(state.step)}), flush=True)
    # Compact device memory: weight loading and state init leave holes, and at b64 the step needs one
    # ~51.5 GiB contiguous temp buffer. Whether it fit depended on that layout (first-step OOM in 4 of 9
    # runs). A host round trip of the 12.5 GiB/GPU state packs it at the start of a free pool.
    t0 = time.perf_counter()
    host_state = jax.device_get(state)
    del state
    gc.collect()
    state = jax.device_put(host_state, state_sharding)
    jax.block_until_ready(state)
    del host_state
    print(json.dumps({"compacted_s": time.perf_counter() - t0,
                      "device0_in_use_gib": jax.local_devices()[0].memory_stats()["bytes_in_use"] / 2**30}), flush=True)
    step_fn = jax.jit(
        functools.partial(openpi_train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,),
    )

    def to_host(batch):
        actions = batch.pop("actions")
        return _model.Observation.from_dict(batch), actions

    def to_device(host_batch):
        return jax.device_put(host_batch, data_sharding)

    def export_params(state, step):
        # ~4.9 GiB per snapshot instead of the ~42 GiB full train state; serving loads bf16 anyway.
        import orbax.checkpoint as ocp

        out = run_dir / f"params_{step}"
        if out.exists():
            if step == resume_step:  # a finished run resumed: same state, keep it
                return
            # Retrained after a resume from an earlier checkpoint: the old export is from the abandoned trajectory.
            out.rename(run_dir / f"params_{step}.superseded_{time.strftime('%Y%m%d%H%M%S')}")
        params = state.ema_params if state.ema_params is not None else state.params
        pure = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), params.to_pure_dict())
        with ocp.PyTreeCheckpointer() as ckptr:
            ckptr.save(out / "params", {"params": pure})
        print(json.dumps({"exported": step}), flush=True)

    # Host-side uint8->float conversion runs in a thread (60-100 ms/step when inline). The
    # host->device copy stays on the main thread, issued right after the step is dispatched: a copy landing
    # at an arbitrary time can split the free region the ~51.5 GiB step temp buffer needs (intermittent OOM);
    # copied while that buffer is live, it lands outside it.
    ready: queue.Queue = queue.Queue(maxsize=2)

    def prefetch():
        try:
            for b in loader:
                ready.put(to_host(b))
        except Exception as exc:  # surface loader errors in the main thread
            ready.put(exc)
        ready.put(None)

    threading.Thread(target=prefetch, daemon=True).start()
    log = (run_dir / "train.jsonl").open("a")
    start = int(state.step)
    t_prev = time.perf_counter()

    def next_host(step):
        b = ready.get()
        if b is None or isinstance(b, Exception):
            raise RuntimeError(f"data loader ended at step {step}") from b
        return b

    with sharding.set_mesh(mesh):
        batch = to_device(next_host(start))
        # What is resident before the first step (the step needs one ~51.5 GiB temp buffer at b64).
        live = jax.live_arrays()
        by_shape: dict = {}
        for a in live:
            key = f"{a.dtype}{list(a.shape)}"
            by_shape[key] = by_shape.get(key, 0) + a.nbytes
        top = sorted(by_shape.items(), key=lambda kv: -kv[1])[:8]
        print(json.dumps({"live_arrays": len(live), "live_gib_total": sum(a.nbytes for a in live) / 2**30,
                          "top_gib": {k: round(v / 2**30, 2) for k, v in top},
                          "device0_in_use_gib": jax.local_devices()[0].memory_stats().get("bytes_in_use", 0) / 2**30}),
              flush=True)
        for step in range(start, args.steps):
            state, info = step_fn(train_rng, state, batch)  # async dispatch; temp buffer now allocated
            t0 = time.perf_counter()
            batch = to_device(next_host(step + 1)) if step + 1 < args.steps else None
            wait = time.perf_counter() - t0
            if step % 10 == 0 or step == args.steps - 1:
                info = jax.device_get(info)
                now = time.perf_counter()
                rec = {"step": step, "loss": float(info["loss"]), "grad_norm": float(info["grad_norm"]),
                       "loader_wait_s": wait, "wall_s": now - t_prev}
                t_prev = now
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            done = step + 1  # state.step after this update: snapshots are named by the step they hold
            if args.export_interval and done < args.steps and done % args.export_interval == 0:
                export_params(state, done)
            if mngr and args.save_interval and done < args.steps and done % args.save_interval == 0:
                _checkpoints.save_state(mngr, state, _Assets(), done)
    if args.export_interval:
        export_params(state, args.steps)
    if mngr:
        _checkpoints.save_state(mngr, state, _Assets(), args.steps)
        mngr.wait_until_finished()
    print("TRAIN_DONE", args.tag, flush=True)


if __name__ == "__main__":
    main()
