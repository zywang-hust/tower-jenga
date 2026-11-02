"""Offline checkpoint evaluation on held-out episodes (docs/evaluation.md).

Question: for a params snapshot, how close are executed-prefix actions to the demonstrations in physical
units, per task variant, and how large are chunk-boundary jumps when chunks are stitched every K frames?
Predictions are made every K frames along each val episode from the real observation (open loop:
observations come from the demo, not from executing our actions). Fixed noise per prediction. One GPU per
process; shard episodes with --shard i/n. Writes runs/eval/<tag>/shard_<i>.jsonl, then --summarize.

`--frames contact` instead predicts at every K-th frame within 1.5 s of a grasp or release (the contact phases
`train.py --contact-weight` emphasizes) over the whole episode, and adds grasp_/release_mae8_rad. Jumps are then
measured only between adjacent predictions; the chunks are not a contiguous stretch, so skip stitch.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--params", type=Path, help="snapshot dir containing params/ (OpenPI layout)")
    p.add_argument("--dataset-root", type=Path, default=DATA_ROOT, help="TOWER dataset root (env JENGA_DATA_ROOT)")
    p.add_argument("--prep-dir", type=Path, help="split + norm stats; default: the snapshot's run (run.json)")
    p.add_argument("--k", type=int, default=8, help="execution horizon used for stitching")
    p.add_argument("--max-chunks", type=int, default=120, help="per episode, evenly spread contiguous run")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--noise", choices=("frame", "episode"), default="frame",
                   help="frame: independent noise per prediction; episode: one noise draw reused for every chunk")
    p.add_argument("--num-steps", type=int, default=10, help="flow-matching Euler steps")
    p.add_argument("--frames", choices=("centre", "contact"), default="centre",
                   help="centre: --max-chunks contiguous chunks mid-episode; contact: near grasp/release (docstring)")
    p.add_argument("--prompt", choices=("train", "official"),
                   help="default: the snapshot's training prompt; official: the competition's per-family instruction (jenga.instructions.OFFICIAL_INSTRUCTIONS)")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.params is not None and (args.prompt is None or args.prep_dir is None):
        # prompt and prep dir follow the snapshot unless given (jenga.model.snapshot_inputs)
        from jenga.model import snapshot_inputs

        prompt, prep = snapshot_inputs(args.params)
        args.prompt, args.prep_dir = args.prompt or prompt, args.prep_dir or prep
    return args


def summarize(out_dir: Path) -> dict:
    import numpy as np

    rows = [json.loads(line) for f in sorted(out_dir.glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    by_var: dict[str, list[dict]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    def agg(rs, key):
        return float(np.mean([r[key] for r in rs]))

    contact = ("grasp_mae8_rad", "release_mae8_rad") if rows and "grasp_mae8_rad" in rows[0] else ()
    summary = {
        "episodes": len(rows),
        "per_variant": {
            v: {k: agg(rs, k) for k in ("mae4_rad", "mae8_rad", "hold4_rad", "hold8_rad", "grip8_m",
                                        "jump_p95_rad", "demo_step_p95_rad", *contact)} | {"episodes": len(rs)}
            for v, rs in sorted(by_var.items())
        },
    }
    summary["macro"] = {k: float(np.mean([v[k] for v in summary["per_variant"].values()]))
                        for k in ("mae4_rad", "mae8_rad", "hold8_rad", "grip8_m", "jump_p95_rad", "demo_step_p95_rad",
                                  *contact)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse()
    out_dir = ROOT / "runs/eval" / args.tag
    if args.summarize:
        print(json.dumps(summarize(out_dir)["macro"]))
        print("SUMMARY_DONE", args.tag)
        return
    from jenga.perf import enable_fast_path

    enable_fast_path(training=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.models.model as _model
    import openpi.shared.normalize as normalize
    from openpi.shared import nnx_utils

    import h5py

    from jenga.dataset import CONTACT_RADIUS, TowerDataset, collate
    from jenga.hdf5_dataset import contact_frames
    from jenga.model import load_policy
    from jenga.transforms import GRIPPER_FULL_WIDTH_M, GRIPPERS, JOINTS, from_model_actions

    split = json.loads((args.prep_dir / "split.json").read_text())
    stats = normalize.load(args.prep_dir / "norm_stats")
    q01, q99 = stats["actions"].q01, stats["actions"].q99
    episodes = sorted(p for t in split["tasks"].values() for p in t["val"])
    i, n = (int(x) for x in args.shard.split("/"))
    episodes = episodes[i::n]

    policy = load_policy(args.params)
    sample = nnx_utils.module_jit(policy.sample_actions, static_argnames=("num_steps",))

    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"shard_{i}.jsonl").open("w")
    for ep in episodes:
        variant = Path(ep).parent.name.split("_")[0]
        ds = TowerDataset(args.dataset_root, [ep], stats)
        if args.prompt == "official":
            from jenga.instructions import OFFICIAL_INSTRUCTIONS

            ds.frames.instructions = [OFFICIAL_INSTRUCTIONS[variant[:5]]]
        n_frames = len(ds)  # tail-hold samples: sample t starts at frame t
        if args.frames == "contact":
            with h5py.File(args.dataset_root / ep, "r") as f:
                grasp, release = contact_frames(f, CONTACT_RADIUS)
            frames = [t for t in range(0, n_frames, args.k) if grasp[t] or release[t]]
        else:
            n_chunks = min(args.max_chunks, n_frames // args.k)
            first = (n_frames - n_chunks * args.k) // 2  # a contiguous stretch centred in the episode
            frames = [first + c * args.k for c in range(n_chunks)]
        preds = []
        for b0 in range(0, len(frames), args.batch):
            ids = frames[b0 : b0 + args.batch]
            pad = ids + [ids[-1]] * (args.batch - len(ids))
            batch = collate([ds[t] for t in pad])
            batch.pop("actions")
            obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(batch))
            seeds = [0] * len(pad) if args.noise == "episode" else pad
            noise = jnp.asarray(np.stack([np.random.default_rng(t).standard_normal((32, 32)) for t in seeds]),
                                jnp.float32)
            pred = np.asarray(sample(jax.random.key(0), obs, num_steps=args.num_steps, noise=noise),
                              np.float32)[: len(ids), :, :16]
            preds.extend(pred)
        absolute, targets, states = [], [], []
        for t, p in zip(frames, preds, strict=True):
            raw = ds.frames[t]
            model_space = (p + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            absolute.append(from_model_actions(model_space, raw["state"], GRIPPER_FULL_WIDTH_M))
            targets.append(raw["actions"])
            states.append(raw["state"])
        A, T, S = np.stack(absolute), np.stack(targets), np.stack(states)
        # Full 32-step chunks for offline stitching analysis (scripts/stitch.py).
        np.savez(out_dir / "chunks" / (Path(ep).stem + ".npz"), pred=A, demo=T, state=S, k=args.k,
                 variant=variant, frames=np.asarray(frames))
        k = args.k
        adjacent = np.flatnonzero(np.diff(frames) == k)  # every pair for --frames centre
        A1, A0, T1, T0 = A[adjacent + 1], A[adjacent], T[adjacent + 1], T[adjacent]
        rec = {
            "episode": ep,
            "variant": variant,
            "chunks": len(frames),
            "mae4_rad": float(np.abs(A[:, :4, JOINTS] - T[:, :4, JOINTS]).mean()),
            "mae8_rad": float(np.abs(A[:, :8, JOINTS] - T[:, :8, JOINTS]).mean()),
            "hold4_rad": float(np.abs(S[:, None, JOINTS] - T[:, :4, JOINTS]).mean()),
            "hold8_rad": float(np.abs(S[:, None, JOINTS] - T[:, :8, JOINTS]).mean()),
            "grip8_m": float(np.abs(A[:, :8][..., GRIPPERS] - T[:, :8][..., GRIPPERS]).mean()),
            # Stitching: first command of chunk c+1 vs last executed command of chunk c (max over joints).
            "jump_p95_rad": float(np.percentile(np.abs(A1[:, 0][:, JOINTS] - A0[:, k - 1][:, JOINTS]).max(-1), 95))
            if len(adjacent) else 0.0,
            # Reference: the demonstration's own one-step command change at the same boundaries.
            "demo_step_p95_rad": float(np.percentile(np.abs(T1[:, 0][:, JOINTS] - T0[:, k - 1][:, JOINTS]).max(-1), 95))
            if len(adjacent) else 0.0,
        }
        if args.frames == "contact":
            err8 = np.abs(A[:, :8, JOINTS] - T[:, :8, JOINTS]).mean((1, 2))
            for name, mask in (("grasp", grasp), ("release", release)):
                rec[f"{name}_mae8_rad"] = float(err8[mask[frames]].mean())
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)
    print("EVAL_SHARD_DONE", args.tag, i, flush=True)


if __name__ == "__main__":
    main()
