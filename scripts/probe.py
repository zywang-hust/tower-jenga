"""Offline sensitivity probes for a params snapshot on held-out frames (docs/research.md §1.2).

Question: what does the policy's action chunk depend on, and does it correct a perturbed state? Open-loop
probes cannot show closed-loop success; they show whether the inputs that must drive generalization (external
views for block pose, state for correction, prompt for task identity) actually move the output.

For every probed frame the same fixed noise is used for the baseline and each perturbation, so differences
come from the input alone. Reported per variant, joints only, first k executed steps, in rad:
  sens/<probe>  mean |A_probe - A_base|                          (input dependence)
  mae/<probe>   mean |A_probe - demo|                            (does the change help or hurt)
  recovery/<d>  1 - <A_probe - A_base, delta> / |delta|^2 at step k, for a joint-state offset delta of size d:
                0 = the offset is carried through (no correction), 1 = full return to the unperturbed command.
  gain/swap_views  projection of the change from swapping in another held-out episode's four views (same
                variant, same progress, own state) onto the difference between the two demonstrations' motions:
                ~1 = the policy positions by what it sees, ~0 = it replays from state and timing.
Probes: blank_ext, blank_wrist (camera group set to mid-grey), shift_ext_<px> (external views translated),
state_<d> (joint state offset of d rad, random sign per joint), prompt_<mode> (train / official / sibling:
the other X-2/X-3 variant's training instruction). One GPU per process; --shard i/n; then --summarize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL, WRIST = ("base_0_rgb", "second_base_rgb"), ("left_wrist_0_rgb", "right_wrist_0_rgb")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--params", type=Path)
    p.add_argument("--dataset-root", type=Path, default=DATA_ROOT, help="TOWER dataset root (env JENGA_DATA_ROOT)")
    p.add_argument("--prep-dir", type=Path, help="split + norm stats; default: the snapshot's run (run.json)")
    p.add_argument("--prompt", choices=("train", "official"), help="default: the snapshot's training prompt")
    p.add_argument("--stride", type=int, default=40, help="probe every n-th frame")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--shifts", default="8,16", help="external-view translations in px (224 px input)")
    p.add_argument("--deltas", default="0.004,0.008,0.016", help="joint-state offsets in rad")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.params is not None and (args.prompt is None or args.prep_dir is None):
        # prompt and prep dir follow the snapshot unless given (jenga.model.snapshot_inputs)
        from jenga.model import snapshot_inputs

        prompt, prep = snapshot_inputs(args.params)
        args.prompt, args.prep_dir = args.prompt or prompt, args.prep_dir or prep
    return args


def shift(image, px: int):
    import numpy as np

    out = np.roll(image, (px, px), axis=(0, 1))
    out[:px], out[:, :px] = image[:1], image[:, :1]  # edge-replicate the wrapped border
    return out


def summarize(out_dir: Path) -> dict:
    import numpy as np

    rows = [json.loads(line) for f in sorted(out_dir.glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    keys = sorted({k for r in rows for k in r if "/" in k})
    by_var: dict[str, list] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    per_variant = {v: {k: float(np.mean([r[k] for r in rs if k in r])) for k in keys if any(k in r for r in rs)}
                   for v, rs in sorted(by_var.items())}
    macro = {k: float(np.mean([pv[k] for pv in per_variant.values() if k in pv])) for k in keys}
    summary = {"frames": len(rows), "macro": macro, "per_variant": per_variant}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    args = parse()
    out_dir = ROOT / "runs/probe" / args.tag
    if args.summarize:
        print(json.dumps(summarize(out_dir)["macro"]))
        print("PROBE_SUMMARY_DONE", args.tag)
        return
    from jenga.perf import enable_fast_path

    enable_fast_path(training=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.models.model as _model
    import openpi.shared.normalize as normalize
    from openpi.shared import nnx_utils

    from jenga.dataset import TowerDataset, collate
    from jenga.hdf5_dataset import MODEL_CAMERAS
    from jenga.instructions import OFFICIAL_INSTRUCTIONS, TRAIN_INSTRUCTIONS
    from jenga.model import load_policy
    from jenga.transforms import GRIPPER_FULL_WIDTH_M, JOINTS, from_model_actions

    split = json.loads((args.prep_dir / "split.json").read_text())
    stats = normalize.load(args.prep_dir / "norm_stats")
    q01, q99 = stats["actions"].q01, stats["actions"].q99
    episodes = sorted(p for t in split["tasks"].values() for p in t["val"])
    # Partner for the view swap: the next held-out episode of the same variant (different layout, same task).
    partner = {}
    for t in split["tasks"].values():
        val = sorted(t["val"])
        partner.update({a: b for a, b in zip(val, val[1:] + val[:1], strict=True)})
    i, n = (int(x) for x in args.shard.split("/"))
    episodes = episodes[i::n]
    shifts = [int(x) for x in args.shifts.split(",") if x]
    deltas = [float(x) for x in args.deltas.split(",") if x]

    policy = load_policy(args.params)
    sample = nnx_utils.module_jit(policy.sample_actions, static_argnames=("num_steps",))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"shard_{i}.jsonl").open("w")
    for ep in episodes:
        variant = Path(ep).parent.name.split("_")[0]
        sibling = variant[:-1] + ("3" if variant.endswith("2") else "2")
        ds = TowerDataset(args.dataset_root, [ep], stats)
        other = TowerDataset(args.dataset_root, [partner[ep]], stats)
        frames = list(range(0, len(ds), args.stride))
        for t in frames:
            raw = ds.frames[t]
            raw_b = other.frames[min(len(other) - 1, round(t * len(other) / len(ds)))]  # same task progress
            rng = np.random.default_rng(t)
            signs = rng.choice([-1.0, 1.0], len(JOINTS)).astype(np.float32)
            own = TRAIN_INSTRUCTIONS[variant] if args.prompt == "train" else OFFICIAL_INSTRUCTIONS[variant[:5]]
            cases = {"base": (raw["images"], raw["state"], own)}
            grey = np.full_like(raw["images"][MODEL_CAMERAS[0]], 128)
            cases["blank_ext"] = ({k: grey if k in EXTERNAL else v for k, v in raw["images"].items()},
                                  raw["state"], own)
            cases["blank_wrist"] = ({k: grey if k in WRIST else v for k, v in raw["images"].items()},
                                    raw["state"], own)
            for px in shifts:
                cases[f"shift_ext_{px}"] = ({k: shift(v, px) if k in EXTERNAL else v for k, v in raw["images"].items()},
                                            raw["state"], own)
            for d in deltas:
                s = raw["state"].copy()
                s[JOINTS] += d * signs
                cases[f"state_{d:g}"] = (raw["images"], s, own)
            cases["swap_views"] = (raw_b["images"], raw["state"], own)
            for mode, text in (("train", TRAIN_INSTRUCTIONS[variant]), ("official", OFFICIAL_INSTRUCTIONS[variant[:5]]),
                               ("sibling", TRAIN_INSTRUCTIONS[sibling])):
                if text != own:
                    cases[f"prompt_{mode}"] = (raw["images"], raw["state"], text)

            names = list(cases)
            items = [ds.encode(*cases[name][:2], raw["actions"], cases[name][2]) for name in names]
            pad = (-len(items)) % 4
            batch = collate(items + [items[-1]] * pad)
            batch.pop("actions")
            obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(batch))
            noise = jnp.asarray(np.broadcast_to(np.random.default_rng(t).standard_normal((32, 32)),
                                                (len(items) + pad, 32, 32)), jnp.float32)
            pred = np.asarray(sample(jax.random.key(0), obs, num_steps=10, noise=noise), np.float32)[:, :, :16]
            A = {name: from_model_actions((pred[j] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, cases[name][1],
                                          GRIPPER_FULL_WIDTH_M)[: args.k] for j, name in enumerate(names)}
            demo = raw["actions"][: args.k]
            base = A["base"][:, JOINTS]
            rec = {"episode": ep, "variant": variant, "frame": t,
                   "mae/base": float(np.abs(base - demo[:, JOINTS]).mean())}
            for name in names[1:]:
                a = A[name][:, JOINTS]
                rec[f"sens/{name}"] = float(np.abs(a - base).mean())
                rec[f"mae/{name}"] = float(np.abs(a - demo[:, JOINTS]).mean())
                if name.startswith("state_"):
                    delta = cases[name][1][JOINTS] - raw["state"][JOINTS]
                    carried = float(np.dot(a[-1] - base[-1], delta) / np.dot(delta, delta))
                    rec[f"recovery/{name[6:]}"] = 1.0 - carried
            # Visual servo gain: does swapping in the partner's views move the predicted motion (relative to
            # the state) by the amount the two demonstrations' motions differ? 1 = follows the views, 0 = ignores.
            want = (raw_b["actions"][: args.k, JOINTS] - raw_b["state"][JOINTS]) - (demo[:, JOINTS] - raw["state"][JOINTS])
            got = A["swap_views"][:, JOINTS] - base
            if np.dot(want.ravel(), want.ravel()) > (1e-3) ** 2 * want.size:  # demos differ by > 1 mrad rms
                rec["gain/swap_views"] = float(np.dot(got.ravel(), want.ravel()) / np.dot(want.ravel(), want.ravel()))
            out.write(json.dumps(rec) + "\n")
            out.flush()
        print(json.dumps({"episode": ep, "frames": len(frames)}), flush=True)
    print("PROBE_SHARD_DONE", args.tag, i, flush=True)


if __name__ == "__main__":
    main()
