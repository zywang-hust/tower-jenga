"""Compare chunk stitching rules on chunks saved by scripts/evaluate.py (no GPU).

Question: which inference-time rule brings the executed command stream's chunk-boundary jumps down to the
demonstration's own step size without raising its error against the demonstration? Input is
runs/eval/<tag>/chunks/*.npz: predictions every K frames over a contiguous stretch, each a full 32-step
absolute chunk. The executed stream takes K commands per chunk; rules differ in how the previous chunks'
overlapping predictions are mixed in. Output: runs/eval/<tag>/stitch.json for one tag, or
runs/eval/stitch_multi_<tag1>_<tag2>....json for several; entries are keyed <tag>/<rule> and merged into an
existing report, so re-running a subset of rules does not drop the others.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jenga.stitching import ChunkFader
from jenga.transforms import JOINTS

ROOT = Path(__file__).resolve().parents[1]


def stitch(pred: np.ndarray, k: int, rule: str) -> np.ndarray:
    """pred [C, H, D] absolute chunks issued every k frames -> executed stream [C*k, D]."""
    n, h, _ = pred.shape
    if rule.startswith(("fade", "sg")):  # the serving implementation itself: "fade4", "sg9", "sg9fade4"
        sg, _, fade = rule.removeprefix("sg").partition("fade") if rule.startswith("sg") else ("0", "", rule[4:])
        fader = ChunkFader(int(fade or 0), smooth=int(sg))
        return np.concatenate([fader(pred[c], k)[:k] for c in range(n)])
    out = []
    for c in range(n):
        for j in range(k):
            new = pred[c, j]
            if rule == "none" or c == 0:
                out.append(new)
            elif rule.startswith("ens"):  # temporal ensemble of every chunk still covering this frame
                decay = float(rule[3:])
                ages = [a for a in range(c + 1) if a * k + j < h]
                wts = np.array([decay**a for a in ages])
                out.append(sum(w * pred[c - a, a * k + j] for w, a in zip(wts, ages, strict=True)) / wts.sum())
            else:
                raise ValueError(rule)
    return np.stack(out)


def metrics(pred: np.ndarray, demo: np.ndarray, k: int, rule: str) -> dict:
    e = stitch(pred, k, rule)[:, JOINTS]
    d = demo[:, :k].reshape(-1, demo.shape[-1])[:, JOINTS]
    bounds = np.arange(k, len(e), k)
    step = lambda x: np.abs(np.diff(x, axis=0)).max(-1)  # noqa: E731
    return {
        "mae_rad": float(np.abs(e - d).mean()),
        "jump_p95_rad": float(np.percentile(step(e)[bounds - 1], 95)),
        "demo_jump_p95_rad": float(np.percentile(step(d)[bounds - 1], 95)),
        "step_p95_rad": float(np.percentile(step(e), 95)),
        "rough_rad": float(np.abs(np.diff(e, 2, axis=0)).mean()),
        "demo_rough_rad": float(np.abs(np.diff(d, 2, axis=0)).mean()),
        "demo_step_p95_rad": float(np.percentile(step(d), 95)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--rules", default="none,fade2,fade4,sg9fade2,fade8,ens1.0,ens0.5")
    args = p.parse_args()
    report = {}
    for tag in args.tags:
        files = sorted((ROOT / "runs/eval" / tag / "chunks").glob("*.npz"))
        if not files:
            raise SystemExit(f"no chunks under runs/eval/{tag}/chunks")
        for rule in args.rules.split(","):
            by_var: dict[str, list[dict]] = {}
            for f in files:
                z = np.load(f)
                by_var.setdefault(str(z["variant"]), []).append(metrics(z["pred"], z["demo"], int(z["k"]), rule))
            per_var = {v: {m: float(np.mean([r[m] for r in rs])) for m in rs[0]} for v, rs in sorted(by_var.items())}
            macro = {m: float(np.mean([v[m] for v in per_var.values()])) for m in next(iter(per_var.values()))}
            report[f"{tag}/{rule}"] = {"episodes": len(files), "macro": macro, "per_variant": per_var}
            print(f"{tag:14s} {rule:7s} " + " ".join(f"{m}={x:.4f}" for m, x in macro.items()), flush=True)
    if len(args.tags) == 1:
        out = ROOT / "runs/eval" / args.tags[0] / "stitch.json"
    else:
        out = ROOT / "runs/eval" / f"stitch_multi_{'_'.join(args.tags)}.json"
    if out.exists():
        report = json.loads(out.read_text()) | report
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("STITCH_DONE", out)


if __name__ == "__main__":
    main()
