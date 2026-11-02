"""Side-by-side eval + probe metrics for snapshots (no GPU).

Reads runs/eval/<tag>/summary.json (scripts/evaluate.py) and runs/probe/<tag>/summary.json (scripts/probe.py)
for any number of tags; a missing summary prints as nan. Per-variant tables follow when every tag has probes.
Usage: python scripts/compare.py r1-25000 pA-27000 pB-27000 pC-27000
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ("mae8_rad", "jump_p95_rad")
PROBE = ("mae/base", "gain/swap_views", "sens/shift_ext_16", "sens/blank_ext", "sens/blank_wrist",
         "recovery/0.004", "recovery/0.008", "recovery/0.016", "sens/prompt_sibling", "sens/prompt_train")


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    tags = sys.argv[1:]
    if not tags:
        raise SystemExit(__doc__)
    data = {t: (load(ROOT / "runs/eval" / t / "summary.json"), load(ROOT / "runs/probe" / t / "summary.json"))
            for t in tags}
    width = max(10, *(len(t) + 1 for t in tags))
    print(f"{'metric':22s}" + "".join(f"{t:>{width}s}" for t in tags))
    for k in EVAL:
        print(f"{'eval/' + k:22s}"
              + "".join(f"{(e['macro'][k] * 1e3 if e else float('nan')):{width}.3f}" for e, _ in data.values()))
    for k in PROBE:
        row = []
        for _, p in data.values():
            v = p["macro"].get(k, float("nan")) if p else float("nan")
            row.append(v if k.startswith(("gain", "recovery")) else v * 1e3)
        print(f"{k:22s}" + "".join(f"{v:{width}.3f}" for v in row))
    print("(eval and sens/mae in mrad; gain and recovery are ratios)")
    if all(p for _, p in data.values()):
        variants = sorted(next(iter(data.values()))[1]["per_variant"])
        for k in ("gain/swap_views", "recovery/0.008"):
            print(f"\n{k} per variant")
            for v in variants:
                print(f"{v:22s}" + "".join(f"{p['per_variant'].get(v, {}).get(k, float('nan')):{width}.3f}"
                                           for _, p in data.values()))


if __name__ == "__main__":
    main()
