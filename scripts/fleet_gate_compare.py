"""Compare two sim/fleet_driver.py runs of the same jobs (A: reference, B: candidate): per episode LHPR/TSR/ODR from
the result JSON, and the policy actions from the servers' diagnostic records (jenga.record), matched to episodes by
the first observed joint state. Action differences are in mrad over the joint dimensions: the first request (same
state, same seed: only render and batch noise), and every request until the trajectories first differ by more than
the policy noise floor (docs/throughput.md 8.1: same image, other noise seed, median 0.384 / p90 11.2 / max 89.4).

  .venv/bin/python scripts/fleet_gate_compare.py <A_out_dir> <B_out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jenga.record import read_records  # noqa: E402


def trials(out: Path) -> dict:
    """episode name -> list of action arrays [k, 16] per infer, in step order. A trial file is matched to the episode
    whose logged qpos equals the joint state it observed at every request (all episodes start at the same home pose)."""
    eps = {}
    for j in out.glob("*.json"):
        q = np.load(j.with_suffix(".npz"))["qpos"]
        eps[j.stem] = np.concatenate([q[:, :7], q[:, 8:15]], 1)
    res = {}
    for rec in out.glob("servers/records_*/*/*.rec"):
        recs = sorted((r for r in read_records(rec) if r.get("type") == "infer"), key=lambda r: r["step_index"])
        if not recs:
            continue
        steps = np.array([r["step_index"] for r in recs])
        jp = np.stack([np.asarray(r["joint_positions"], float).reshape(-1) for r in recs])
        err = {n: np.abs(q[steps[steps < len(q)]] - jp[steps < len(q)]).max() for n, q in eps.items()}
        name = min(err, key=err.get)
        assert err[name] < 1e-5, (rec, name, err[name])
        res[name] = [np.asarray(r["actions"], float) for r in recs]
    return res


def main():
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    ra = {j.stem: json.loads(j.read_text()) for j in A.glob("*.json")}
    rb = {j.stem: json.loads(j.read_text()) for j in B.glob("*.json")}
    ta, tb = trials(A), trials(B)
    same = 0
    rows = []
    for n in sorted(ra):
        a, b = ra[n], rb.get(n)
        if b is None:
            rows.append(dict(ep=n, missing_in_B=True))
            continue
        out = all(abs(a[k] - b[k]) < 1e-9 for k in ("lhpr", "tsr", "odr"))
        same += out
        row = dict(ep=n, A=[a["lhpr"], a["tsr"], round(a["odr"], 3)], B=[b["lhpr"], b["tsr"], round(b["odr"], 3)],
                   same_outcome=out)
        if n in ta and n in tb:
            xa, xb = ta[n], tb[n]
            m = min(len(xa), len(xb))
            d = [np.abs(xa[i] - xb[i]).max() * 1e3 for i in range(m)]  # mrad (joint dims; gripper in mm*1e3 scale)
            div = next((i for i, v in enumerate(d) if v > 89.4), m)
            row.update(requests=[len(xa), len(xb)], first_mrad=round(d[0], 4),
                       pre_div_median_mrad=round(float(np.median(d[:max(div, 1)])), 4),
                       first_request_over_noise_max=div)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(episodes=len(ra), same_outcome=same,
                   mean_A=[float(np.mean([r[k] for r in ra.values()])) for k in ("lhpr", "tsr", "odr")],
                   mean_B=[float(np.mean([r[k] for r in rb.values()])) for k in ("lhpr", "tsr", "odr")] if rb else None)
    print("GATE_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
