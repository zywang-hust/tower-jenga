"""Read the two reports of scripts/fleet_nsys.sh into the docs/throughput.md §8.2 table: GPU busy fraction over the
common window (forward kernels, render kernels, both, neither), the forwards (sizes, durations, the gaps between
consecutive ones and what the host was doing in them), batch close latency, render batches and stage times, the
per-request host chain of the driver's env threads, and the OS runtime calls each thread class waits in.
  .venv/bin/python sim/fleet_nsys_stats.py runs/simdev/prof/<tag>_server.nsys-rep runs/simdev/prof/<tag>_driver.nsys-rep
"""
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jenga.paths import CUDA_HOME  # noqa: E402

NSYS = str(CUDA_HOME / "bin/nsys")  # env CUDA_HOME, else nvcc on PATH


def load(rep):
    db = Path(rep).with_suffix(".sqlite")
    if not db.exists():
        subprocess.run([NSYS, "export", "--type=sqlite", "-o", str(db), rep], check=True, capture_output=True)
    c = sqlite3.connect(db)
    t0 = c.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    tabs = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kern = np.array([(s + t0, e + t0, st, pid) for s, e, st, pid in
                     c.execute("SELECT start, end, streamId, globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL")], dtype=np.int64).reshape(-1, 4)
    mem = np.array([(s + t0, e + t0) for s, e in c.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY")],
                   dtype=np.int64).reshape(-1, 2) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else np.zeros((0, 2), np.int64)
    nvtx = [(s + t0, e + t0, txt, tid) for s, e, txt, tid in
            c.execute("SELECT start, end, text, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND text IS NOT NULL")]
    osrt = []
    if "OSRT_API" in tabs:
        osrt = [(s + t0, e + t0, n, tid) for s, e, n, tid in c.execute(
            "SELECT o.start, o.end, s.value, o.globalTid FROM OSRT_API o JOIN StringIds s ON o.nameId = s.id")]
    names = {tid: n for n, tid in c.execute("SELECT s.value, t.globalTid FROM ThreadNames t JOIN StringIds s ON t.nameId = s.id")}
    return kern, mem, nvtx, osrt, names


def union(iv):
    """Total covered length of [start, end) intervals (ns)."""
    if len(iv) == 0:
        return 0
    iv = iv[np.argsort(iv[:, 0])]
    tot, s0, e0 = 0, iv[0, 0], iv[0, 1]
    for s, e in iv[1:]:
        if s > e0:
            tot += e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    return tot + e0 - s0


def coverage(iv, w0, w1, res=100_000):
    """Boolean occupancy of the window at `res` ns resolution."""
    n = (w1 - w0) // res + 1
    occ = np.zeros(n, bool)
    for s, e in iv:
        occ[max(0, (s - w0) // res):min(n, (e - w0) // res + 1)] = True
    return occ


def pct(x, q=(10, 50, 90)):
    return " / ".join(f"{v:.0f}" for v in np.percentile(x, q)) if len(x) else "-"


def main(srv, drv):
    ks, ms, ns, os_s, names_s = load(srv)
    kd, md, nd, os_d, names_d = load(drv)
    # each report over its own capture window; the joint GPU occupancy needs the windows to overlap
    # (scripts/fleet_nsys.sh NSYS_SERVER_SKEW)
    ws0, ws1 = ks[:, 0].min(), ks[:, 1].max()
    wd0, wd1 = kd[:, 0].min(), kd[:, 1].max()
    w0, w1 = max(ws0, wd0), min(ws1, wd1)
    Ws, Wd, W = (ws1 - ws0) / 1e9, (wd1 - wd0) / 1e9, (w1 - w0) / 1e9
    print(f"server window {Ws:.1f} s ({len(ks)} kernels, streams {sorted(set(ks[:, 2]))}), driver window {Wd:.1f} s "
          f"({len(kd)} kernels, streams {sorted(set(kd[:, 2]))}), overlap {W:.1f} s")
    print(f"GPU busy with forward kernels over the server window {union(ks[:, :2]) / (ws1 - ws0) * 100:.1f}% "
          f"(kernel sum {(ks[:, 1] - ks[:, 0]).sum() / 1e9:.1f} s); with render kernels over the driver window "
          f"{union(kd[:, :2]) / (wd1 - wd0) * 100:.1f}% (kernel sum {(kd[:, 1] - kd[:, 0]).sum() / 1e9:.1f} s)")
    if W > 0:
        clip = lambda a: a[(a[:, 0] >= w0) & (a[:, 1] <= w1)]  # noqa: E731
        fw = coverage(np.r_[clip(ks)[:, :2], clip(ms)], w0, w1)
        rd = coverage(np.r_[clip(kd)[:, :2], clip(md)], w0, w1)
        print(f"GPU occupancy over the overlap: forward-only {np.mean(fw & ~rd) * 100:.1f}%, render-only "
              f"{np.mean(rd & ~fw) * 100:.1f}%, both {np.mean(fw & rd) * 100:.1f}%, idle {np.mean(~fw & ~rd) * 100:.1f}%")
    W = Ws  # server-side rates below are per server window, driver-side per driver window

    def ranges(nv, pat):
        out = [(s, e, t, tid) for s, e, t, tid in nv if re.match(pat, t)]
        return sorted(out)

    fwd = ranges(ns, r"forward \d+")
    sizes = np.array([int(t.split()[1]) for _, _, t, _ in fwd])
    dur = np.array([(e - s) / 1e6 for s, e, _, _ in fwd])
    print(f"forwards: {len(fwd)} ({len(fwd) / W:.2f}/s), batch p10/50/90 {pct(sizes)} (mean {sizes.mean():.1f}, "
          f"sum {sizes.sum()} = {sizes.sum() / W:.1f} req/s), ms p10/50/90 {pct(dur)}, "
          f"ms/sample {dur.sum() / sizes.sum():.2f}")
    gaps = np.array([(fwd[i + 1][0] - fwd[i][1]) / 1e6 for i in range(len(fwd) - 1)])
    print(f"gap forward->forward: ms p10/50/90 {pct(gaps)}, sum {gaps.sum() / 1e3:.1f} s = {gaps.sum() / 1e3 / W * 100:.0f}% of window")
    # what the replica and the server did in the gaps
    rep_tid = fwd[0][3]
    by = defaultdict(float)
    for s, e, t, tid in ns:
        if tid == rep_tid and not t.startswith("forward"):
            by[t.split()[0]] += (e - s) / 1e6
    print("  replica thread, sum ms per stage over the window:", {k: round(v / len(fwd), 1) for k, v in by.items()}, "(per forward)")
    bc = ranges(ns, r"batch_close")
    rpc = ranges(ns, r"batch_rpc \d+")
    print(f"  batch_close ms p10/50/90 {pct([(e - s) / 1e6 for s, e, _, _ in bc])}; "
          f"batch_rpc minus forward ms p50 {np.median([(e - s) / 1e6 for s, e, _, _ in rpc]) - np.median(dur):.1f}")
    for stage in ("unpack", "stage", "adapter", "pack"):
        rr = ranges(ns, stage + "$")
        d = np.array([(e - s) / 1e6 for s, e, _, _ in rr])
        if len(d):
            print(f"  server {stage}: {len(d)} ({len(d) / W:.1f}/s), ms p10/50/90 {pct(d)}, sum {d.sum() / 1e3:.1f} s")
    # the render side
    W = Wd
    ren = ranges(nd, r"render \d+")
    rs = np.array([int(t.split()[1]) for _, _, t, _ in ren])
    rdur = np.array([(e - s) / 1e6 for s, e, _, _ in ren])
    print(f"renders: {len(ren)} ({len(ren) / W:.2f}/s), batch p10/50/90 {pct(rs)} (mean {rs.mean():.1f}, sum {rs.sum()}), "
          f"ms p10/50/90 {pct(rdur)}, ms/env {rdur.sum() / rs.sum():.2f}")
    for stage in ("render_close", "render_lock", "readback"):
        d = np.array([(e - s) / 1e6 for s, e, _, _ in ranges(nd, stage + "$")])
        print(f"  {stage}: ms p10/50/90 {pct(d)}, sum {d.sum() / 1e3:.1f} s")
    sub = defaultdict(float)
    for s, e, t, tid in nd:
        if t in ("transform", "raster", "shadow", "surface", "texture", "lighting", "antialias", "gbuffer", "refine", "uint8"):
            sub[t] += (e - s) / 1e6
    print("  render stages, ms per env (host-side span):", {k: round(v / rs.sum(), 2) for k, v in sub.items()})
    # per-request host chain in the env threads
    chain = {}
    for stage in ("poses", "sync", "render_wait", "pack", "ws_send", "ws_recv", "physics"):
        d = np.array([(e - s) / 1e6 for s, e, _, _ in ranges(nd, stage + "$")])
        chain[stage] = (len(d), np.median(d) if len(d) else 0, d.sum() / 1e3)
    n_req = chain["ws_recv"][0]
    print(f"env threads, per request ({n_req} requests = {n_req / W:.1f} req/s): " +
          ", ".join(f"{k} p50 {v[1]:.1f} ms" for k, v in chain.items()))
    if w1 > w0:  # forward vs render overlap per forward: fraction of each forward's span with render kernels running
        occ_r = coverage(np.r_[kd[:, :2], md], w0, w1)
        ov = [occ_r[(s - w0) // 100_000:(e - w0) // 100_000 + 1].mean() for s, e, _, _ in fwd if s >= w0 and e <= w1]
        print(f"render kernels active during a forward: fraction p10/50/90 {pct(np.array(ov) * 100)} %")
    # OSRT: where the threads wait
    for label, os_, names, (a, b) in (("server", os_s, names_s, (ws0, ws1)), ("driver", os_d, names_d, (wd0, wd1))):
        W = (b - a) / 1e9
        os_ = [r for r in os_ if r[0] >= a and r[1] <= b]
        per = defaultdict(lambda: Counter())
        for s, e, n, tid in os_:
            per[names.get(tid, str(tid))][n] += (e - s) / 1e9
        print(f"{label} OSRT, s in calls over the {W:.0f} s window, by thread name (top threads):")
        tops = sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:6]
        for tname, cnt in tops:
            print(f"  {tname[:24]:24s} " + ", ".join(f"{n} {v:.1f}" for n, v in cnt.most_common(4)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
