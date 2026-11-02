"""Paired per-episode comparison of two rollout dirs on their common (episode, seed) files: mean LHPR, TSR, ODR and
reach@20 of each, the mean per-episode difference (B - A) with its standard error and a sign count, TSR discordant
pairs, and the episodes that moved most. reach@20 = share of goal operations (pushdown excluded) within 20 mm of
the goal at some point. runs/simdev/render_venv/bin/python sim/paired.py <dir A> <dir B>"""
import glob
import json
import sys
from pathlib import Path
import numpy as np

A, B = (Path(d) for d in sys.argv[1:3])
load = lambda d: {Path(p).name: json.load(open(p)) for p in glob.glob(f"{d}/*.json")}  # noqa: E731
ra, rb = load(A), load(B)
common = sorted(set(ra) & set(rb))
reach = lambda r: (np.mean(g) if (g := [x <= 0.02 for x in r["reach_m"] if x is not None]) else np.nan)  # noqa: E731
print(f"A {A.name}: {len(ra)}  B {B.name}: {len(rb)}  common {len(common)}")
for k, f in (("lhpr", lambda r: r["lhpr"]), ("tsr", lambda r: float(r["tsr"])), ("odr", lambda r: float(r["odr"])),
             ("reach@20", reach)):
    a, b = (np.array([f(r[n]) for n in common], float) for r in (ra, rb))
    ok = ~(np.isnan(a) | np.isnan(b))
    d = b[ok] - a[ok]
    print(f"{k:9s} A {np.mean(a[ok]):.3f}  B {np.mean(b[ok]):.3f}  B-A {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE, n={len(d)})"
          f"  B better {int((d > 0).sum())}, worse {int((d < 0).sum())}, tied {int((d == 0).sum())}")
d = sorted(((rb[n]["lhpr"] - ra[n]["lhpr"], n) for n in common))
print("largest drops:", ", ".join(f"{n[:-5]} {x:+.2f}" for x, n in d[:5] if x < 0))
print("largest gains:", ", ".join(f"{n[:-5]} {x:+.2f}" for x, n in d[::-1][:5] if x > 0))

# per variant: mean LHPR and reach@20 of A and B, paired LHPR difference
from collections import defaultdict  # noqa: E402
by = defaultdict(list)
for n in common:
    by[ra[n]["variant"]].append(n)
print(f"{'variant':32s} {'n':>3s} {'LHPR A':>7s} {'LHPR B':>7s} {'B-A':>14s} {'TSR A/B':>8s} {'reach A':>8s} {'reach B':>8s}")
for v, ns in sorted(by.items()):
    a, b = np.array([ra[n]["lhpr"] for n in ns]), np.array([rb[n]["lhpr"] for n in ns])
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    ga = np.nanmean([reach(ra[n]) for n in ns]) if any(not np.isnan(reach(ra[n])) for n in ns) else np.nan
    gb = np.nanmean([reach(rb[n]) for n in ns]) if any(not np.isnan(reach(rb[n])) for n in ns) else np.nan
    print(f"{v:32s} {len(ns):3d} {a.mean():7.2f} {b.mean():7.2f} {d.mean():+7.2f} ±{se:5.2f} "
          f"{sum(ra[n]['tsr'] for n in ns):3.0f}/{sum(rb[n]['tsr'] for n in ns):<3.0f} {ga:8.2f} {gb:8.2f}")
