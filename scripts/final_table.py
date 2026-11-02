"""Per-policy LHPR of the final selection runs (runs/simdev/fleet/final/<policy>_<t14|t5>_r<k>/), by variant and
overall, with episode counts so incomplete runs show.
  python3 scripts/final_table.py [root]"""
import collections
import glob
import json
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "runs/simdev/fleet/final"
by = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{root}/*/*.json"):
    pol = re.match(r"(.+)_t(14|5)_r\d$", f.split("/")[-2]).group(1)
    r = json.load(open(f))
    by[pol][r["variant"].split("_")[0]].append(r["lhpr"])
variants = sorted({v for d in by.values() for v in d})
print("policy".ljust(10), "all".rjust(11), " ".join(v.replace("task", "").rjust(9) for v in variants))
for pol, d in sorted(by.items()):
    every = [x for v in d.values() for x in v]
    t14 = [x for v, xs in d.items() if not v.startswith("task5") for x in xs]
    cells = [f"{sum(d[v]) / len(d[v]):.2f}/{len(d[v]):<3}" if d[v] else "-".rjust(9) for v in variants]
    print(pol.ljust(10), f"{sum(every) / len(every):.3f}/{len(every):<4}", " ".join(c.rjust(9) for c in cells),
          f" t1-4 {sum(t14) / max(len(t14), 1):.3f}")
