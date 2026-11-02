"""Time breakdown of a profiled expert rollout (scripts/phys_prof.sh): py-spy raw stacks (--idle, so the main thread's
samples are its wall clock) of the driver and each Isaac worker, classified by the first matching rule; per-thread CPU
from threads.txt; Kit log marks.
  python3 scripts/phys_prof_stats.py <prof dir> [--rate 50]
"""
import argparse
import collections
import re
from pathlib import Path

# (label, all of these substrings in the stack); first match wins. Python frames read "func (path:line)".
WORKER = [
    ("ipc: wait for driver", ["recv_bytes ("]),
    ("ipc: wait for driver connect", ["accept ("]),
    ("PhysX CUDA probe at episode release (gpu.foundation)", ["gpu.foundation", "release ("]),
    ("episode switch: release (stop, close stage)", ["release ("]),
    ("PhysX CUDA probe at shutdown (gpu.foundation)", ["gpu.foundation", "close ("]),
    ("PhysX CUDA probe at stage open (gpu.foundation)", ["gpu.foundation", "scene.py"]),
    ("app start: resourcemonitor (platforminfo wait4)", ["resourcemonitor"]),
    ("PhysX CUDA probe at app start (gpu.foundation)", ["gpu.foundation", "simulation_app.py"]),
    ("PhysX CUDA probe (other)", ["gpu.foundation"]),
    ("episode: physics tick", ["tick (", "scene.py"]),
    ("episode: state readback", ["state (", "physics_worker.py"]),
    ("episode: reset/init", ["reset (", "scene.py"]),
    ("shutdown", ["close (", "simulation_app.py"]),
    ("stage load (open_stage)", ["openStage"]),
    ("stage: physics init + play", ["__init__ (", "scene.py"]),
    ("app start: extension load", ["_start_app ("]),
    ("app start: other (new stage, update)", ["__init__ (", "simulation_app.py"]),
    ("import/bootstrap/setup", []),
]
DRIVER = [
    ("wait for worker boot (connect)", ["_connect ("]),
    ("wait for worker boot (connect)", ["connect (", "rollout.py"]),
    ("episode setup (EPISODE: reopen stage)", ["episode (", "physics_client.py"]),
    ("episode: wait for physics (ipc)", ["step (", "physics_client.py"]),
    ("episode: wait for physics (ipc)", ["step (", "rollout.py"]),
    ("episode: expert plan (act)", ["act (", "expert.py", "run ("]),
    ("post: active-block labels (held)", ["held (", "run ("]),
    ("post: score/save", ["run ("]),
    ("episode: other", ["run ("]),
    ("wait for worker shutdown (close)", ["close (", "rollout.py"]),
    ("save npz/json", ["savez"]),
    ("import", ["<module> ("]),
    ("other", []),
]


def classify(rules, stack):
    for label, keys in rules:
        if all(k in stack for k in keys):
            return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--rate", type=float, default=50)
    args = ap.parse_args()
    out = collections.defaultdict(collections.Counter)
    for line in (args.dir / "spy.raw").open():
        stack, n = line.rstrip("\n").rsplit(" ", 1)
        fr = stack.split(";")
        procs = [f for f in fr if f.startswith("process ")]
        main_py = "physics_worker.py" if any("physics_worker.py" in p for p in procs) else None
        frames = ";".join(f for f in fr if not f.startswith("process "))
        if main_py is None:  # driver: the main thread has the driver script's frames
            if not re.search(r"[a-z_]+\.py:\d+\)", frames):
                continue
            proc = "driver " + procs[-1].split(":")[0].split()[-1]
            rules = DRIVER
        else:
            if "physics_worker.py:" not in frames:
                continue  # Kit's other threads
            proc = "worker " + procs[-1].split(":")[0].split()[-1]
            rules = WORKER
        out[proc][classify(rules, frames)] += int(n)
    for proc, c in out.items():
        tot = sum(c.values())
        print(f"\n## {proc}: {tot / args.rate:.1f} s main-thread wall")
        print("| item | s | % |\n|---|---|---|")
        for k, v in c.most_common():
            print(f"| {k} | {v / args.rate:.2f} | {100 * v / tot:.1f} |")
    th = args.dir / "threads.txt"
    if th.exists() and th.stat().st_size:
        last = {}
        for line in th.open():
            p = line.split()
            if len(p) < 5:
                continue
            key = (p[1], p[2])
            # tids are not logged; sum ticks over same-named threads per sample time
            last.setdefault(key, {})[p[0]] = last.get(key, {}).get(p[0], 0) + int(p[3]) + int(p[4])
        print("\n## worker CPU by thread name (s, utime+stime at the last sample)")
        tot = collections.Counter()
        for (pid, comm), series in last.items():
            tot[comm] += max(series.values()) / 100
        for comm, s in tot.most_common(10):
            print(f"| {comm} | {s:.2f} |")
        print(f"| total | {sum(tot.values()):.2f} |")
    for log in sorted(args.dir.glob("*.physics.log")):
        print(f"\n## Kit marks {log.name}")
        for line in log.open(errors="replace"):
            m = re.search(r"\[([\d,]+)ms\]|^\[([\d.]+)s\]", line)
            if m and re.search(r"app ready|App Starting|No CUDA devices|createGpuFoundation|Unable to create", line):
                t = float(m.group(1).replace(",", "")) / 1000 if m.group(1) else float(m.group(2))
                print(f"{t:7.2f}  {line.strip()[-90:]}")
            elif "PHYSICS_WORKER_DONE" in line or "EPISODE" in line:
                print(f"         {line.strip()[:90]}")


if __name__ == "__main__":
    main()
