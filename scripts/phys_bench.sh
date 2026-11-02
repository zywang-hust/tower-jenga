#!/usr/bin/env bash
# Throughput of expert rollout generation on one pod: the same 200 jobs (10 variants x 10 episodes x 2 layouts, wide +
# DART noise) through the old path (BASE worktree sim/expert_batch.py --jobs J) and the new one (sim/expert_pool.py
# --workers W). Logs wall clock, the process tree's CPU (bash `times`: children user+sys, nvidia-smi included) and the
# pod cgroup's cpu.stat usage before/after.
#   bash scripts/phys_bench.sh <out root> <base worktree> <old|new> <J or W> [tag]
set -uo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
NEW=$(cd "$(dirname "$0")/.." && pwd) OUT=$1 BASE=$2 WHICH=$3 N=$4 TAG=${5:-$3$4}
A=(--episodes 10 --start 0 --repeats 2 --perturb ${PERTURB:-wide} --noise-max 0.02 --seed 21 --k 4)
D="$OUT/$TAG"
mkdir -p "$D"
usage() { awk '/usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat; }
u0=$(usage) t0=$(date +%s.%N)
echo "start $(date -Is) cgroup $(cat /sys/fs/cgroup/cpu.max) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$D.bench"
if [ "$WHICH" = old ]; then
  (cd "$BASE" && .venv/bin/python sim/expert_batch.py "$D" "${A[@]}" --jobs "$N" > "$D.log" 2>&1; times > "$D.times")
else
  (cd "$NEW" && .venv/bin/python sim/expert_pool.py "$D" "${A[@]}" --workers "$N" > "$D.log" 2>&1; times > "$D.times")
fi
u1=$(usage) t1=$(date +%s.%N)
python3 - "$D" "$t0" "$t1" "$u0" "$u1" >> "$D.bench" <<'EOF'
import sys, glob, json
d, t0, t1, u0, u1 = sys.argv[1], *map(float, sys.argv[2:])
n = len(glob.glob(d + "/task*.json"))
cpu = open(d + ".times").read().split("\n")[1].split()  # children: user sys as XmY.Zs
s = lambda x: float(x.split("m")[0]) * 60 + float(x.split("m")[1].rstrip("s"))
wall = t1 - t0
print(f"rollouts {n} wall_s {wall:.1f} per_hour {3600 * n / wall:.0f} tree_cpu_s {s(cpu[0]) + s(cpu[1]):.0f} "
      f"cpu_s_per_rollout {(s(cpu[0]) + s(cpu[1])) / max(n, 1):.1f} pod_cgroup_cores {(u1 - u0) / 1e6 / wall:.1f}")
EOF
tail -1 "$D.bench"
echo PHYS_BENCH_DONE
