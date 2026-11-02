#!/usr/bin/env bash
# Profile one expert rollout end to end (docs/throughput.md 8.4): py-spy --native --idle over the rollout process and
# its Isaac worker, per-thread /proc CPU of the worker, wall-clock marks. LOAD extra rollouts run alongside (the
# 40-concurrent condition), each a plain sim/expert_rollout.py of a different training episode.
#   bash scripts/phys_prof.sh <out dir> <load> [driver command...]
# default driver: sim/expert_rollout.py task1-3 ep1022 with the pilot s0 offset and noise.
set -uo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
cd "$(dirname "$0")/.."
OUT=$1 LOAD=$2; shift 2
mkdir -p "$OUT"
PY=.venv/bin/python
SPY=${SPY:-$(command -v py-spy)}
DATA=${JENGA_DATA_ROOT:-data/tower}
DEMO=$DATA/data/task1-3_tower_transfer_3/episode_001022_20260909T153526Z_254a6af4.hdf5
CMD=("$@")
[ ${#CMD[@]} -eq 0 ] && CMD=($PY sim/expert_rollout.py "$DEMO" "$OUT/ep.json" --offset 0.0038 -0.0233 0.0041 --noise 0.0042 7)
{ echo "host $(hostname) cgroup $(cat /sys/fs/cgroup/cpu.max) load $(cut -d' ' -f1-3 /proc/loadavg)"; } > "$OUT/meta.txt"
pids=()
if [ "$LOAD" -gt 0 ] && [ -n "${POOL_LOAD:-}" ]; then  # load = a persistent-worker batch with LOAD workers
  $PY sim/expert_pool.py "$OUT/load" --start 20 --episodes 20 --perturb wide --noise-max 0.02 --seed 5 \
    --workers "$LOAD" > "$OUT/load.log" 2>&1 &
  pids+=($!)
  sleep "${POOL_LOAD_WARM:-60}"  # past the workers' boot: the profiled driver runs inside the steady state
elif [ "$LOAD" -gt 0 ]; then
  mapfile -t demos < <(python3 -c "
import json; s = json.load(open('runs/prep/split.json'))
eps = [r for v in s['tasks'].values() for r in v['train'][30:36]]
print('\n'.join(eps[:$LOAD]))")
  mkdir -p "$OUT/load"
  for i in "${!demos[@]}"; do
    $PY sim/expert_rollout.py "$DATA/${demos[$i]}" "$OUT/load/l$i.json" > "$OUT/load/l$i.log" 2>&1 &
    pids+=($!)
  done
  sleep 2  # the profiled rollout boots inside the pile-up, as in a batch
fi
echo "t_start $(date +%s.%N) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/meta.txt"
$SPY record --subprocesses --idle --native -r 50 -f raw -o "$OUT/spy.raw" -- "${CMD[@]}" > "$OUT/driver.log" 2>&1 &
SPYPID=$!
# per-thread CPU of every Isaac worker under the profiled driver, once a second (cumulative ticks)
(
  while kill -0 $SPYPID 2>/dev/null; do
    for p in $(for d in $(pgrep -P $SPYPID); do pgrep -P "$d"; done 2>/dev/null); do
      ts=$(date +%s.%N)
      for t in /proc/$p/task/*; do
        [ -r "$t/stat" ] && echo "$ts $p $(cat "$t/comm" 2>/dev/null | tr ' ' '_') $(awk '{print $14, $15}' "$t/stat" 2>/dev/null)"
      done
    done
    sleep 1
  done
) > "$OUT/threads.txt" 2>/dev/null &
wait $SPYPID
echo "t_end $(date +%s.%N) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/meta.txt"
for p in "${pids[@]}"; do wait "$p"; done
echo "t_load_end $(date +%s.%N)" >> "$OUT/meta.txt"
echo PHYS_PROF_DONE
