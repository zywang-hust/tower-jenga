#!/bin/bash
# Whole-closed-loop nsys timeline of one GPU of the fleet (docs/throughput.md §8.2): the scripts/fleet.sh configuration
# (MPS daemon, policy server with its replica process, sim/fleet_driver.py with <envs> envs) on <gpu>, every process
# under `nsys launch`; collection runs from <delay> s after the driver starts for <dur> s, so it spans a steady slice
# of the first wave of episodes (one job per env). Reports: runs/simdev/prof/<tag>_{server,driver}.nsys-rep; the
# server report holds the replica process too (--trace-fork-before-exec). Isaac workers are not traced.
#   scripts/fleet_nsys.sh <gpu> <tag> <envs> <seconds> <delay> <dur> [serve_policy args]
# The server's --delay runs from its replica's exec, about 40 s into the model load, not from its launch (fleet48s: its
# window opened 43 s after the driver's), so the server's delay is NSYS_SERVER_SKEW (default 43) s shorter. Check the
# overlap sim/fleet_nsys_stats.py prints: the per-process figures hold either way, the joint GPU occupancy needs it.
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
gpu=$1 tag=$2 envs=$3 seconds=$4 delay=$5 dur=$6; shift 6
NSYS=${NSYS:-$(command -v nsys || echo "${CUDA_HOME:-/usr/local/cuda}/bin/nsys")}
scratch=${TMPDIR:-/tmp}/jenga_prof2/$tag
rm -rf "$scratch"; mkdir -p "$scratch/servers" runs/simdev/prof
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax-jenga}
pipe=/tmp/prof2-mps-$gpu
pids=()
mine() {  # every process of this run: server, replica and estimator pool, driver and its Isaac workers (nsys launch
  # puts them two levels down, so parent pids are not enough); all carry the scratch dir, the port or the pipe
  local p
  for p in $(ps -eo pid=,args= | awk -v a="$scratch" -v b="port $port" -v c="$pipe" '(index($0, a) || index($0, b) || index($0, c)) && $2 != "awk" {print $1}'); do
    echo "$p"; ps -o pid= --ppid "$p"  # the Isaac workers name only their demo and socket
  done
}
cleanup() {
  kill "${pids[@]}" $(mine) 2>/dev/null; sleep 3; kill -9 $(mine) 2>/dev/null; wait
  echo quit | CUDA_MPS_PIPE_DIRECTORY=$pipe nvidia-cuda-mps-control >/dev/null 2>&1
}
trap cleanup EXIT
echo quit | CUDA_MPS_PIPE_DIRECTORY=$pipe nvidia-cuda-mps-control >/dev/null 2>&1
mkdir -p $pipe $pipe.log
CUDA_VISIBLE_DEVICES=$gpu CUDA_MPS_PIPE_DIRECTORY=$pipe CUDA_MPS_LOG_DIRECTORY=$pipe.log nvidia-cuda-mps-control -d || { echo PROF_FAILED mps; exit 1; }
dev() { env CUDA_MPS_PIPE_DIRECTORY=$pipe CUDA_VISIBLE_DEVICES=0 "$@"; }
groups=${FLEET_GROUPS:-2} group=$(((envs + ${FLEET_GROUPS:-2} - 1) / ${FLEET_GROUPS:-2}))
port=9403
# Both captures are `profile --delay/--duration` (the launch/start/stop session mode wrote no report on our machines:
# `nsys stop` dies in a boost file_size error). The driver starts a fixed `lead` s after the server, so the two
# delays line up; the server must be ready by then (warm compile cache: 40 s unloaded, 111 s under a saturated CPU).
lead=${NSYS_LEAD:-180}
prof="$NSYS profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --kill=none -f true"
dev XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OPENBLAS_NUM_THREADS=1 \
  $prof --trace-fork-before-exec=true --delay=$((lead + delay - ${NSYS_SERVER_SKEW:-43})) --duration=$dur -o runs/simdev/prof/${tag}_server \
  .venv/bin/python scripts/serve_policy.py --track configs/track_20261013-v1-task45-30hz.json --params runs/r2/params_40000 \
  --devices 0 --host 127.0.0.1 --port $port --policy-id prof --records "$scratch/servers/records" \
  --buckets 1,2,4,8,16,32,$group "$@" > "$scratch/servers/serve.log" 2>&1 &
pids+=($!)
t0=$SECONDS
until grep -q SERVER_READY "$scratch/servers/serve.log"; do
  if ! kill -0 "${pids[0]}" 2>/dev/null || ((SECONDS - t0 > lead - 5)); then echo "PROF_FAILED server not ready in $lead s"; tail -5 "$scratch/servers/serve.log"; exit 1; fi
  sleep 2
done
echo "server ready after $((SECONDS - t0)) s"
sleep $((lead - (SECONDS - t0)))
awk 'NF {print $1, 0}' runs/simdev/rollout/val_all.txt | head -n "$envs" > "$scratch/jobs.txt"
dev $prof --delay=$delay --duration=$dur -o runs/simdev/prof/${tag}_driver \
  runs/simdev/render_venv/bin/python sim/fleet_driver.py "$scratch/jobs.txt" "$scratch" --url ws://127.0.0.1:$port \
  --envs "$envs" --groups "$groups" --seconds "$seconds" > "$scratch/servers/driver.log" 2>&1 &
pids+=($!)
echo "driver started at $((SECONDS - t0)) s; capture $delay s after it for $dur s"
sleep $((delay + dur + 10))
t1=$SECONDS  # nsys writes the reports when the windows end and prints "Generated:" into each log; give it 15 min
until { grep -q "^Generated:" "$scratch/servers/serve.log" && grep -q "^Generated:" "$scratch/servers/driver.log"; } || ((SECONDS - t1 > 900)); do sleep 5; done
grep -h "req/s" "$scratch/servers/driver.log"
grep -h "replica" "$scratch/servers/serve.log" | tail -4
ls -la runs/simdev/prof/${tag}_server.nsys-rep runs/simdev/prof/${tag}_driver.nsys-rep || { echo PROF_FAILED reports; exit 1; }
echo PROF_DONE
