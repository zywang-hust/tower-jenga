#!/bin/bash
# Closed-loop evaluation on every GPU of the machine it runs on (or on FLEET_GPUS, e.g. FLEET_GPUS="2 3"): per GPU one policy
# server (127.0.0.1:9200+g) and one sim/fleet_driver.py running <envs> episodes at a time against it, with one shared
# batched renderer. (episode, seed) pairs are dealt round-robin over the GPUs and written as
# <variant>_<episode>_s<seed>.json; pairs already done are skipped, so a rerun resumes. Servers and drivers (and their
# Isaac workers) stop on exit. Compiled server buckets persist in JAX_COMPILATION_CACHE_DIR across runs. <envs> plus
# the number of variants must stay within the server's 64 sessions (48 with the 9 test variants; 48 is also the
# fastest measured: two lockstep groups of 24).
#   fleet.sh <params_dir> <policy_id> <episodes.txt> <out_dir> <envs> <seeds> [serve_policy args] [-- fleet_driver args]
# e.g. scripts/fleet.sh runs/r2/params_40000 jenga-r2-40000 runs/simdev/rollout/val_all.txt runs/simdev/fleet/r2 48 0,1,2
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
cd "$(dirname "$(readlink -f "$0")")/.."
params=$1 policy=$2 list=$3 out=$4 envs=$5 seeds=$6; shift 6
serve=() drive=()
while (($#)); do [[ $1 == -- ]] && { shift; drive=("$@"); break; }; serve+=("$1"); shift; done
gpus=(${FLEET_GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))})
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax-jenga}
mkdir -p "$out/servers" && rm -f "$out"/servers/serve_*.log  # a stale ready line must not pass the wait below
pids=()
# Per GPU an MPS daemon (FLEET_MPS=0: none), so the server's and the renderer's kernels share the GPU instead of
# time-slicing it. Its clients see the GPU as device 0; processes without the pipe variable never see the daemon.
mps() { CUDA_MPS_PIPE_DIRECTORY=/tmp/fleet-mps-$1 CUDA_MPS_LOG_DIRECTORY=/tmp/fleet-mps-$1.log "${@:2}"; }
quit_mps() { for g in "${gpus[@]}"; do echo quit | mps $g nvidia-cuda-mps-control >/dev/null 2>&1; done; }
# children too: a server's worker processes can outlive it
trap 'kill "${pids[@]}" $(for p in "${pids[@]}"; do ps -o pid= --ppid "$p"; done) 2>/dev/null; wait
      [[ ${FLEET_MPS:-1} == 1 ]] && quit_mps' EXIT
dev() {  # dev <gpu> <cmd...> &: exec, so the pid in $! is the command's and the trap's kill reaches it
  if [[ ${FLEET_MPS:-1} == 1 ]]; then exec env CUDA_MPS_PIPE_DIRECTORY=/tmp/fleet-mps-$1 CUDA_VISIBLE_DEVICES=0 "${@:2}"
  else exec env CUDA_VISIBLE_DEVICES=$1 "${@:2}"; fi
}
if [[ ${FLEET_MPS:-1} == 1 ]]; then
  quit_mps  # a daemon left by a killed run
  for g in "${gpus[@]}"; do
    mkdir -p /tmp/fleet-mps-$g /tmp/fleet-mps-$g.log
    CUDA_VISIBLE_DEVICES=$g mps $g nvidia-cuda-mps-control -d || { echo "FLEET_FAILED MPS on GPU $g"; exit 1; }
  done
fi
groups=${FLEET_GROUPS:-2}
# Per GPU: its share of the jobs and its wave size, min(<envs>, remaining jobs): every episode of the first wave starts
# in the same round (sim/fleet_driver.py). FLEET_BALANCE=1 instead runs the jobs in equal waves, one wave whenever
# jobs + variants <= 64; measured slower for 51 jobs (docs/throughput.md 8.5: groups above 24 fragment the batches).
declare -A genvs
for i in "${!gpus[@]}"; do
  g=${gpus[i]}
  for s in ${seeds//,/ }; do awk -v s=$s 'NF {print $1, s, $2, $3, $4}' "$list"; done |  # $2-4: layout offset
    awk -v i=$i -v n=${#gpus[@]} '(NR - 1) % n == i' > "$out/servers/jobs_$g.txt"
  genvs[$g]=$envs
  if [[ ${FLEET_BALANCE:-0} == 1 ]]; then
    genvs[$g]=$(python3 - "$out/servers/jobs_$g.txt" "$out" "$envs" <<'PY'
import math, sys
from pathlib import Path
jobs, out, envs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
todo = [w for w in (l.split() for l in jobs.read_text().splitlines()) if w]
name = lambda d, s: f"{Path(d).parent.name}_{Path(d).name.split('_')[1]}_s{s}"  # sim/fleet_driver.py name()
todo = [w for w in todo if not (out / f"{name(w[0], w[1])}.json").exists()]
variants = len({Path(w[0]).parent.name for w in todo})
waves = 1 if len(todo) <= 64 - variants else math.ceil(len(todo) / envs)
print(max(1, math.ceil(len(todo) / waves)))
PY
)
  fi
done
declare -A ggroups
for g in "${gpus[@]}"; do
  # lockstep groups of at most FLEET_GROUP_MAX (24): a wave of 51 in two groups of 25/26 fragmented the server's
  # batches (mean 22, min 1) and lost 10% steady req/s against groups of 24 (docs/throughput.md 8.5)
  ggroups[$g]=$(( (genvs[$g] + ${FLEET_GROUP_MAX:-24} - 1) / ${FLEET_GROUP_MAX:-24} ))
  ((ggroups[$g] < groups)) && ggroups[$g]=$groups
  group=$(((genvs[$g] + ggroups[$g] - 1) / ggroups[$g]))
  # One OpenBLAS thread: its idle workers spin in sched_yield, about 5 cores per server. One model replica: the
  # driver's lockstep groups deliver whole batches of `group` requests, which the replica runs while the other
  # group renders; a bucket of that size keeps them unpadded.
  dev $g XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/serve_policy.py \
    --track configs/track_20261013-v1-task45-30hz.json --params "$params" --devices 0 --host 127.0.0.1 \
    --port $((9200 + g)) --policy-id "$policy" --records "$out/servers/records_$g" --buckets 1,2,4,8,16,32,$group \
    "${serve[@]}" > "$out/servers/serve_$g.log" 2>&1 &
  pids+=($!)
done
# The drivers start now: they boot their Isaac workers and load the renderer while the servers compile, and wait for
# their server before the first episode.
t0=$SECONDS drivers=()
for i in "${!gpus[@]}"; do
  g=${gpus[i]}
  dev $g runs/simdev/render_venv/bin/python sim/fleet_driver.py "$out/servers/jobs_$g.txt" "$out" \
    --url ws://127.0.0.1:$((9200 + g)) --envs "${genvs[$g]}" --groups "${ggroups[$g]}" "${drive[@]}" > "$out/servers/driver_$g.log" 2>&1 &
  pids+=($!) drivers+=($!)
done
for i in "${!gpus[@]}"; do  # ready = alive and the ready line, within 900 s (8 concurrent cold compiles are slow)
  g=${gpus[i]}
  until grep -q SERVER_READY "$out/servers/serve_$g.log"; do
    if ! kill -0 "${pids[i]}" 2>/dev/null || ((SECONDS - t0 > 900)); then
      echo "FLEET_FAILED server $g not ready after $((SECONDS - t0)) s"; tail -5 "$out/servers/serve_$g.log"; exit 1
    fi
    sleep 2
  done
done
echo "servers ready after $((SECONDS - t0)) s on GPUs ${gpus[*]} (envs/groups per GPU: $(for g in "${gpus[@]}"; do echo -n "${genvs[$g]}/${ggroups[$g]} "; done))"
wait "${drivers[@]}"
grep -h FLEET_DRIVER_DONE "$out"/servers/driver_*.log
echo "FLEET_DONE $((SECONDS - t0)) s"
