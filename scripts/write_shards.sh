#!/usr/bin/env bash
# Render solved expert rollouts into a training dataset, one writer per allocated GPU, on chosen shards of a fixed split.
#   CUDA_VISIBLE_DEVICES=<k GPUs> bash scripts/write_shards.sh <out root> <N> <shard,shard,...> <batch dir>...
# The k-th GPU of CUDA_VISIBLE_DEVICES writes shard list[k] of N for every batch dir in turn; jobs started later take the other
# shards. expert_write.py skips finished files, so a rerun resumes. WRITE_ARGS (e.g. "--frame-stride 4") is passed on.
set -euo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
cd "$(dirname "$0")/.."
OUT=$1 N=$2 SHARDS=$3; shift 3
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
IFS=, read -ra SH <<< "$SHARDS"
[ ${#GPUS[@]} -eq ${#SH[@]} ] || { echo "need one shard per GPU"; exit 1; }
pids=()
for k in "${!GPUS[@]}"; do
  (for b in "$@"; do
     CUDA_VISIBLE_DEVICES=${GPUS[$k]} runs/simdev/render_venv/bin/python sim/expert_write.py "$b" "$OUT" \
       --refine runs/simdev/refine/v2/refine.pt --shard "${SH[$k]}/$N" --workers 3 ${WRITE_ARGS:-}
   done) > "$(dirname "$1")/write_$(basename "$OUT")_${SH[$k]}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h WRITE_DONE "$(dirname "$1")"/write_"$(basename "$OUT")"_*.log
[ $rc -eq 0 ] && echo "WRITE_SHARDS_DONE $SHARDS"
