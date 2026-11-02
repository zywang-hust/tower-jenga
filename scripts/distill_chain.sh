#!/usr/bin/env bash
# One job on the GPUs in CUDA_VISIBLE_DEVICES: render the solved expert rollouts of BATCH into OUT (one writer per
# GPU), then train a policy from pi05_base on them (expert-only for the variants they cover).
#   CUDA_VISIBLE_DEVICES=0,...,7 bash scripts/distill_chain.sh <batch dir s*> <out root> <tag> <steps> [EXTRA]
# EXTRA (optional): expert_batch.py arguments for a second batch <batch dir>/../s1 run on this machine's CPUs while the
# first batch finishes and is written; it is written too before training.
set -euo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
cd "$(dirname "$0")/.."
BATCH=$1 OUT=$2 TAG=$3 STEPS=$4 EXTRA=${5:-}
LOG=$(dirname "$BATCH")
if [ -n "$EXTRA" ]; then
  .venv/bin/python sim/expert_batch.py "$LOG/s1" $EXTRA > "$LOG/batch_s1.log" 2>&1 &
fi
wait_batch() {
  for i in $(seq 540); do  # at most 90 min
    grep -q EXPERT_BATCH_DONE "$LOG/batch_$1.log" 2>/dev/null && return 0
    sleep 10
  done
  echo "batch $1 not done"; exit 1
}
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
write() {
  local pids=()
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} runs/simdev/render_venv/bin/python sim/expert_write.py "$LOG/$1" "$OUT" \
      --refine runs/simdev/refine/v2/refine.pt --shard "$k/$N" --workers 3 > "$LOG/write_${TAG}_$1_$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  grep -h WRITE_DONE "$LOG"/write_"${TAG}"_"$1"_*.log
}
wait_batch "$(basename "$BATCH")"; write "$(basename "$BATCH")"
if [ -n "$EXTRA" ]; then wait_batch s1; write s1; fi
echo "WRITE_ALL_DONE $(find "$OUT" -name '*.hdf5' | wc -l) files"
.venv/bin/python scripts/train.py --tag "$TAG" --steps "$STEPS" --batch 64 --lr 2.5e-5 --workers 16 \
  --synthetic-root "$OUT" --synthetic-frac 1.0 --export-interval 2000 --save-interval 0
