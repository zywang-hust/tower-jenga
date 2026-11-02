#!/usr/bin/env bash
# Offline evaluation, stitching and probes for one params snapshot on this machine's GPUs.
# Usage: scripts/eval_snapshot.sh <tag> <params_dir> <gpus e.g. 0,1,2,3> [extra evaluate.py/probe.py args, e.g. --prompt official]
# Tags are <run>-<step> (e.g. r1-25000, pA-27000). Shards evaluate.py over the GPUs, summarizes, runs
# stitch.py <tag> --rules none,fade4,sg9fade2, shards probe.py, summarizes. Outputs runs/eval/<tag>/ and
# runs/probe/<tag>/; shard logs runs/logs/{eval,probe}/<tag>_<i>.log. Prints SNAPSHOT_DONE <tag> on success;
# exits non-zero without it if any shard lacks its completion marker.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
if (( $# < 3 )); then
  sed -n 3p "$0" >&2
  exit 2
fi
tag=$1 params=$2
IFS=, read -ra gpus <<< "$3"
extra=("${@:4}")
n=${#gpus[@]}
mkdir -p runs/logs/eval runs/logs/probe

shards() {  # $1 = evaluate|probe, $2 = log dir, $3 = completion marker
  local i
  for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 timeout 3000 .venv/bin/python \
      "scripts/$1.py" --tag "$tag" --params "$params" --shard "$i/$n" "${extra[@]}" > "$2/${tag}_$i.log" 2>&1 &
  done
  wait
  for i in "${!gpus[@]}"; do
    if ! grep -qxF "$3 $tag $i" "$2/${tag}_$i.log"; then
      echo "$1 shard $i/$n failed: $2/${tag}_$i.log" >&2
      return 1
    fi
  done
}

shards evaluate runs/logs/eval EVAL_SHARD_DONE || exit 1
.venv/bin/python scripts/evaluate.py --tag "$tag" --summarize || exit 1
.venv/bin/python scripts/stitch.py "$tag" --rules none,fade4,sg9fade2 || exit 1
shards probe runs/logs/probe PROBE_SHARD_DONE || exit 1
.venv/bin/python scripts/probe.py --tag "$tag" --summarize || exit 1
echo "SNAPSHOT_DONE $tag"
