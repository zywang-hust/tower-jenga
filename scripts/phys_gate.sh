#!/usr/bin/env bash
# Correctness gate: the same 20 rollouts (2 per variant, wide layouts + DART noise) through the old per-rollout path
# (BASE worktree, sim/expert_batch.py) twice and the new persistent-worker path (sim/expert_pool.py, 4 workers so
# each runs ~5 episodes of mixed variants back to back); plus 6 rollouts with JENGA_LAYER_JITTER old vs new.
#   bash scripts/phys_gate.sh <out root> <base worktree>
set -uo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
NEW=$(cd "$(dirname "$0")/.." && pwd) OUT=$1 BASE=$2
A=(--episodes 1 --start 40 --repeats 2 --perturb wide --noise-max 0.02 --seed 11 --k 4)
J=(--variants task1-3 task3-2 task5-3 --episodes 1 --start 41 --repeats 2 --perturb wide --seed 12)
cd "$BASE" && .venv/bin/python sim/expert_batch.py "$OUT/old1" "${A[@]}" --jobs 20 > "$OUT/old1.log" 2>&1
cd "$NEW" && .venv/bin/python sim/expert_pool.py "$OUT/new" "${A[@]}" --workers 4 > "$OUT/new.log" 2>&1
cd "$BASE" && .venv/bin/python sim/expert_batch.py "$OUT/old2" "${A[@]}" --jobs 20 > "$OUT/old2.log" 2>&1
export JENGA_LAYER_JITTER=3,8,5
cd "$BASE" && .venv/bin/python sim/expert_batch.py "$OUT/jold" "${J[@]}" --jobs 6 > "$OUT/jold.log" 2>&1
cd "$NEW" && .venv/bin/python sim/expert_pool.py "$OUT/jnew" "${J[@]}" --workers 2 > "$OUT/jnew.log" 2>&1
cd "$NEW"
for p in "old1 new" "old1 old2" "jold jnew"; do set -- $p; echo "== $1 vs $2"; .venv/bin/python scripts/phys_gate.py "$OUT/$1" "$OUT/$2" | tail -25; done
echo PHYS_GATE_DONE
