#!/bin/bash
# Closed-loop robustness suite after the official robustness release cases (docs/research.md §2.4, §8), on the
# held-out val episodes of tasks 1-4 (2 per variant, runs/simdev/rollout/ood/*.txt), one fleet run per case on the
# GPUs in CUDA_VISIBLE_DEVICES, results in <out root>/<case>/. Cases:
#   shift_yaw10  assembly shift <= 15 mm per axis, yaw 5-10 deg (training covers +-5 deg)
#   layers       irregular layers 8 mm / 5 deg (JENGA_LAYER_JITTER), no shift
#   combined     shift_yaw10 offsets plus irregular layers
#   mass04 mass20 friction015 friction13   release endpoints, no shift
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ood_suite.sh <params_dir> <policy_id> <out root> [case ...]
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
cd "$(dirname "$(readlink -f "$0")")/.."
params=$1 policy=$2 root=$3; shift 3
cases=("$@"); ((${#cases[@]})) || cases=(shift_yaw10 layers combined mass04 mass20 friction015 friction13)
L=runs/simdev/rollout/ood
export FLEET_GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
for c in "${cases[@]}"; do
  case $c in
    shift_yaw10) env=() list=$L/shift_yaw10.txt ;;
    layers) env=(JENGA_LAYER_JITTER=1031,8,5) list=$L/nominal.txt ;;
    combined) env=(JENGA_LAYER_JITTER=1031,8,5) list=$L/shift_yaw10.txt ;;
    mass04) env=(JENGA_MASS_SCALE=0.4) list=$L/nominal.txt ;;
    mass20) env=(JENGA_MASS_SCALE=2.0) list=$L/nominal.txt ;;
    friction015) env=(JENGA_FRICTION=0.15,0.15) list=$L/nominal.txt ;;
    friction13) env=(JENGA_FRICTION=1.3,1.3) list=$L/nominal.txt ;;
    *) echo "unknown case $c"; exit 1 ;;
  esac
  env "${env[@]}" bash scripts/fleet.sh "$params" "$policy" "$list" "$root/$c" 48 0 > "$root.$c.log" 2>&1
  echo "OOD_CASE_DONE $c rc=$?"
done
echo OOD_SUITE_DONE
