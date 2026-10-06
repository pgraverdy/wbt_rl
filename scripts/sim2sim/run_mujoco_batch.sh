#!/usr/bin/env bash
# MuJoCo sim-to-sim rollouts for several policies and checkpoints, one after another (one rollout each).
#
# Usage: bash scripts/sim2sim/run_mujoco_batch.sh <jobs file>      (CKPTS="08000 12000" by default)
#   jobs file: one job per line, `#` comments allowed, fields separated by `|`:
#     tag | policy run dir | reference trainer_input.npz | duration_s
#   relative paths are taken relative to $DATA_ROOT. Example: examples/pg_demo_jobs.txt
#   duration ≈ 2 s (gantry/start) + 2 s (lead-in) + clip length + ~4 s margin.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

JOBS=${1:?usage: run_mujoco_batch.sh <jobs file>}
CKPTS=${CKPTS:-"08000 12000"}
mkdir -p "$DATA_ROOT/logs"; LOG=$DATA_ROOT/logs/mujoco_batch.log
abs() { [[ "$1" = /* ]] && echo "$1" || echo "$DATA_ROOT/$1"; }

grep -vE '^\s*(#|$)' "$JOBS" | while IFS='|' read -r TAG RUN REF DUR; do
  TAG=$(echo $TAG); RUN=$(abs "$(echo $RUN)"); REF=$(abs "$(echo $REF)"); DUR=$(echo $DUR)
  for CK in $CKPTS; do
    NAME=${TAG}_model_${CK}
    # the trainer's own export (not wandb's copy, not the evaluation's re-export)
    ONNX=$(find "$RUN" \( -path '*/.wandb' -o -path '*/exported' \) -prune -o -name "model_${CK}.onnx" -print -quit)
    [[ -n "$ONNX" ]] || { echo "[$(date +%T)] $NAME: no model_${CK}.onnx under $RUN" | tee -a "$LOG"; continue; }
    echo "[$(date +%T)] === $NAME ($DUR s) onnx=$ONNX" | tee -a "$LOG"
    bash "$HERE/run_mujoco_eval.sh" "$ONNX" "$REF" "$NAME" "$DUR" >> "$LOG" 2>&1 < /dev/null
    echo "[$(date +%T)] $NAME exit $?" | tee -a "$LOG"
  done
done
echo "[$(date +%T)] batch finished" | tee -a "$LOG"
