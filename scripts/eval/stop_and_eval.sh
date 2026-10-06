#!/usr/bin/env bash
# Companion to scripts/pipeline/train_latest.sh: wait until a training run writes model_<STOP>, stop the trainer,
# then evaluate the requested checkpoints in Isaac Sim against the clip's reference.
#
# Usage (detached): setsid nohup bash scripts/eval/stop_and_eval.sh <policy_run_dir> <reference trainer_input.npz> \
#                       [STOP=12000] [EVAL_ITERS="8000 12000"] > stop_and_eval.log 2>&1 &
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_DIR=${1:?policy run dir}
REF=${2:?reference trainer_input.npz}
STOP=${3:-12000}
EVAL_ITERS=${4:-"8000 $STOP"}
log() { echo "[$(date +%F_%T)] $*"; }
# the python trainer itself — not the `sh -c conda run …` wrappers whose command lines also contain its name
trainer_pids() { pgrep -f '^python .*train_agent\.py exp:'; }

stop_file=$(printf "model_%05d.onnx" "$STOP")      # the .onnx is written right after the .pt
log "waiting for $stop_file in $RUN_DIR"
until [[ -n "$(find "$RUN_DIR" -name "$stop_file" -print -quit)" ]]; do
  [[ -n "$(trainer_pids)" ]] || { log "ERROR: training is no longer running and $stop_file never appeared"; exit 1; }
  sleep 30
done
sleep 10
log "$stop_file written; stopping training"
pids=$(trainer_pids)
if [[ -n "$pids" ]]; then
  kill -INT $pids 2>/dev/null
  for _ in $(seq 1 12); do sleep 5; [[ -z "$(trainer_pids)" ]] && break; done
  if [[ -n "$(trainer_pids)" ]]; then log "trainer ignored SIGINT for 60 s; sending SIGTERM"; kill -TERM $(trainer_pids) 2>/dev/null; sleep 20; fi
  if [[ -n "$(trainer_pids)" ]]; then log "trainer still alive; sending SIGKILL"; kill -KILL $(trainer_pids) 2>/dev/null; sleep 5; fi
fi
[[ -n "$(trainer_pids)" ]] && log "WARNING: trainer still running" || log "training stopped"

# eval length: reference frames + 300 (2 s lead-in + 2 s tail added by the trainer, + margin), at least 600
T_REF=$("$ANALYSIS_PY" -c "import zipfile,io,numpy as np; z=zipfile.ZipFile('$REF'); print(np.load(io.BytesIO(z.read('joint_pos.npy'))).shape[0])" 2>/dev/null) \
    || { log "ERROR: cannot read $REF"; exit 1; }
STEPS=$(( T_REF + 300 )); (( STEPS < 600 )) && STEPS=600
log "reference has $T_REF frames -> $STEPS eval steps"
for it in $EVAL_ITERS; do
  # the trainer's own checkpoint, not wandb's copy under .wandb/
  ckpt=$(find "$RUN_DIR" -path '*/.wandb' -prune -o -name "$(printf "model_%05d.pt" "$it")" -print -quit)
  [[ -n "$ckpt" ]] || { log "ERROR: no checkpoint for iteration $it"; continue; }
  log "evaluating $ckpt"
  bash "$HERE/eval_isaacsim.sh" "$ckpt" "$STEPS" "$REF"
  log "eval of iteration $it finished (exit $?)"
done
log "done"
