#!/usr/bin/env bash
# Full pipeline for several clips, one after another:
#   capture + AMASS + retarget (pipeline/capture_and_retarget.sh) -> train (pipeline/train_latest.sh, own session)
#   -> stop at model_STOP and evaluate 8000 and STOP in Isaac Sim (eval/stop_and_eval.sh) -> refresh the summary
# A failed clip is logged and skipped; the queue continues. Logs in $DATA_ROOT/logs/.
#
# Usage (detached): setsid nohup bash scripts/eval/run_clips_queue.sh <clip:video> [<clip:video> …] > queue.out 2>&1 &
#   e.g. … run_clips_queue.sh pg_demo3:video_trimmed_3 pg_demo4:video_trimmed_4
#   other retargeter: RETARGETER=holosoma … (outputs and logs are kept apart per retargeter; see CHANGING_RETARGETER.md)
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PIPE="$(cd "$HERE/../pipeline" && pwd)"

[[ $# -ge 1 ]] || { echo "usage: run_clips_queue.sh <clip:video> [<clip:video> …]" >&2; exit 1; }
STOP=${STOP:-12000}
POLICIES=$DATA_ROOT/02_policies/${DATASET}_${ROBOT}/${RETARGETER,,}_${TRAINER}
mkdir -p "$DATA_ROOT/logs"; LOG=$DATA_ROOT/logs/queue.log
TAG=${RETARGETER,,}_${TRAINER}                      # per-clip logs: <step>_<clip>_<TAG>.log
log() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

for item in "$@"; do
  CLIP=${item%%:*}; VIDEO=${item##*:}
  log "=== $CLIP/$VIDEO ($TAG): capture, conversion, retargeting"
  if ! bash "$PIPE/capture_and_retarget.sh" "$CLIP" "$VIDEO" > "$DATA_ROOT/logs/run_${CLIP}_$TAG.log" 2>&1; then
    log "ERROR: capture_and_retarget.sh failed for $CLIP (see logs/run_${CLIP}_$TAG.log); skipping"; continue
  fi
  RET=$(readlink -f "$DATA_ROOT/01_retargeted_motions/${DATASET}_${ROBOT}/${RETARGETER^^}/latest")
  [[ -f "$RET/${VIDEO}_stageii_output_unified.npz" ]] || { log "ERROR: latest retarget run $RET does not hold $VIDEO; skipping"; continue; }
  log "retarget run: $RET"

  log "=== $CLIP: training (stop at $STOP)"
  before=$(ls -d "$POLICIES"/run_* 2>/dev/null | sort | tail -1)
  setsid bash "$PIPE/train_latest.sh" > "$DATA_ROOT/logs/train_${CLIP}_$TAG.log" 2>&1 < /dev/null &   # own session
  TPID=$!
  RUN=""
  for _ in $(seq 1 120); do                       # wait up to 10 min for the new run folder
    cand=$(ls -d "$POLICIES"/run_* 2>/dev/null | sort | tail -1)
    [[ -n "$cand" && "$cand" != "$before" ]] && { RUN=$cand; break; }
    kill -0 $TPID 2>/dev/null || break
    sleep 5
  done
  [[ -n "$RUN" ]] || { log "ERROR: no policy run folder appeared for $CLIP (see logs/train_${CLIP}_$TAG.log); skipping"; continue; }
  log "policy run: $RUN"

  REF=$RET/${VIDEO}_stageii_trainer_input.npz
  for _ in $(seq 1 120); do [[ -f "$REF" ]] && break; sleep 5; done
  bash "$HERE/stop_and_eval.sh" "$RUN" "$REF" "$STOP" "8000 $STOP" > "$DATA_ROOT/logs/stop_and_eval_${CLIP}_$TAG.log" 2>&1
  log "stop_and_eval exit $? for $CLIP"
  wait $TPID 2>/dev/null
  "$ANALYSIS_PY" "$HERE/eval_summary.py" > /dev/null 2>&1 && log "summary refreshed: 03_eval/summary.md"
  log "=== $CLIP done"
done
log "queue finished"
