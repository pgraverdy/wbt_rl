#!/usr/bin/env bash
# Evaluate a WBT checkpoint in Isaac Sim (holosoma eval_agent.py, headless, 1 env, motion started at frame 0),
# record the robot trajectory and compare it with the reference motion (eval_analyze.py).
#
# Usage: bash scripts/eval/eval_isaacsim.sh <checkpoint.pt> <steps> <reference trainer_input.npz>
#   <reference> must be the clip the checkpoint was trained on (<retarget run>/<seq>_trainer_input.npz);
#   <steps> at 50 Hz should cover the 2 s lead-in + the clip + the 2 s tail (stop_and_eval.sh computes it).
#   writes $DATA_ROOT/03_eval/<dataset>_<robot>/gmr_holosoma_custom/<timestamp>_<clip>_<model>/
#          (eval_recording.npz, report.txt, eval_tracking.png when matplotlib is available, eval.log, inputs.txt)
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CKPT=${1:?usage: eval_isaacsim.sh <checkpoint.pt> <steps> <reference trainer_input.npz>}
STEPS=${2:?steps}
REF=${3:?reference trainer_input.npz}
[[ -f "$REF" ]] || { echo "ERROR: reference not found: $REF" >&2; exit 1; }
export OMNI_KIT_ACCEPT_EULA=1

NAME=$(basename "$CKPT" .pt)
CLIP_TAG=$(basename "$REF" _stageii_trainer_input.npz)
OUT_DIR=$DATA_ROOT/03_eval/${DATASET}_${ROBOT}/${RETARGETER,,}_${TRAINER}/$(date +%Y%m%d_%H%M%S)_${CLIP_TAG}_$NAME
mkdir -p "$OUT_DIR"
REC=$OUT_DIR/eval_recording.npz
printf "checkpoint: %s\nreference: %s\nsimulator: isaacsim\n" "$CKPT" "$REF" > "$OUT_DIR/inputs.txt"

# Eval-callback options are bare boolean flags; experiment options take explicit True/False values.
# Headless (RTX rendering may be unavailable). Pushes are disabled in eval mode; the other start-up domain
# randomization of the training config is not overridden.
cd "$WBT_RL"
conda run -n hssim --no-capture-output bash -c "cd $WBT_RL && OMNI_KIT_ACCEPT_EULA=1 python \
    modules/02_training/holosoma_custom/eval_agent.py \
    --checkpoint=$CKPT \
    --recording.config.enabled \
    --recording.config.output_path $REC \
    --training.headless True \
    --training.max_eval_steps $STEPS \
    --command.setup_terms.motion_command.params.motion_config.eval_start_at_zero True" \
    2>&1 | tee "$OUT_DIR/eval.log"

[[ -f "$REC" ]] || { echo "ERROR: no recording at $REC" >&2; exit 1; }
"$ANALYSIS_PY" "$HERE/eval_analyze.py" --rec "$REC" --ref "$REF" --out "$OUT_DIR" | tee "$OUT_DIR/report.txt"
echo "OK. Eval output: $OUT_DIR"
