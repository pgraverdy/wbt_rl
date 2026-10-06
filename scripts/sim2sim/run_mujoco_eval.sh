#!/usr/bin/env bash
# One MuJoCo sim-to-sim rollout of a WBT policy, fully scripted, then the standard metrics.
#
# Usage: bash scripts/sim2sim/run_mujoco_eval.sh <model.onnx> <reference trainer_input.npz> <tag> [duration_s]
#   writes $DATA_ROOT/03_eval_mujoco/<dataset>_<robot>/gmr_holosoma_custom/<timestamp>_<tag>/
#
#   tmux session "mjeval":
#     window "sim": holosoma run_sim (env hsmujoco) through run_sim_eval.py (gantry lowering/release + recorder)
#     window "pol": run_policy.py in the holosoma-inference Docker container (DDS over lo, simulator clock)
#   sequence: sim ready -> policy "Press Enter" (stiff hold) -> lower gantry -> ']' (start policy) -> release
#             -> 'm' (start motion) -> recorder stops after <duration> s -> eval_analyze.py --align-yaw
# Requirements: env hsmujoco, Docker image/container holosoma-inference (src/holosoma_inference/docker/), tmux,
# an X display. The MuJoCo window opens on $DISPLAY during the run.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ONNX=${1:?model.onnx}; REF=${2:?reference trainer_input.npz}; TAG=${3:?tag}; DUR=${4:-16}
HS=$HOLOSOMA_CUSTOM
OUT=$DATA_ROOT/03_eval_mujoco/${DATASET}_${ROBOT}/${RETARGETER,,}_${TRAINER}/$(date +%Y%m%d_%H%M%S)_$TAG
REL=/tmp/mujoco_eval_release_$$
CONTAINER=$INFERENCE_CONTAINER
SESSION=mjeval
mkdir -p "$OUT"; rm -f "$REL" "$REL.lower"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/driver.log"; }
cleanup() { # always: stop the policy in the container, the simulator and the tmux session
  docker exec "$CONTAINER" pkill -INT -f run_policy.py >/dev/null 2>&1
  tmux kill-session -t "$SESSION" 2>/dev/null
  pkill -f "run_sim_eval.py.*$REL" 2>/dev/null
  rm -f "$REL" "$REL.lower"
}
trap cleanup EXIT
wait_for() { # wait_for <file> <regex> <timeout_s>
  for _ in $(seq 1 "$3"); do grep -qaE "$2" "$1" 2>/dev/null && return 0; sleep 1; done; return 1; }

# the policy must be visible inside the container: copy it under the mounted repo (ignored: *.onnx)
mkdir -p "$HS/eval_models"; cp "$ONNX" "$HS/eval_models/$TAG.onnx"
printf "onnx: %s\nreference: %s\nsimulator: mujoco (run_sim)\n" "$ONNX" "$REF" > "$OUT/inputs.txt"

# bind-mount sources the container expects on the host (see src/holosoma_inference/docker/run.sh)
mkdir -p "$HOME/cyclonedds_ws"; touch "$HOME/.holosoma_deps/bash_history"
docker start "$CONTAINER" >/dev/null && log "container $CONTAINER running"

tmux kill-session -t "$SESSION" 2>/dev/null
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
xhost +local: >/dev/null 2>&1
tmux new-session -d -s "$SESSION" -n sim -x 220 -y 50
tmux pipe-pane -t "$SESSION:sim" -o "cat >> $OUT/sim.log"
tmux send-keys -t "$SESSION:sim" "export DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY; cd $HS && source scripts/source_mujoco_setup.sh && python $HERE/run_sim_eval.py robot:g1-29dof --eval-out $OUT/eval_recording.npz --eval-duration $DUR --eval-release-file $REL" Enter
log "sim starting"
wait_for "$OUT/sim.log" "waiting for $REL.lower" 180 || { log "ERROR: simulator not ready (see sim.log)"; exit 1; }
log "sim ready"

tmux new-window -t "$SESSION" -n pol
tmux pipe-pane -t "$SESSION:pol" -o "cat >> $OUT/policy.log"
tmux send-keys -t "$SESSION:pol" "docker exec -it $CONTAINER bash -c 'source /opt/ros/humble/setup.bash && cd /workspace/holosoma && python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt --task.model-path /workspace/holosoma/eval_models/$TAG.onnx --task.no-use-joystick --task.use-sim-time --task.rl-rate 50 --task.interface lo'" Enter
log "policy starting"
wait_for "$OUT/policy.log" "Press Enter to continue" 120 || { log "ERROR: policy did not reach the stiff-hold prompt (see policy.log)"; exit 1; }
log "policy asks 'Press Enter to continue' (stiff hold); sending Enter"; tmux send-keys -t "$SESSION:pol" Enter
wait_for "$OUT/policy.log" "Policy initialized successfully" 120 || { log "ERROR: policy not ready (see policy.log)"; exit 1; }
sleep 2; log "policy ready; lowering gantry"; touch "$REL.lower"
wait_for "$OUT/sim.log" "\[eval\] lowered" 120 || { log "ERROR: gantry lowering did not finish (see sim.log)"; exit 1; }
log "sending ']' (start policy)"; tmux send-keys -t "$SESSION:pol" "]"
sleep 2
log "releasing gantry"; touch "$REL"
wait_for "$OUT/sim.log" "\[eval\] gantry released" 30 || { log "ERROR: release not confirmed"; exit 1; }
sleep 1
log "sending 'm' (start motion)"; tmux send-keys -t "$SESSION:pol" "m"

wait_for "$OUT/sim.log" "\[eval\] done" $(( ${DUR%.*} * 3 + 60 )) || log "WARNING: recorder did not report done"
log "stopping policy"; tmux send-keys -t "$SESSION:pol" C-c; sleep 3
cleanup

if [[ -f "$OUT/eval_recording.npz" ]]; then
  # MuJoCo spawns the robot at a fixed heading: compare in the reference heading (--align-yaw)
  "$ANALYSIS_PY" "$WBT_RL/scripts/eval/eval_analyze.py" --rec "$OUT/eval_recording.npz" --ref "$REF" --out "$OUT" --align-yaw | tee "$OUT/report.txt"
  log "OK: $OUT"
else
  log "ERROR: no recording"; exit 1
fi
