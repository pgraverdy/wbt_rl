#!/usr/bin/env bash
# Render the Isaac Sim and MuJoCo rollouts of several clips with the capture camera, plus side-by-side videos
# aligned on the motion start.
#
# Usage: bash scripts/viz/render_all.sh <jobs file> [out dir]        (out dir defaults to $DATA_ROOT/videos)
#   jobs file: one clip per line, `#` comments allowed, fields separated by `|`:
#     name | Isaac Sim eval dir | MuJoCo eval dir | description | camera geometry json
#   relative paths are taken relative to $DATA_ROOT; the checkpoint label is read from the eval dir name (model_NNNNN).
#   The Isaac Sim eval dir needs its yaw_aligned/ re-analysis (created by sim2sim/sim2sim_compare.py).
#   Example: examples/pg_demo_render_jobs.txt
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

JOBS=${1:?usage: render_all.sh <jobs file> [out dir]}
VID=${2:-$DATA_ROOT/videos}; mkdir -p "$VID"
abs() { [[ "$1" = /* ]] && echo "$1" || echo "$DATA_ROOT/$1"; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; echo "${s%"${s##*[![:space:]]}"}"; }
render() { MUJOCO_GL=egl "$RENDER_PY" "$HERE/render_videos.py" "$@" 2>&1 | grep '"video"'; }

grep -vE '^\s*(#|$)' "$JOBS" | while IFS='|' read -r C ID MD DESC G; do
  C=$(trim "$C"); ID=$(abs "$(trim "$ID")"); MD=$(abs "$(trim "$MD")"); DESC=$(trim "$DESC"); G=$(abs "$(trim "$G")")
  CK=$(grep -oE 'model_[0-9]+' <<< "$ID" | tail -1)
  render "$ID/eval_recording.npz" "$VID/${C}_isaacsim_$CK.mp4" "${C^} ($DESC) · Isaac Sim rollout · $CK" "$G"
  render "$MD/eval_recording.npz" "$VID/${C}_mujoco_$CK.mp4" "${C^} ($DESC) · MuJoCo rollout · $CK" "$G"
  # align on the clip start (offset found by eval_analyze), keeping 2 s of lead-in in both
  ki=$(grep -oP 'recording step \K\d+' "$ID/yaw_aligned/report.txt"); km=$(grep -oP 'recording step \K\d+' "$MD/report.txt")
  si=$(awk "BEGIN{v=($ki-100)/50; print (v<0)?0:v}"); sm=$(awk "BEGIN{v=($km-100)/50; print (v<0)?0:v}")
  ffmpeg -loglevel error -y -ss "$si" -i "$VID/${C}_isaacsim_$CK.mp4" -ss "$sm" -i "$VID/${C}_mujoco_$CK.mp4" \
    -filter_complex "[0:v][1:v]hstack=inputs=2:shortest=1[v]" -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 22 \
    "$VID/${C}_isaacsim_vs_mujoco_$CK.mp4" < /dev/null && echo "side-by-side $C (isaac from ${si}s, mujoco from ${sm}s)"
done
