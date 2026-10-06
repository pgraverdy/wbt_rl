#!/usr/bin/env bash
# Video -> SMPL-X (GVHMR) -> AMASS npz -> G1 robot motion (GMR) for one clip.
#
# Usage: bash scripts/pipeline/capture_and_retarget.sh <clip> <video> [extra retarget.py args]
#   reads  $WBT_CLIPS/<clip>/<video>.mp4
#   writes $GVHMR_OUT/<clip>/<video>/…, $DATA_ROOT/00_raw_datasets/SFU/SFU/<clip>/<video>_stageii.npz,
#          $DATA_ROOT/01_retargeted_motions/SFU_G1_29dof/GMR/run_<ts>/ (holding this clip only)
# Paths and envs: scripts/common_env.sh. Steps 1–2 are skipped when their output is already up to date.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"

CLIP=${1:?usage: capture_and_retarget.sh <clip> <video> [retarget.py args]}
VIDEO=${2:?usage: capture_and_retarget.sh <clip> <video> [retarget.py args]}
shift 2

# GVHMR with torch >= 2.6: torch.load defaults to weights_only=True, which breaks its (local, trusted) checkpoints
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RAW_DIR=$DATA_ROOT/00_raw_datasets/$DATASET/$DATASET
RETARGET_DIR=$DATA_ROOT/01_retargeted_motions/${DATASET}_${ROBOT}/$RETARGETER
NPZ=$RAW_DIR/$CLIP/${VIDEO}_stageii.npz
GVHMR_RUN=$GVHMR_OUT/$CLIP/$VIDEO                         # --output_root given to demo.py
PT=$GVHMR_RUN/$VIDEO/hmr4d_results.pt                     # demo.py writes <output_root>/<video stem>/
MODEL_DIR=${SMPLX_MODEL_DIR:-$WBT_RL/data/00_raw_datasets/models/models_smplx_v1_1/models}   # contains smplx/

link_into_data "$RAW_DIR"      "$WBT_RL/data/00_raw_datasets/$DATASET/$DATASET"
link_into_data "$RETARGET_DIR" "$WBT_RL/data/01_retargeted_motions/${DATASET}_${ROBOT}/$RETARGETER"
[[ -f "$MODEL_DIR/smplx/SMPLX_NEUTRAL.npz" ]] || { echo "ERROR: SMPL-X models not found in $MODEL_DIR/smplx (set SMPLX_MODEL_DIR)" >&2; exit 1; }

# ---- 1. video -> SMPL-X (GVHMR). Run from the GVHMR repo (relative checkpoint paths). No -s: camera motion estimated.
if [[ ! -f "$PT" ]]; then
  mkdir -p "$GVHMR_RUN"
  ( cd "$GVHMR_DIR" && "$MICROMAMBA" run -n "$GVHMR_ENV" python tools/demo/demo.py \
        --video="$WBT_CLIPS/$CLIP/$VIDEO.mp4" --output_root="$GVHMR_RUN" )
  [[ -f "$PT" ]] || { echo "ERROR: GVHMR produced no $PT" >&2; exit 1; }
else
  echo "GVHMR result found, skipping step 1: $PT"
fi

# ---- 2. GVHMR .pt -> AMASS SMPL-X npz (Z-up, feet on the floor); rebuilt when missing or older than the GVHMR result
if [[ ! -f "$NPZ" || "$PT" -nt "$NPZ" ]]; then
  mkdir -p "$(dirname "$NPZ")"
  "$RENDER_PY" "$WBT_RL/modules/00_capture/gvhmr-fp-pipeline/clip_to_amass.py" --pt "$PT" --out "$NPZ" --model-dir "$MODEL_DIR"
  # add the AMASS stage-II keys the unified-input step reads (poses, num_betas, pose_jaw, pose_eye)
  "$RENDER_PY" - "$NPZ" <<'PY'
import sys, numpy as np
f = sys.argv[1]; d = dict(np.load(f, allow_pickle=True)); T = d["pose_body"].shape[0]
d["pose_jaw"] = np.zeros((T, 3), np.float32); d["pose_eye"] = np.zeros((T, 6), np.float32)
d["poses"] = np.concatenate([d["root_orient"], d["pose_body"], d["pose_jaw"], d["pose_eye"], d["pose_hand"]], 1).astype(np.float32)
d["num_betas"] = np.array(d["betas"].shape[0]); d["mocap_time_length"] = np.array(T / float(d["mocap_frame_rate"]), np.float32)
np.savez(f, **d)
PY
else
  echo "AMASS npz is up to date, skipping step 2: $NPZ"
fi

# ---- 3. retarget only this clip (--sequences), so the run train.py picks up as `latest` holds this clip only
cd "$WBT_RL"
"$WBT_DEPS/envs/wbt_rl/bin/python" scripts/retarget.py \
    --dataset "$DATASET" --robot "$ROBOT" --retargeter "$RETARGETER" --sequences "${VIDEO}_stageii" "$@"

# retarget.py can print "Done" and exit 0 after a failed sub-step: check the final output instead
OUT_NPZ=$RETARGET_DIR/latest/${VIDEO}_stageii_output_unified.npz
[[ -f "$OUT_NPZ" ]] || { echo "ERROR: retarget failed, missing $OUT_NPZ" >&2; exit 1; }
echo "OK. Results: $RETARGET_DIR/latest"
