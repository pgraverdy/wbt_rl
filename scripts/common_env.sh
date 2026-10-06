# Shared defaults for the helper scripts under scripts/{pipeline,eval,sim2sim,viz}/ — sourced, not executed.
# Every value can be overridden from the environment before calling a script, e.g.
#   DATA_ROOT=/data/wbt GVHMR_OUT=/data/wbt/gvhmr_output_v2 bash scripts/pipeline/capture_and_retarget.sh pg_demo1 video_trimmed_1
#
# DATA_ROOT is where generated data lives. It defaults to the repository's own data/ folder; when it points elsewhere,
# the pipeline scripts symlink the data/ sub-folders that retarget.py / train.py read and write into DATA_ROOT.

_COMMON_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export WBT_RL=${WBT_RL:-$(cd "$_COMMON_ENV_DIR/.." && pwd)}            # repository root
export DATA_ROOT=${DATA_ROOT:-$WBT_RL/data}                             # generated data
export WBT_CLIPS=${WBT_CLIPS:-$DATA_ROOT/wbt_clips}                     # input videos: <clip>/<video>.mp4
export GVHMR_OUT=${GVHMR_OUT:-$DATA_ROOT/gvhmr_output}                  # GVHMR results: <clip>/<video>/<video>/hmr4d_results.pt
export GVHMR_DIR=${GVHMR_DIR:-$(cd "$WBT_RL/.." && pwd)/GVHMR}          # GVHMR checkout (checkpoints under inputs/)
export GVHMR_ENV=${GVHMR_ENV:-gvhmr_v2}                                 # micromamba env running GVHMR
export WBT_DEPS=${WBT_DEPS:-$HOME/.wbt_deps/miniconda3}                 # wbt conda root (envs wbt_rl, gmr, ...)
export HOLOSOMA_DEPS=${HOLOSOMA_DEPS:-$HOME/.holosoma_deps/miniconda3}  # holosoma conda root (envs hssim, hsmujoco, ...)
export MICROMAMBA=${MICROMAMBA:-$(command -v micromamba || echo "$HOME/.local/bin/micromamba")}
export HOLOSOMA_CUSTOM=${HOLOSOMA_CUSTOM:-$WBT_RL/modules/third_party/holosoma_custom}
export INFERENCE_CONTAINER=${INFERENCE_CONTAINER:-holosoma-inference-container}   # Docker container of holosoma_inference

# python for the analysis scripts (numpy + scipy; matplotlib optional, plots are skipped without it)
export ANALYSIS_PY=${ANALYSIS_PY:-$WBT_DEPS/envs/wbt_rl/bin/python}
# python for MuJoCo offscreen rendering and SMPL-X (mujoco, smplx, torch, PIL): the gmr env
export RENDER_PY=${RENDER_PY:-$WBT_DEPS/envs/gmr/bin/python}

# pipeline constants (the clip is stored in the SFU slot, which takes AMASS stage-II files)
export DATASET=${DATASET:-SFU} ROBOT=${ROBOT:-G1_29dof} RETARGETER=${RETARGETER:-GMR} TRAINER=${TRAINER:-holosoma_custom}

# retarget.py / train.py run `conda run -n <env>` through `sh`: wbt's own conda must come first on PATH
export PATH=$WBT_DEPS/bin:$PATH
# GMR and the GMR -> trainer bridge open MuJoCo viewers: they need an X display
export DISPLAY=${DISPLAY:-:1}

# link_into_data <target dir in DATA_ROOT> <path under WBT_RL/data>: symlink, unless DATA_ROOT is the repo's data/
link_into_data() {
  [[ "$(realpath -m "$1")" == "$(realpath -m "$2")" ]] && { mkdir -p "$1"; return 0; }
  mkdir -p "$1" "$(dirname "$2")"
  if [[ -L "$2" ]]; then ln -sfn "$1" "$2"
  elif [[ -e "$2" ]]; then echo "ERROR: $2 exists and is not a symlink" >&2; return 1
  else ln -s "$1" "$2"; fi
}
