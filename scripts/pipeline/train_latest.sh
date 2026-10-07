#!/usr/bin/env bash
# Whole-body-tracking RL training ($TRAINER, Isaac Sim, PPO) on the latest $RETARGETER retarget run (default GMR).
#
# Usage: bash scripts/pipeline/train_latest.sh [extra train.py args, e.g. --num-envs 2048 --checkpoint <model.pt>]
#   writes $DATA_ROOT/02_policies/SFU_G1_29dof/<retargeter lower case>_<trainer>/run_<ts>/, e.g. gmr_holosoma_custom
# Long run (~1.6 s / iteration on an RTX 5070 Ti, 30000 iterations by default): start it detached
# (setsid nohup …) and stop it at a checkpoint with scripts/eval/stop_and_eval.sh.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/common_env.sh"

export OMNI_KIT_ACCEPT_EULA=1     # Isaac Sim EULA (also set by cfg/02_training/holosoma_custom.yaml)

# train.py names the policy folder with the retargeter LOWERCASED (retarget.py keeps GMR uppercase)
POLICY_DIR=$DATA_ROOT/02_policies/${DATASET}_${ROBOT}/${RETARGETER,,}_${TRAINER}
link_into_data "$POLICY_DIR" "$WBT_RL/data/02_policies/${DATASET}_${ROBOT}/${RETARGETER,,}_${TRAINER}"

cd "$WBT_RL"
"$WBT_DEPS/envs/wbt_rl/bin/python" scripts/train.py \
    --dataset "$DATASET" --robot "$ROBOT" --retargeter "$RETARGETER" --trainer "$TRAINER" --simulator isaacsim \
    --logger-type wandb_offline --no-video "$@"

# train.py can exit 0 after a failure: require a checkpoint (latest/WholeBodyTracking/<date>-<exp>/model_*.pt)
[[ -n "$(find -L "$POLICY_DIR/latest/" -name 'model_*.pt' -print -quit 2>/dev/null)" ]] \
    || { echo "ERROR: no checkpoint found under $POLICY_DIR/latest" >&2; exit 1; }
echo "OK. Policy: $POLICY_DIR/latest"
