# Per-clip pipeline wrappers

Wrappers around `scripts/retarget.py` and `scripts/train.py` for clips captured on video. Paths and environments come
from `scripts/common_env.sh` (override any value from the environment, e.g. `DATA_ROOT=/elsewhere`).

| Script | Does |
|---|---|
| `capture_and_retarget.sh <clip> <video>` | GVHMR on `$WBT_CLIPS/<clip>/<video>.mp4` (env `gvhmr_v2`), conversion to an AMASS stage-II npz in the SFU dataset slot (`clip_to_amass.py` + the stage-II keys the unified-input step needs), then `retarget.py --retargeter GMR --sequences <video>_stageii`. Skips steps whose output is up to date; checks the final output (retarget.py can exit 0 on failure). |
| `train_latest.sh [train.py args]` | `train.py` with holosoma_custom / Isaac Sim / PPO / wandb offline / no video on the latest retarget run; checks that a checkpoint was written. |
| `run_clips_queue.sh <clip:video> …` (in `scripts/eval/`, since it also evaluates) | the full chain clip after clip. |

Requirements: envs `wbt_rl`, `gmr`, `hsretargeting`, `hssim` (`./install.sh`), the `gvhmr_v2` micromamba env and a
GVHMR checkout (`$GVHMR_DIR`), SMPL-X models (`$SMPLX_MODEL_DIR`), and an X display (`$DISPLAY`) for the MuJoCo
viewers opened by GMR and by the trainer-input bridge.

Using another retargeter (e.g. holosoma instead of GMR): [CHANGING_RETARGETER.md](CHANGING_RETARGETER.md).
