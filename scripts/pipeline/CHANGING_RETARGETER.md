# Changing the retargeter (GMR → holosoma)

How to run the capture → retarget → train → evaluate helpers with another retargeter, what changes on disk, and
the two problems found the first time it was done (October 2026, pg_demo clips 1–4, holosoma instead of GMR).

## How

The retargeter is just an environment variable read by `scripts/common_env.sh` (default `GMR`):

```bash
P=/path/to/data          # DATA_ROOT
DATA_ROOT=$P GVHMR_OUT=$P/gvhmr_output_v2 WBT_CLIPS=~/wbt_clips RETARGETER=holosoma \
  setsid nohup bash scripts/eval/run_clips_queue.sh pg_demo1:video_trimmed_1 pg_demo2:video_trimmed_2 \
  > $P/logs/queue_holosoma.out 2>&1 < /dev/null &
```

`retarget.py` and `train.py` take it as `--retargeter`. No change to them, to `cfg/`, or to the modules is needed. The
combination must exist in `cfg/01_retargeting/<retargeter>.yaml` (`native_input_format` for the dataset) and in
`src/motion_convertor/connectors.py` (`(<retargeter output format>, holosoma_trainer_npz)`). For holosoma with SFU:
`sfu_smplx_npz → holosoma_smplx_npz → holosoma_qpos_npz → holosoma_trainer_npz`.

GVHMR capture and the AMASS conversion are skipped when their outputs are up to date, so the same `.npz` input
feeds both retargeters. The capture result is not recomputed.

## What changes on disk: nothing is overwritten

| Output | GMR | holosoma |
|---|---|---|
| retarget runs | `01_retargeted_motions/SFU_G1_29dof/GMR/run_<ts>/` | `…/HOLOSOMA/run_<ts>/` |
| policies | `02_policies/SFU_G1_29dof/gmr_holosoma_custom/run_<ts>/` | `…/holosoma_holosoma_custom/run_<ts>/` |
| Isaac Sim evals | `03_eval/SFU_G1_29dof/gmr_holosoma_custom/<ts>_…/` | `03_eval/…/holosoma_holosoma_custom/<ts>_…/` |
| MuJoCo evals | `03_eval_mujoco/…/gmr_holosoma_custom/…` | `03_eval_mujoco/…/holosoma_holosoma_custom/…` |
| per-clip logs | `logs/{run,train,stop_and_eval}_<clip>_gmr_holosoma_custom.log` | `…_<clip>_holosoma_holosoma_custom.log` |
| `latest` links | per retargeter / per policy folder | per retargeter / per policy folder |

Folder casing is set by the entry points: `retarget.py` **upper-cases** the retargeter (`GMR`, `HOLOSOMA`), while
`train.py` **lower-cases** it in the policy folder (`gmr_…`, `holosoma_…`). The scripts use `${RETARGETER^^}` and
`${RETARGETER,,}` accordingly. With GMR the two spellings coincide, which is why the mismatch only showed up with
holosoma.

The shared tables `03_eval/summary.md` (`eval/eval_summary.py`) and `03_eval_mujoco/comparison.md`
(`sim2sim/sim2sim_compare.py`) now have a **Method** column (the `<retargeter>_<trainer>` folder) and are keyed by
(clip, method, checkpoint), so both retargeters' rows coexist. Before, a holosoma evaluation would have replaced the
GMR row of the same clip.

## Problems found and fixed

1. **Folder casing**: `capture_and_retarget.sh` looked for `…/holosoma/latest` while `retarget.py` wrote
   `…/HOLOSOMA/`. Also, `run_clips_queue.sh` checked for GMR's `*_output_raw.pkl`; it now checks the
   retargeter-independent `*_output_unified.npz`.
2. **holosoma → trainer conversion crashed** (`ZeroDivisionError` in `convert_data_format_mj.py`). Upstream holosoma
   reads the npz `fps` (30, a frame rate) as a time step: `round(1 / fps)` gives 0. Since upstream commit `fb835ec`
   it then recomputes `input_dt = 1 / input_fps`. The `holosoma_custom` copy of the bridge has the same line but
   keeps `input_dt` from `--input_fps` (it only prints "input fps: 0"). `_to_trainer_input/holosoma_holosoma.py`
   now uses the `holosoma_custom` bridge, like `gmr_holosoma.py` already did. Checked on clip 1: same keys and
   shapes as the GMR trainer input, with 168 frames at 30 Hz → 279 frames at 50 Hz. The upstream bug should be
   reported to holosoma.

## Comparing the two

The retargeters differ in the reference motion itself (IK solution, foot contact handling, timing), so tracking
errors are each measured against **their own** reference (`*_stageii_trainer_input.npz` of their retarget run). For a
fair comparison, look at falls, the tracking error relative to each reference, and how close each reference is to the
human motion (`*_output_unified.npz` vs `*_input_unified.npz`). Then compare the videos.
