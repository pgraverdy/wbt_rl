# Isaac Sim evaluation

Deterministic evaluation of trained WBT checkpoints in Isaac Sim (the training simulator) and tracking metrics against
the reference motion. Paths and environments: `scripts/common_env.sh`.

| Script | Does |
|---|---|
| `eval_isaacsim.sh <ckpt.pt> <steps> <reference.npz>` | holosoma `eval_agent.py` (env `hssim`), headless, 1 env, motion forced to start at frame 0, trajectory recording on; then `eval_analyze.py`. Output: `$DATA_ROOT/03_eval/<dataset>_<robot>/gmr_holosoma_custom/<timestamp>_<clip>_<model>/`. |
| `eval_analyze.py --rec --ref --out [--align-yaw]` | aligns the recording to the reference in time (offset with the lowest joint error), matches joints by name, and reports falls/resets, min pelvis height, joint error, root drift, pelvis height error, root orientation error, body position error (relative to the root); `--align-yaw` rotates the recording about z to the reference heading (simulators that spawn at a fixed heading, e.g. MuJoCo). Writes `eval_metrics.npz` and, when matplotlib is available, `eval_tracking.png`. |
| `stop_and_eval.sh <run dir> <reference.npz> [STOP] [ITERS]` | waits for `model_<STOP>`, stops the Python trainer (SIGINT → SIGTERM → SIGKILL), evaluates the listed checkpoints with an evaluation length of reference frames + 300 steps. |
| `run_clips_queue.sh <clip:video> …` | capture → retarget → train → stop → evaluate, clip after clip, then `eval_summary.py`. |
| `eval_summary.py` | one markdown table of all evaluations → `$DATA_ROOT/03_eval/summary.md`. |

Metric definitions and how to read them: `docs` of the project report (evaluation glossary).
Plots need matplotlib in `$ANALYSIS_PY` (the default `wbt_rl` env has numpy and scipy only).
