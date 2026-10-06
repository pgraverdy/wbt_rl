# MuJoCo sim-to-sim evaluation

Rolls a policy trained in Isaac Sim out in MuJoCo, with the exported ONNX running in the holosoma inference stack
exactly as on the robot, and scores it with the same metrics as the Isaac Sim evaluation. Paths and environments:
`scripts/common_env.sh`.

| Script | Does |
|---|---|
| `run_sim_eval.py` | holosoma `run_sim` (env `hsmujoco`) with `DirectSimulation.run` replaced by a loop that lowers the virtual gantry until the pelvis is at ≤ 0.80 m (the robot hangs at ~1.24 m at start and falls if released directly), releases it, records the robot at 50 Hz in the `EvalRecordingCallback` layout, and exits after a fixed simulated duration. holosoma itself is not modified. |
| `run_mujoco_eval.sh <onnx> <reference.npz> <tag> [duration]` | one unattended rollout in tmux: simulator + `run_policy.py inference:g1-29dof-wbt … --task.use-sim-time --task.interface lo` in the `holosoma-inference` Docker container; answers the policy's "Press Enter" prompt, lowers the gantry, sends `]` then `m`; then `eval_analyze.py --align-yaw`. ~40–60 s per rollout. |
| `run_mujoco_batch.sh <jobs file>` | several policies × checkpoints (`CKPTS`, default `08000 12000`); jobs format in the script and `examples/pg_demo_jobs.txt`. |
| `sim2sim_compare.py` | Isaac Sim vs MuJoCo table per (clip, checkpoint) with heading-aligned metrics for both → `$DATA_ROOT/03_eval_mujoco/comparison.md`. |

Why `--align-yaw`: MuJoCo spawns the robot facing +x and the inference policy tracks the motion relative to its own
starting heading, while the Isaac Sim evaluation starts at the reference's heading; world-frame metrics are only
comparable after rotating the recording to the reference heading.
