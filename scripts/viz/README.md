# Visualization: robot rollouts filmed from the capture camera

| Script | Does |
|---|---|
| `camera_geometry.py <hmr4d_results.pt> <out.json> [smplx dir]` | camera pose relative to the person at frame 0, from GVHMR's camera-frame and world-frame SMPL-X estimates: horizontal/3D distance, camera height, pitch, roll, bearing from the person's facing, off-axis angle, FOV. Env `gmr`. SMPL-X models: argument, `$SMPLX_MODEL_DIR`, or `data/00_raw_datasets/models/models_smplx_v1_1/models`. |
| `render_videos.py <recording.npz> <out.mp4> <title> <geometry.json>` | replays an evaluation recording (Isaac Sim or MuJoCo, 50 Hz root pose + 29 joints) kinematically on the GMR G1 model and films it with MuJoCo's offscreen renderer from a fixed camera placed with that geometry relative to the robot's starting pose; prints the smallest margin between the robot and each image edge over the clip (negative = the robot left the frame). Env `gmr`, `MUJOCO_GL=egl`. |
| `render_all.sh <jobs file> [out dir]` | both simulators for several clips + side-by-side videos aligned on the motion start. Jobs format in the script; example `examples/pg_demo_render_jobs.txt`. |

Caveat: GVHMR estimates the camera with an assumed focal length unless the real intrinsics are given (`--f_mm`/K);
with the wrong focal length the measured camera distance scales accordingly.
