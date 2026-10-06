#!/usr/bin/env python3
"""Render evaluation recordings (Isaac Sim or MuJoCo rollouts) as mp4 with MuJoCo's offscreen renderer.

The robot is replayed kinematically from the recording (root pose + 29 joint angles at 50 Hz) on the G1 model, and filmed
by a fixed camera placed relative to the robot's starting pose with the geometry measured on the real capture
(camera_geometry.py output): horizontal distance, height above the floor, upward tilt,
robot offset from the optical axis, and the same field of view.

Usage (env gmr):  MUJOCO_GL=egl python scripts/viz/render_videos.py <recording.npz> <out.mp4> <title> <camera_geometry.json>
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

REC, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
GEOM = json.load(open(sys.argv[4]))   # per-clip camera geometry (scripts/viz/camera_geometry.py)
XML = str(Path(__file__).resolve().parents[2] / "modules/01_retargeting/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
W, H = 1280, 720                                      # 16:9 like the capture

m = mujoco.MjModel.from_xml_path(XML)
m.vis.global_.fovy = GEOM["vfov_deg"]                 # 27.5 deg vertical -> 47.1 deg horizontal at 16:9
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
d = mujoco.MjData(m)
rec = np.load(REC, allow_pickle=True)
meta = json.loads(str(rec["_metadata_json"]))
fps = float(meta.get("fps", 50))
names = [m.joint(j).name for j in range(m.njnt)]
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
qadr = [m.jnt_qposadr[names.index(n)] for n in meta["dof_names"]]
root_q = m.jnt_qposadr[free]

# ---- camera from the robot's first frame
p0 = rec["root_pos"][0].astype(float)
yaw0 = R.from_quat(rec["root_quat_xyzw"][0]).as_euler("ZYX")[0]
fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])                       # robot facing (x axis of the pelvis)
D, hgt = GEOM["horizontal_distance_m"], GEOM["camera_height_m"]
b = np.radians(GEOM["camera_bearing_from_facing_deg"])                  # where the camera stood w.r.t. the person's facing
cdir = np.array([np.cos(yaw0 + b), np.sin(yaw0 + b), 0.0])
cam_pos = np.array([p0[0], p0[1], 0.0]) + D * cdir + np.array([0, 0, hgt])
to_robot = -cdir
az = np.arctan2(to_robot[1], to_robot[0]) + np.radians(GEOM["person_off_axis_signed_deg"])  # + = robot right of centre
el = np.radians(-GEOM["camera_pitch_down_deg"])                          # tilted up
view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 3.0
cam.lookat[:] = cam_pos + cam.distance * view
cam.azimuth, cam.elevation = np.degrees(az), np.degrees(el)

# projection for the in-frame check (pinhole, same intrinsics as the renderer)
f_px = (H / 2) / np.tan(np.radians(GEOM["vfov_deg"]) / 2)
right = np.cross(view, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, view)
def project(pts):
    v = pts - cam_pos
    z = v @ view
    return np.stack([W / 2 + f_px * (v @ right) / z, H / 2 - f_px * (v @ up) / z], 1)
robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]       # everything but the world (floor)

renderer = mujoco.Renderer(m, H, W)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
except OSError:
    font = None
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
                      stdin=subprocess.PIPE)
margin = dict(left=1e9, right=1e9, top=1e9, bottom=1e9)
N = len(rec["root_pos"])
for i in range(N):
    d.qpos[root_q:root_q + 3] = rec["root_pos"][i]
    x, y, z, w = rec["root_quat_xyzw"][i]
    d.qpos[root_q + 3:root_q + 7] = [w, x, y, z]
    d.qpos[qadr] = rec["dof_pos"][i]
    mujoco.mj_forward(m, d)
    # conservative robot extent: every geom's bounding sphere
    c = d.geom_xpos[robot_geoms]; rb = m.geom_rbound[robot_geoms][:, None]
    pts = np.concatenate([c + rb * a for a in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])])
    uv = project(pts)
    margin["left"] = min(margin["left"], uv[:, 0].min()); margin["right"] = min(margin["right"], W - uv[:, 0].max())
    margin["top"] = min(margin["top"], uv[:, 1].min()); margin["bottom"] = min(margin["bottom"], H - uv[:, 1].max())
    renderer.update_scene(d, cam)
    img = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(img)
    dr.text((20, 16), TITLE, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    dr.text((20, 50), f"t = {i / fps:5.2f} s", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    ff.stdin.write(img.tobytes())
ff.stdin.close(); ff.wait()
print(json.dumps({"video": OUT, "frames": N, "seconds": N / fps, "camera_pos": cam_pos.round(3).tolist(),
                  "min_margin_px": {k: round(float(v)) for k, v in margin.items()},
                  "min_margin_pct": {k: round(100 * float(v) / (W if k in ("left", "right") else H), 1) for k, v in margin.items()}}))
