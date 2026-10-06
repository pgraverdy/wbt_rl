#!/usr/bin/env python3
"""Camera pose relative to the person at frame 0 of a clip, from GVHMR's camera-frame and world-frame estimates.
Usage (env gmr): python scripts/viz/camera_geometry.py <hmr4d_results.pt> <out.json> [smplx model dir]
  smplx model dir defaults to $SMPLX_MODEL_DIR, then <wbt_rl>/data/00_raw_datasets/models/models_smplx_v1_1/models"""
import json, os, sys
from pathlib import Path
import numpy as np, smplx, torch
from scipy.spatial.transform import Rotation as R
pt, out = sys.argv[1], sys.argv[2]
SMPLX = sys.argv[3] if len(sys.argv) > 3 else os.environ.get(
    "SMPLX_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "data/00_raw_datasets/models/models_smplx_v1_1/models"))
d = torch.load(pt, weights_only=False, map_location="cpu")
g, c = d["smpl_params_global"], d["smpl_params_incam"]; K = d["K_fullimg"][0].numpy()
m = smplx.create(SMPLX, "smplx", gender="neutral", num_betas=10, use_pca=False)
def run(p, i):
    with torch.no_grad():
        o = m(betas=p["betas"][i:i+1].float(), global_orient=p["global_orient"][i:i+1].float(),
              body_pose=p["body_pose"][i:i+1].float(), transl=p["transl"][i:i+1].float())
    return o.joints[0].numpy(), o.vertices[0].numpy()
Y2Z = R.from_euler("x", 90, degrees=True)          # GVHMR world (y up) -> z up
F = 0
jc, _ = run(c, F); jw, _ = run(g, F)
Rc = R.from_rotvec(c["global_orient"][F].numpy()); Rw = R.from_rotvec(g["global_orient"][F].numpy())
Rwc = Y2Z * Rw * Rc.inv()                           # camera -> world (z up)
pel = Y2Z.apply(jw[0]); C = pel - Rwc.apply(jc[0])
floor = min(Y2Z.apply(run(g, i)[1])[:, 2].min() for i in range(0, min(30, len(g["transl"])), 5))
fwd_cam = Rwc.apply([0, 0, 1.]); right_cam = Rwc.apply([1, 0, 0.])
vec = pel - C; hvec = vec * [1, 1, 0]
body_fwd = (Y2Z * Rw).apply([0, 0, 1.])            # SMPL body forward = +z
ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
wrap = lambda a: (a + 180) % 360 - 180
res = dict(frame=F, focal_px=float(K[0, 0]), vfov_deg=float(np.degrees(2 * np.arctan(K[1, 2] / K[1, 1]))),
           hfov_deg=float(np.degrees(2 * np.arctan(K[0, 2] / K[0, 0]))),
           camera_height_m=float(C[2] - floor), camera_pitch_down_deg=float(np.degrees(np.arcsin(-fwd_cam[2]))),
           camera_roll_deg=float(np.degrees(np.arcsin(right_cam[2]))),
           horizontal_distance_m=float(np.linalg.norm(hvec)), distance_3d_m=float(np.linalg.norm(vec)),
           # where the camera stands relative to the person's facing: 0 = in front, +-180 = behind, + = person's left
           camera_bearing_from_facing_deg=float(wrap(ang(-hvec) - ang(body_fwd))),
           # person's horizontal offset from the optical axis, + = right of the image centre
           person_off_axis_signed_deg=float(np.degrees(np.arctan2(jc[0][0], jc[0][2]))),
           pelvis_height_m=float(pel[2] - floor))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: round(v, 2) for k, v in res.items()}))
