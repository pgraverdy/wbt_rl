#!/usr/bin/env python3
"""Compare an eval recording (holosoma EvalRecordingCallback .npz) with the reference motion
(holosoma trainer-input .npz) and print tracking metrics.

The trainer pads the reference with default-pose transition frames before and after the clip, so the
recording is aligned to the reference by searching the frame offset that minimises the joint error
(expected near 100 frames = 2 s at 50 Hz). Joints are matched by name (Isaac Sim's joint order can
differ from the config order); root positions are compared relative to the first aligned frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-offset", type=int, default=250)
    ap.add_argument("--align-yaw", action="store_true",
                    help="rotate the recording about z so its heading matches the reference at the aligned start "
                         "(for simulators that spawn the robot at a fixed heading, e.g. MuJoCo sim-to-sim)")
    args = ap.parse_args()

    rec = np.load(args.rec, allow_pickle=True)
    ref = np.load(args.ref, allow_pickle=True)
    meta = json.loads(str(rec["_metadata_json"]))

    rec_dof_names = list(meta["dof_names"])
    ref_joint_names = [str(n) for n in ref["joint_names"]]
    idx = [rec_dof_names.index(n) for n in ref_joint_names]           # rec column for each ref joint
    q_rec = rec["dof_pos"][:, idx]                                     # (N, 29) in ref order
    q_ref = ref["joint_pos"][:, 7:]                                    # (T, 29)
    root_rec, root_ref = rec["root_pos"], ref["joint_pos"][:, :3]
    quat_rec = rec["root_quat_xyzw"]
    quat_ref = ref["joint_pos"][:, [4, 5, 6, 3]]                       # wxyz -> xyzw
    N, T = len(q_rec), len(q_ref)
    fps = meta.get("fps", 50)

    # --- resets / falls over the whole recording
    jumps = np.where(np.linalg.norm(np.diff(root_rec, axis=0), axis=1) > 0.3)[0] + 1
    min_height = float(root_rec[:, 2].min())

    # --- frame alignment: offset k such that rec[k + t] ~ ref[t]
    best = None
    for k in range(0, min(args.max_offset, N - T) + 1):
        e = np.abs(q_rec[k:k + T] - q_ref).mean()
        if best is None or e < best[1]:
            best = (k, e)
    if best is None:
        raise SystemExit(f"recording too short ({N} steps) for the reference ({T} frames)")
    k = best[0]
    q, rp, rq = q_rec[k:k + T], root_rec[k:k + T], quat_rec[k:k + T]
    yaw_offset_deg = 0.0
    body_rec_all = rec["body_pos_w"] if "body_pos_w" in rec.files else None
    if args.align_yaw:
        yaw = lambda qq: R.from_quat(qq).as_euler("ZYX")[0]  # heading (rotation about z)
        dyaw = yaw(quat_ref[0]) - yaw(rq[0])
        yaw_offset_deg = float(np.degrees(dyaw))
        Rz = R.from_euler("z", dyaw)
        p0 = rp[0].copy()
        rp = p0 + Rz.apply(rp - p0)
        rq = (Rz * R.from_quat(rq)).as_quat()
        if body_rec_all is not None:
            seg = body_rec_all[k:k + T]
            body_rec_all = body_rec_all.copy()
            body_rec_all[k:k + T] = p0 + Rz.apply((seg - p0).reshape(-1, 3)).reshape(seg.shape)

    jerr = np.abs(q - q_ref)                                           # (T, 29) rad
    disp_rec = rp - rp[0]
    disp_ref = root_ref - root_ref[0]
    pos_err = np.linalg.norm(disp_rec - disp_ref, axis=1)              # root displacement error (m)
    z_err = np.abs(rp[:, 2] - root_ref[:, 2])                          # absolute root height error (m)
    ang_err = np.degrees((R.from_quat(rq) * R.from_quat(quat_ref).inv()).magnitude())

    # body positions relative to the root (world frame), bodies matched by name
    body_err = None
    if "body_pos_w" in rec.files and "body_names" in meta:
        rb = [str(b) for b in meta["body_names"]]
        fb = [str(b) for b in ref["body_names"]]
        common = [b for b in fb if b in rb and b not in ("world",)]
        bi_rec = [rb.index(b) for b in common]
        bi_ref = [fb.index(b) for b in common]
        brec = body_rec_all[k:k + T][:, bi_rec] - rp[:, None, :]
        bref = ref["body_pos_w"][:, bi_ref] - root_ref[:, None, :]
        body_err = np.linalg.norm(brec - bref, axis=2)                 # (T, B) m

    clip_window = f"steps {k}..{k + T - 1} of {N}"
    print(f"Recording: {N} control steps at {fps} Hz ({N / fps:.1f} s); reference clip: {T} frames ({T / fps:.1f} s)")
    print(f"Alignment: reference frame 0 = recording step {k} (expected ~100: 2 s default-pose transition); {clip_window}")
    if args.align_yaw:
        print(f"Heading alignment: recording rotated by {yaw_offset_deg:+.1f} deg about z at the aligned start")
    print(f"Resets/teleports (root jump > 0.3 m between steps): {len(jumps)} {list(jumps[:10])}")
    print(f"Min root (pelvis) height over the whole recording: {min_height:.3f} m (reference min {root_ref[:, 2].min():.3f} m)")
    print("Over the reference clip:")
    print(f"  joint position error  mean {jerr.mean():.4f} rad ({np.degrees(jerr.mean()):.2f} deg), "
          f"per-frame mean max {jerr.mean(1).max():.4f} rad, worst single joint {jerr.max():.3f} rad")
    print(f"  root displacement err mean {pos_err.mean():.3f} m, max {pos_err.max():.3f} m, at end {pos_err[-1]:.3f} m "
          f"(reference travel {np.linalg.norm(disp_ref[-1]):.3f} m)")
    print(f"  root height error     mean {z_err.mean():.3f} m, max {z_err.max():.3f} m")
    print(f"  root orientation err  mean {ang_err.mean():.2f} deg, max {ang_err.max():.2f} deg")
    if body_err is not None:
        print(f"  body pos err (rel. to root, {body_err.shape[1]} bodies) mean {body_err.mean():.3f} m, "
              f"max {body_err.max():.3f} m")
    worst = np.argsort(-jerr.mean(0))[:5]
    print("  worst joints (mean abs error, rad): " + ", ".join(f"{ref_joint_names[j]} {jerr[:, j].mean():.3f}" for j in worst))

    out = Path(args.out)
    np.savez(out / "eval_metrics.npz", offset=k, joint_err=jerr, root_disp_err=pos_err, root_z_err=z_err,
             root_ang_err_deg=ang_err, joint_names=np.array(ref_joint_names))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(T) / fps
        fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        ax[0].plot(t, root_ref[:, 2], label="reference"); ax[0].plot(t, rp[:, 2], label="policy")
        ax[0].set_ylabel("pelvis height (m)"); ax[0].legend()
        ax[1].plot(t, np.degrees(jerr.mean(1))); ax[1].set_ylabel("mean joint err (deg)")
        ax[2].plot(t, pos_err, label="root displacement"); ax[2].plot(t, z_err, label="root height")
        ax[2].set_ylabel("error (m)"); ax[2].legend()
        for j in worst[:3]:
            ax[3].plot(t, q_ref[:, j], "--", label=f"{ref_joint_names[j]} ref")
            ax[3].plot(t, q[:, j], label=f"{ref_joint_names[j]} policy")
        ax[3].set_ylabel("joint angle (rad)"); ax[3].set_xlabel("time in clip (s)"); ax[3].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "eval_tracking.png", dpi=110)
        print(f"Plot: {out / 'eval_tracking.png'}")
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
