"""
GMR retargeter output → holosoma trainer input (form B).

The holosoma WBT trainers (holosoma and holosoma_custom) expect the "holosoma format":
    fps            : float (50 Hz)
    joint_pos      : (T, 7 + N_dof)  root [xyz, wxyz] followed by the joints
    joint_vel      : (T, 6 + N_dof)  root [lin vel xyz, ang vel xyz] followed by the joints
    body_pos_w     : (T, N_bodies, 3), body_quat_w : (T, N_bodies, 4) wxyz
    body_lin_vel_w, body_ang_vel_w : (T, N_bodies, 3)
    joint_names, body_names : the *holosoma* robot model's names (51 bodies for G1 29-DOF,
                              including "world")

An earlier version of this module did its own FK on GMR's mocap MJCF (38 bodies, no root DOFs in
joint_pos, empty joint_names), which the trainer rejects ("Joint names (0) != joint_pos columns").
Instead, build the robot qpos from the GMR pkl and run it through the same holosoma bridge
(convert_data_format_mj.py via wrappers/holosoma_convert.py) used for the holosoma retargeters, so
the output is produced by the model the trainer will actually simulate.

GMR pkl : fps, root_pos (T,3), root_rot (T,4) **xyzw**, dof_pos (T,N_dof) in G1 actuator order
          (same order as the holosoma G1 joints).
Runs in the `hsretargeting` env (MuJoCo bridge). The bridge opens a MuJoCo viewer, so a DISPLAY is
required.
"""
import pickle
import re
from pathlib import Path

import numpy as np

from .._subprocess import conda_run
from .._config import repo_root

_WRAPPER = "src/motion_convertor/wrappers/holosoma_convert.py"
_ENV = "hsretargeting"
_RETARGETER_ROOT = "modules/01_retargeting/holosoma_custom"
_OUTPUT_FPS = 50


def convert(
    pkl_path: Path | str,
    out_path: Path | str,
    robot: str = "G1_29dof",
    output_fps: int = _OUTPUT_FPS,
    object_name: str = "ground",
    xml_path: Path | str | None = None,  # kept for call-site compatibility; unused
    **kwargs,
) -> None:
    """
    Convert a GMR output .pkl to a holosoma trainer input .npz.

    Parameters
    ----------
    pkl_path : GMR output .pkl
    out_path : destination .npz path
    robot    : e.g. "G1_29dof" (selects the holosoma robot model)
    """
    # absolute paths: the holosoma bridge runs with its own working directory
    pkl_path = Path(pkl_path).absolute()
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    input_fps = int(round(float(pkl["fps"])))
    root_pos = np.asarray(pkl["root_pos"], dtype=np.float32)             # (T, 3)
    root_rot_wxyz = np.asarray(pkl["root_rot"], dtype=np.float32)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_pos = np.asarray(pkl["dof_pos"], dtype=np.float32)               # (T, N_dof)
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)    # (T, 7 + N_dof)

    # Parse robot name (e.g. G1_27dof -> type=g1, dof=27)
    m = re.match(r"([A-Za-z0-9]+)_(\d+)dof", robot)
    robot_type = m.group(1).lower() if m else "g1"
    robot_dof = int(m.group(2)) if m else 29
    if dof_pos.shape[1] != robot_dof:
        raise ValueError(f"GMR dof_pos has {dof_pos.shape[1]} joints but robot {robot} has {robot_dof}")

    # Bridge input: npz with `qpos` and `fps`
    qpos_path = out_path.with_name(out_path.stem + "_qpos_tmp.npz")
    np.savez(qpos_path, qpos=qpos, fps=np.array(input_fps))

    wrapper = repo_root() / _WRAPPER
    retargeter_root = repo_root() / _RETARGETER_ROOT
    cmd = (
        f"python {wrapper} "
        f"--input_file {qpos_path} "
        f"--output_name {out_path} "
        f"--robot_type {robot_type} "
        f"--robot_dof {robot_dof} "
        f"--input_fps {input_fps} "
        f"--output_fps {output_fps} "
        f"--object_name {object_name} "
        f"--retargeter_root {retargeter_root}"
    )
    try:
        conda_run(_ENV, cmd, cwd=repo_root())
    finally:
        qpos_path.unlink(missing_ok=True)

    if not out_path.exists():
        raise RuntimeError(f"holosoma bridge did not produce {out_path}")
