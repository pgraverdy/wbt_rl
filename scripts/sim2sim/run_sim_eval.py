#!/usr/bin/env python3
"""holosoma run_sim (MuJoCo sim-to-sim) + scripted gantry release + trajectory recorder.

Reuses holosoma's `run_sim.run_simulation` and `DirectSimulation` unchanged; only replaces the loop of
`DirectSimulation.run` with one that:
  * waits until a release file exists, then disables the virtual gantry (the equivalent of key 9),
  * from that moment records the robot at 50 Hz (joint angles, root pose, body positions, sim time),
  * stops after --eval-duration seconds of simulated time and writes an .npz in the same layout as
    holosoma's EvalRecordingCallback (so data/eval_analyze.py works unchanged).

Run in the hsmujoco env from the holosoma_custom repo root, e.g.
    python run_sim_eval.py robot:g1-29dof --eval-out rec.npz --eval-duration 16 --eval-release-file /tmp/release
All other arguments go to holosoma's RunSimConfig (tyro), as for run_sim.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np


def pop_flag(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return cast(val)
    return default


EVAL_OUT = pop_flag("--eval-out")
EVAL_DURATION = pop_flag("--eval-duration", 16.0, float)
RELEASE_FILE = pop_flag("--eval-release-file", "/tmp/mujoco_eval_release")
REC_HZ = pop_flag("--eval-record-hz", 50, int)
STAND_Z = pop_flag("--eval-stand-z", 0.80, float)   # lower the gantry until the pelvis is at or below this height
LOWER_FILE = RELEASE_FILE + ".lower"
assert EVAL_OUT, "--eval-out is required"

import mujoco  # noqa: E402
import tyro  # noqa: E402
from loguru import logger  # noqa: E402

import holosoma.run_sim as run_sim  # noqa: E402
from holosoma.config_types.run_sim import RunSimConfig  # noqa: E402
from holosoma.utils import sim_utils  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from holosoma.simulator.shared.virtual_gantry import GantryCommand, GantryCommandData  # noqa: E402


def _match(names_short, mj_names):
    """Map holosoma names to MuJoCo element names that may carry a scene prefix."""
    out = []
    for n in names_short:
        cands = [i for i, m in enumerate(mj_names) if m == n or m.endswith("/" + n) or m.endswith("_" + n) and m[-len(n):] == n]
        exact = [i for i in cands if mj_names[i] == n or mj_names[i].endswith("/" + n)]
        out.append((exact or cands or [None])[0])
    return out


def eval_run(self) -> None:
    sim = self.simulator
    m, d = sim.root_model, sim.root_data
    fps = self.config.simulator.config.sim.fps
    rate = sim_utils.RateLimiter(fps)
    viewer_steps = self._calculate_viewer_steps()
    rec_every = max(1, round(fps / REC_HZ))

    jnames = [m.joint(j).name for j in range(m.njnt)]
    bnames = [m.body(b).name for b in range(m.nbody)]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    root_qadr = m.jnt_qposadr[free[0]]
    dof_names = list(sim.dof_names)
    body_names = list(sim.body_names)
    j_idx = _match(dof_names, jnames)
    b_idx = _match(body_names, bnames)
    missing = [n for n, i in zip(dof_names, j_idx) if i is None]
    assert not missing, f"joints not found in MuJoCo model: {missing}"
    q_adr = [m.jnt_qposadr[i] for i in j_idx]
    b_keep = [(n, i) for n, i in zip(body_names, b_idx) if i is not None]
    logger.info(f"[eval] sim {fps} Hz, recording every {rec_every} steps ({fps / rec_every:.0f} Hz); "
                f"{len(dof_names)} joints, {len(b_keep)}/{len(body_names)} bodies matched; waiting for {LOWER_FILE}")

    buf = {k: [] for k in ("dof_pos", "dof_vel", "root_pos", "root_quat_xyzw", "body_pos_w", "sim_time")}
    v_adr = [m.jnt_dofadr[i] for i in j_idx]
    released, t_rel, step = False, None, 0
    phase = "wait"                      # wait -> lowering -> lowered -> (released)
    g = getattr(sim, "virtual_gantry", None)
    lower_every = max(1, int(0.05 * fps))   # one length step every 50 ms of sim time
    sign, z_prev, n_adj, settle_until = -1.0, None, 0, None   # key 7 uses a negative length step
    while True:
        try:
            sim.simulate_at_each_physics_step()
            if step % viewer_steps == 0:
                sim.render()
            z = float(d.qpos[root_qadr + 2])
            if phase == "wait" and os.path.exists(LOWER_FILE):
                phase = "lowering"
                logger.info(f"[eval] lowering gantry from pelvis z={z:.3f} m towards {STAND_Z:.2f} m")
            if phase == "lowering" and step % lower_every == 0 and g is not None and g.enabled:
                if z <= STAND_Z:
                    phase, settle_until = "settling", d.time + 1.0
                    logger.info(f"[eval] pelvis at {z:.3f} m after {n_adj} steps; settling 1 s")
                else:
                    if n_adj == 6 and z_prev is not None and z > z_prev + 0.005:
                        sign = -sign   # went up: the length convention is the other way
                        logger.info("[eval] gantry moved the robot up; flipping the length step sign")
                    if n_adj == 0 or n_adj == 6:
                        z_prev = z
                    g.handle_command(GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": sign * 0.02}))
                    n_adj += 1
                    if n_adj > 200:
                        phase, settle_until = "settling", d.time + 1.0
                        logger.warning(f"[eval] stopped lowering after {n_adj} steps at z={z:.3f} m")
            if phase == "settling" and d.time >= settle_until:
                phase = "lowered"
                logger.info(f"[eval] lowered (pelvis z={z:.3f} m); waiting for release")
            if not released and phase == "lowered" and os.path.exists(RELEASE_FILE):
                if g is not None and g.enabled:
                    g.set_enable(False)
                released, t_rel = True, d.time
                logger.info(f"[eval] gantry released at sim time {t_rel:.3f} s (pelvis z={z:.3f} m); recording {EVAL_DURATION} s")
            if released and step % rec_every == 0:
                q = d.qpos
                buf["dof_pos"].append(q[q_adr].copy())
                buf["dof_vel"].append(d.qvel[v_adr].copy())
                buf["root_pos"].append(q[root_qadr:root_qadr + 3].copy())
                w, x, y, z = q[root_qadr + 3:root_qadr + 7]
                buf["root_quat_xyzw"].append(np.array([x, y, z, w]))
                buf["body_pos_w"].append(np.stack([d.xpos[i] for _, i in b_keep]).copy())
                buf["sim_time"].append(d.time - t_rel)
                if d.time - t_rel >= EVAL_DURATION:
                    break
            step += 1
            rate.sleep()
        except KeyboardInterrupt:
            logger.info("[eval] interrupted")
            break
        except Exception as e:  # keep holosoma's behaviour: log and stop
            logger.error(f"[eval] error at step {step}: {e}")
            traceback.print_exc()
            break

    if buf["dof_pos"]:
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}
        meta = {"fps": fps / rec_every, "dt": rec_every / fps, "sim_fps": fps, "dof_names": dof_names,
                "body_names": [n for n, _ in b_keep], "source": "mujoco run_sim (sim-to-sim)",
                "release_sim_time": t_rel}
        arrays["_metadata_json"] = np.array(json.dumps(meta))
        os.makedirs(os.path.dirname(os.path.abspath(EVAL_OUT)), exist_ok=True)
        np.savez_compressed(EVAL_OUT, **arrays)
        logger.info(f"[eval] saved {len(buf['dof_pos'])} frames to {EVAL_OUT}")
    else:
        logger.error("[eval] nothing recorded (release file never appeared?)")
    logger.info("[eval] done")
    os._exit(0)  # skip viewer teardown (MuJoCo's passive viewer can hang on close)


sim_utils.DirectSimulation.run = eval_run

if __name__ == "__main__":
    from holosoma.utils.eval_utils import init_eval_logging  # same as run_sim.main
    init_eval_logging()
    config = tyro.cli(RunSimConfig, config=TYRO_CONIFG)
    run_sim.run_simulation(config)
