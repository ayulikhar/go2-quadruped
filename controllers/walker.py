"""Forward trot gait for Unitree Go2 (MuJoCo)"""

import math
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_bridge

import mujoco
import mujoco.viewer

SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "unitree_go2",
    "scene.xml"
)


# Nominal standing pose (Go2 "home" keyframe values).
NOMINAL = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}

# gait parameters.
HIP_AMP  = 0.20
KNEE_AMP = 0.45
CALF_AMP = 0.65
FREQ     = 1.4
DELTA    = -0.45

# Diagonal trot phase assignment
PHASE = {"FR": 0.0, "RL": 0.0, "FL": math.pi, "RR": math.pi}

# PD gains for torque-actuator tracking.
KP_HIP,   KD_HIP   = 35.0, 4.0
KP_THIGH, KD_THIGH = 180.0, 12.0
KP_CALF,  KD_CALF  = 150.0, 10.0

JOINTS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# ---------------------------------------------------------------------------
# Fall recovery / get-up (see quadruped-recovery-and-getup skill).
#
# Recovery is a phased MOMENTUM maneuver, not a target pose: servoing
# straight to NOMINAL from a fallen orientation pushes all four legs into
# the ground symmetrically and the robot just extends its legs without
# rolling -- the "jumped, flipped, legs wrapped around it" failure mode.
# Instead we tuck -> abduct the up-side hips to initiate a roll -> push
# asymmetrically to flip onto the belly -> gather the feet -> stand.
# ---------------------------------------------------------------------------

# Sign of "outward abduction" is mirrored left/right. If sprawl makes the
# up-side legs fold IN instead of splay out in your model, flip these.
HAA_OUT_SIGN = {"FL": +1.0, "FR": -1.0, "RL": +1.0, "RR": -1.0}

# Per-leg keyframes as (hip, thigh, calf) radians. Starting values only --
# tune in sim once the structural checks (actuator type / HAA wired up /
# joint range / torque authority / foot friction) are confirmed sound.
TUCKED        = (0.0, 1.9, -2.6)   # legs pulled in, mass near roll axis
SPRAWL_OUT    = (1.0, 1.3, -2.0)   # up-side HAA driven outward (x HAA_OUT_SIGN)
SPRAWL_HOLD   = (0.0, 1.3, -2.0)   # down-side legs held tucked during sprawl
ROLL_DOWN_EXT = (0.0, 0.3, -1.2)   # down-side legs extend hard -> roll torque
ROLL_UP_PULL  = (0.6, 1.9, -2.6)   # up-side legs stay pulled tight during push
GATHER        = (0.0, 1.6, -2.6)   # all four gathered under the torso

# name, duration_s -- pose per phase resolved by recovery_phase_targets().
RECOVERY_PHASES = [
    ("tuck", 0.4),
    ("sprawl", 0.3),
    ("roll_push", 0.4),
    ("gather", 0.3),
    ("stand", 0.6),
    ("hold", 0.6),
]

# Recovery needs decisive, higher-authority pushes than walking does,
# especially at the hip (abduction has to roll/lift far more mass than
# sagittal leg swing) -- so it gets its own gains, not the walk gains.
RECOVERY_KP = {"hip": 120.0, "thigh": 60.0, "calf": 60.0}
RECOVERY_KD = {"hip": 3.0, "thigh": 2.0, "calf": 2.0}
RECOVERY_TORQUE_LIMIT = {"hip": 45.0, "thigh": 45.0, "calf": 45.0}

# Fall-detection thresholds.
UPRIGHT_ALIGN_MIN = 0.5   # base local z-axis . world z-axis; below this = tipped
FALLEN_HEIGHT_MAX = 0.2   # base height (m) below this while tipped = fallen
FALL_CONFIRM_TIME = 0.25  # must hold the fallen condition this long to trigger
                           # (filters out the brief tilt of a normal gait bounce)
BASE_BODY_CANDIDATES = ("base_link", "trunk", "base", "body")


def imu_bridge_reader(model, data, step):
    for sensor_name in ("imu_accel", "imu_gyro", "imu_quat"):
        value = imu_bridge.read_sensor(sensor_name)
        print(f"[IMU] {sensor_name}: {value}")


def find_base_body(model):
    """Resolve the floating-base body under a few common naming schemes."""
    for candidate in BASE_BODY_CANDIDATES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid != -1:
            return bid
    print(f"[recovery] none of {BASE_BODY_CANDIDATES} resolved as a body name; "
          f"falling back to body id 1. Fix BASE_BODY_CANDIDATES if this is wrong.")
    return 1


def mat_vec_transpose(xmat9, v):
    """Return xmat^T @ v for a flat row-major 3x3 (xmat^T maps world->body)."""
    m = xmat9
    return (
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    )


def base_state(model, data, base_id):
    """Return (upright_alignment, height, gravity_in_body_frame)."""
    xmat = data.xmat[base_id]
    upright_align = xmat[8]          # body local z-axis . world z-axis
    height = data.xpos[base_id][2]
    g_body = mat_vec_transpose(xmat, (0.0, 0.0, -1.0))
    return upright_align, height, g_body


def up_down_legs(g_body):
    """Classify which legs are 'up' (drive HAA outward) vs 'down' (push-extend)
    from gravity expressed in the base's local frame."""
    _, gy, gz = g_body
    if gz > 0.7:
        return ["FL", "FR"], ["RL", "RR"]      # on its back -- arbitrary roll side
    if gy > 0.5:
        return ["FR", "RR"], ["FL", "RL"]      # left side down -> right legs up
    if gy < -0.5:
        return ["FL", "RL"], ["FR", "RR"]      # right side down -> left legs up
    return ["FR", "RR"], ["FL", "RL"]          # prone / ambiguous -- pick a side


def recovery_phase_targets(phase_name, up, down):
    """Per-leg (hip, thigh, calf) targets for one named recovery phase."""
    targets = {}
    for leg in ("FL", "FR", "RL", "RR"):
        if phase_name == "tuck":
            targets[leg] = TUCKED
        elif phase_name == "sprawl":
            if leg in up:
                targets[leg] = (SPRAWL_OUT[0] * HAA_OUT_SIGN[leg], SPRAWL_OUT[1], SPRAWL_OUT[2])
            else:
                targets[leg] = SPRAWL_HOLD
        elif phase_name == "roll_push":
            targets[leg] = ROLL_DOWN_EXT if leg in down else ROLL_UP_PULL
        elif phase_name == "gather":
            targets[leg] = GATHER
        else:  # "stand", "hold"
            targets[leg] = (NOMINAL["hip"], NOMINAL["thigh"], NOMINAL["calf"])
    return targets


def ease(t):
    """Smoothstep 0->1 so phase transitions don't spike qacc."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3 - 2 * t)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data  = mujoco.MjData(model)

    qposadr, dofadr, actid = {}, {}, {}
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
        qposadr[name] = model.jnt_qposadr[jid]
        dofadr[name]  = model.jnt_dofadr[jid]
        actid[name]   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    # Seed standing pose.
    for leg in ("FL", "FR", "RL", "RR"):
        data.qpos[qposadr[f"{leg}_hip"]]   = NOMINAL["hip"]
        data.qpos[qposadr[f"{leg}_thigh"]] = NOMINAL["thigh"]
        data.qpos[qposadr[f"{leg}_calf"]]  = NOMINAL["calf"]
    mujoco.mj_forward(model, data)

    print(f"Loaded: {SCENE}")
    print(f"SKILL.md forward trot: DELTA={DELTA}, FREQ={FREQ} Hz. Close viewer to exit.")

    imu_bridge.publish_state(model, data)
    imu_bridge.run_reader_in_thread(imu_bridge_reader)

    omega = 2.0 * math.pi * FREQ
    base_id = find_base_body(model)

    # --- state machine: "walk" runs the trot gait, "recover" runs the
    # tuck/sprawl/roll/gather/stand phase sequence, then hands back to walk.
    mode = "walk"
    fallen_timer = 0.0
    recover_phase_idx = 0
    recover_phase_t = 0.0
    recover_prev_target = {}   # leg -> (hip, thigh, calf), interpolation start
    recover_up, recover_down = [], []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start   = time.time()
        prev_ts = start
        SETTLE  = 0.5
        RAMP    = 1.5
        while viewer.is_running():
            now = time.time()
            dt  = now - prev_ts
            prev_ts = now

            upright_align, height, g_body = base_state(model, data, base_id)
            is_tipped = upright_align < UPRIGHT_ALIGN_MIN and height < FALLEN_HEIGHT_MAX
            fallen_timer = fallen_timer + dt if is_tipped else 0.0

            if mode == "walk" and fallen_timer >= FALL_CONFIRM_TIME:
                print("[recovery] fall detected -- switching to get-up sequence")
                mode = "recover"
                recover_phase_idx = 0
                recover_phase_t = 0.0
                recover_up, recover_down = up_down_legs(g_body)
                print(f"[recovery] up-side legs: {recover_up}, down-side legs: {recover_down}")
                recover_prev_target = {
                    leg: (
                        data.qpos[qposadr[f"{leg}_hip"]],
                        data.qpos[qposadr[f"{leg}_thigh"]],
                        data.qpos[qposadr[f"{leg}_calf"]],
                    )
                    for leg in ("FL", "FR", "RL", "RR")
                }

            if mode == "recover":
                phase_name, phase_dur = RECOVERY_PHASES[recover_phase_idx]
                target = recovery_phase_targets(phase_name, recover_up, recover_down)
                alpha = ease(recover_phase_t / phase_dur)

                for leg in ("FL", "FR", "RL", "RR"):
                    prev_hip, prev_thigh, prev_calf = recover_prev_target[leg]
                    tgt_hip, tgt_thigh, tgt_calf = target[leg]
                    for jsuffix, prev_q, tgt_q in (
                        ("hip", prev_hip, tgt_hip),
                        ("thigh", prev_thigh, tgt_thigh),
                        ("calf", prev_calf, tgt_calf),
                    ):
                        jname = f"{leg}_{jsuffix}"
                        q_d = prev_q + alpha * (tgt_q - prev_q)
                        q   = data.qpos[qposadr[jname]]
                        qd  = data.qvel[dofadr[jname]]
                        kp, kd = RECOVERY_KP[jsuffix], RECOVERY_KD[jsuffix]
                        tau = kp * (q_d - q) - kd * qd
                        limit = RECOVERY_TORQUE_LIMIT[jsuffix]
                        tau = max(-limit, min(limit, tau))
                        data.ctrl[actid[jname]] = tau

                recover_phase_t += model.opt.timestep
                if recover_phase_t >= phase_dur:
                    recover_prev_target = target
                    recover_phase_idx += 1
                    recover_phase_t = 0.0
                    if recover_phase_idx >= len(RECOVERY_PHASES):
                        print("[recovery] sequence complete -- resuming trot")
                        mode = "walk"
                        fallen_timer = 0.0
                        start = time.time()   # re-ramp the gait from standstill

            else:  # mode == "walk"
                t      = time.time() - start
                gait_t = max(0.0, t - SETTLE)
                ramp   = min(1.0, gait_t / RAMP)

                for leg in ("FL", "FR", "RL", "RR"):
                    phi = omega * gait_t + PHASE[leg]

                    # SKILL.md waveforms — thigh uses phi directly (no delta);
                    # calf uses phi + delta. This is the ONE structural point of
                    # the skill: the offset lives on the foot-lift, not the swing.
                    thigh_offset = HIP_AMP  * math.sin(phi)
                    calf_offset  = -KNEE_AMP * (0.5 + 0.5 * math.sin(phi + DELTA))

                    thigh_d = NOMINAL["thigh"] + ramp * thigh_offset
                    calf_d  = NOMINAL["calf"]  + ramp * calf_offset
                    hip_d   = NOMINAL["hip"]

                    # PD -> torque, because Go2 actuators are <motor> (torque).
                    for jname, q_d, kp, kd in (
                        (f"{leg}_hip",   hip_d,   KP_HIP,   KD_HIP),
                        (f"{leg}_thigh", thigh_d, KP_THIGH, KD_THIGH),
                        (f"{leg}_calf",  calf_d,  KP_CALF,  KD_CALF),
                    ):
                        q   = data.qpos[qposadr[jname]]
                        qd  = data.qvel[dofadr[jname]]
                        tau = kp * (q_d - q) - kd * qd
                        data.ctrl[actid[jname]] = tau

            mujoco.mj_step(model, data)
            imu_bridge.tick()
            viewer.sync()


if __name__ == "__main__":
    main()
