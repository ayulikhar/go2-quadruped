"""Forward trot gait for Unitree Go2 (MuJoCo)"""

import math
import os
import time

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
HIP_AMP  = 0.3
KNEE_AMP = 0.6
FREQ     = 2.0
DELTA    = -0.61

# Diagonal trot phase assignment
PHASE = {"FR": 0.0, "RL": 0.0, "FL": math.pi, "RR": math.pi}

# PD gains for torque-actuator tracking.
KP_HIP,   KD_HIP   = 40.0, 2.0
KP_THIGH, KD_THIGH = 250.0, 8.0
KP_CALF,  KD_CALF  = 200.0, 6.0

JOINTS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


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

    omega = 2.0 * math.pi * FREQ

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start  = time.time()
        SETTLE = 0.5
        RAMP   = 0.8
        while viewer.is_running():
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
            viewer.sync()


if __name__ == "__main__":
    main()
