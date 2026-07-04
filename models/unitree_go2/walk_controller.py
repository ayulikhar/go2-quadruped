"""Trot walking controller for the Unitree Go2 quadruped in MuJoCo.

Implements a simple open-loop trot gait by sending sinusoidal position
targets to the 12 leg actuators (hip/thigh/calf for FL, FR, RL, RR).

Diagonal pairs (FL+RR) and (FR+RL) swing in anti-phase to produce a trot.

Run:
    python3 walk_controller.py
"""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE_XML = str(Path(__file__).parent / "scene.xml")

# Actuator order in go2.xml:
#   0: FL_hip   1: FL_thigh   2: FL_calf
#   3: FR_hip   4: FR_thigh   5: FR_calf
#   6: RL_hip   7: RL_thigh   8: RL_calf
#   9: RR_hip  10: RR_thigh  11: RR_calf
ACTUATOR_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Nominal standing pose (radians) — knees bent, calves folded back.
# These match the "home" keyframe typically used for Go2.
STAND_POSE = {
    "FL_hip": 0.0, "FL_thigh": 0.9, "FL_calf": -1.8,
    "FR_hip": 0.0, "FR_thigh": 0.9, "FR_calf": -1.8,
    "RL_hip": 0.0, "RL_thigh": 0.9, "RL_calf": -1.8,
    "RR_hip": 0.0, "RR_thigh": 0.9, "RR_calf": -1.8,
}

# Gait parameters
GAIT_FREQ_HZ = 1.5          # step cycles per second
THIGH_AMPLITUDE = 0.35      # radians — how much the thigh swings
CALF_AMPLITUDE = 0.5        # radians — knee flexion during swing
STAND_DURATION_S = 1.5      # settle time before walking starts


def trot_targets(t: float) -> dict:
    """Return desired actuator positions at time t (seconds).

    Diagonal legs (FL+RR) and (FR+RL) are 180 degrees out of phase.
    """
    omega = 2.0 * math.pi * GAIT_FREQ_HZ
    phase_a = math.sin(omega * t)           # FL, RR
    phase_b = math.sin(omega * t + math.pi) # FR, RL

    # When a leg lifts (phase > 0), thigh rotates forward and calf folds.
    def leg_offset(phase):
        lift = max(phase, 0.0)  # only lift during the swing half-cycle
        thigh = -THIGH_AMPLITUDE * phase
        calf = CALF_AMPLITUDE * lift
        return thigh, calf

    fl_t, fl_c = leg_offset(phase_a)
    rr_t, rr_c = leg_offset(phase_a)
    fr_t, fr_c = leg_offset(phase_b)
    rl_t, rl_c = leg_offset(phase_b)

    targets = dict(STAND_POSE)
    targets["FL_thigh"] += fl_t; targets["FL_calf"] += fl_c
    targets["RR_thigh"] += rr_t; targets["RR_calf"] += rr_c
    targets["FR_thigh"] += fr_t; targets["FR_calf"] += fr_c
    targets["RL_thigh"] += rl_t; targets["RL_calf"] += rl_c
    return targets


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # Map actuator name -> ctrl index for fast lookup.
    act_index = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_NAMES
    }
    missing = [n for n, i in act_index.items() if i < 0]
    if missing:
        raise RuntimeError(f"Actuators not found in model: {missing}")

    # Note: Go2 actuators are 'motor' type (torque). The model's default
    # gains turn position-style targets into torques via the built-in
    # PD behavior of the actuator class. If your build of go2.xml uses
    # pure torque actuators you'll need to wrap this with an explicit PD
    # loop; the menagerie default works with position-like targets.

    print(f"Loaded {SCENE_XML}")
    print(f"Actuators: {len(act_index)}  Joints: {model.njnt}")
    print("Standing for", STAND_DURATION_S, "s, then trotting...")

    sim_start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - sim_start

            if t < STAND_DURATION_S:
                targets = STAND_POSE
            else:
                targets = trot_targets(t - STAND_DURATION_S)

            for name, q_des in targets.items():
                data.ctrl[act_index[name]] = q_des

            mujoco.mj_step(model, data)
            viewer.sync()
            # Real-time pacing
            time.sleep(max(0.0, model.opt.timestep - 1e-4))


if __name__ == "__main__":
    main()
