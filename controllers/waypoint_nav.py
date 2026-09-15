"""Interactive waypoint navigation for the Unitree Go2 in MuJoCo.

Hover the mouse over the ground plane to see a highlighted green circle
preview of the target location. Single left-click commits that point as
the navigation goal and the quadruped walks there. Move the cursor while
the robot is walking to preview a new goal, then click to re-target in
real time.

Run:
    python3 controllers/waypoint_nav.py

Controls:
    - Hover over the floor to preview the target circle.
    - Left-click on the floor to set / update the waypoint.
    - SPACE clears the current goal so the robot stands still.
    - ESC or window close to quit.
"""

from __future__ import annotations

import math
import os
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(HERE, "..", "models", "unitree_go2", "scene_waypoint.xml")

# Waypoint / navigation parameters
WAYPOINT_REACHED_THRESHOLD = 0.20   # metres; robot considered "arrived"
MAX_LIN_SPEED = 0.6                 # m/s forward speed cap
MAX_YAW_RATE = 1.2                  # rad/s yaw rate cap
HEADING_GAIN = 2.0                  # P-gain: heading error -> yaw rate
APPROACH_GAIN = 1.5                 # P-gain: range -> forward speed

# Low-pass smoothing: new = (1-a)*prev + a*target
SMOOTH_ALPHA = 0.08

# Trot gait
GAIT_FREQ_HZ = 2.2
HIP_AMPLITUDE = 0.25
KNEE_AMPLITUDE = 0.35
STEP_LENGTH_SCALE = 0.6
YAW_STEP_SCALE = 0.25

# Nominal joint targets (matches the "home" keyframe in go2.xml)
HOME_HIP, HOME_THIGH, HOME_KNEE = 0.0, 0.9, -1.8

# PD gains
KP_HIP, KD_HIP = 60.0, 3.0
KP_THIGH, KD_THIGH = 80.0, 3.5
KP_KNEE, KD_KNEE = 80.0, 3.5

LEG_ORDER = ["FL", "FR", "RL", "RR"]
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}
LEG_HIP_SIGN = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}


# ---------------------------------------------------------------------------
# Click state (shared between GLFW callback and navigation loop)
# ---------------------------------------------------------------------------

_click_pending = [False]


# ---------------------------------------------------------------------------
# Waypoint state
# ---------------------------------------------------------------------------

class WaypointState:
    """Holds the current navigation goal."""

    def __init__(self) -> None:
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        self.have_goal: bool = False

    def update_goal(self, x: float, y: float) -> None:
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.have_goal = True

    def clear(self) -> None:
        self.have_goal = False


# ---------------------------------------------------------------------------
# Robot pose helpers
# ---------------------------------------------------------------------------

def trunk_xy_yaw(data: mujoco.MjData, trunk_body_id: int) -> tuple[np.ndarray, float]:
    xy = data.xpos[trunk_body_id][:2].copy()
    w, x, y, z = data.xquat[trunk_body_id]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return xy, yaw


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def navigate_to_waypoint(
    pos_xy: np.ndarray, yaw: float, goal_x: float, goal_y: float
) -> tuple[float, float, float]:
    """Return (forward_velocity, yaw_rate, range_to_goal)."""
    dx = goal_x - pos_xy[0]
    dy = goal_y - pos_xy[1]
    range_to_goal = math.hypot(dx, dy)

    if range_to_goal < WAYPOINT_REACHED_THRESHOLD:
        return 0.0, 0.0, range_to_goal

    desired_yaw = math.atan2(dy, dx)
    yaw_err = math.atan2(
        math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw)
    )
    yaw_rate = float(np.clip(HEADING_GAIN * yaw_err, -MAX_YAW_RATE, MAX_YAW_RATE))

    align = max(0.0, math.cos(yaw_err))
    v_forward = float(
        np.clip(APPROACH_GAIN * range_to_goal * align, 0.0, MAX_LIN_SPEED)
    )
    return v_forward, yaw_rate, range_to_goal


# ---------------------------------------------------------------------------
# Gait
# ---------------------------------------------------------------------------

def leg_targets(
    leg: str, phase: float, v_cmd: float, yaw_cmd: float
) -> tuple[float, float, float]:
    leg_phase = phase + LEG_PHASE[leg]
    swing = math.sin(leg_phase)
    lift = max(0.0, math.cos(leg_phase))

    stride_scale = STEP_LENGTH_SCALE * min(1.0, abs(v_cmd) / MAX_LIN_SPEED)
    v_dir = 1.0 if v_cmd >= 0 else -1.0

    thigh = HOME_THIGH + HIP_AMPLITUDE * stride_scale * swing * v_dir
    knee = HOME_KNEE + KNEE_AMPLITUDE * stride_scale * lift

    side_sign = 1.0 if leg in ("FL", "RL") else -1.0
    hip = HOME_HIP + LEG_HIP_SIGN[leg] * YAW_STEP_SCALE * yaw_cmd * side_sign
    return hip, thigh, knee


def apply_pd_torques(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: dict[str, tuple[float, float, float]],
) -> None:
    for leg in LEG_ORDER:
        hip_t, thigh_t, knee_t = targets[leg]
        for name, target, kp, kd in (
            (f"{leg}_hip",   hip_t,   KP_HIP,   KD_HIP),
            (f"{leg}_thigh", thigh_t, KP_THIGH, KD_THIGH),
            (f"{leg}_calf",  knee_t,  KP_KNEE,  KD_KNEE),
        ):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = model.actuator_trnid[act_id, 0]
            qpos_adr = model.jnt_qposadr[joint_id]
            qvel_adr = model.jnt_dofadr[joint_id]
            q = data.qpos[qpos_adr]
            dq = data.qvel[qvel_adr]
            tau = kp * (target - q) - kd * dq
            lo, hi = model.actuator_ctrlrange[act_id]
            data.ctrl[act_id] = float(np.clip(tau, lo, hi))


# ---------------------------------------------------------------------------
# Hover marker + click detection
# ---------------------------------------------------------------------------

def update_hover_marker(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_id: int,
    marker_mocap_id: int,
) -> np.ndarray | None:
    """Move the mocap marker to wherever the cursor hovers over the floor.

    The MuJoCo passive viewer continuously updates its internal perturbation
    object (_pert) with the geom under the cursor (pert.select) and the
    world-space hit point (pert.refpos). When the cursor is over the floor
    geom we move the marker there; otherwise we park it below the scene.

    Returns the hover world-position as an ndarray, or None if the cursor
    is not over the floor.
    """
    pert = getattr(viewer, "_pert", None)
    if pert is not None and int(pert.select) == floor_geom_id:
        hit = np.array(pert.refpos, dtype=float)
        data.mocap_pos[marker_mocap_id] = np.array([hit[0], hit[1], 0.01])
        return hit
    # Cursor not over the floor — hide the marker.
    data.mocap_pos[marker_mocap_id] = np.array([0.0, 0.0, -10.0])
    return None


def poll_click() -> bool:
    """Return True if a left-click was registered via the GLFW callback.

    Resets the flag so each click fires exactly once.
    """
    if _click_pending[0]:
        _click_pending[0] = False
        return True
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def navigation_loop(
    model, data, viewer, state,
    trunk_body_id, floor_geom_id, marker_mocap_id,
) -> None:
    gait_phase = 0.0
    v_cmd_smooth = 0.0
    yaw_cmd_smooth = 0.0
    last_sim_time = data.time

    while viewer.is_running():
        step_start = time.time()

        # Hover: move the green circle to wherever the cursor is on the floor.
        hover_point = update_hover_marker(
            viewer, model, data, floor_geom_id, marker_mocap_id
        )

        # Click: commit the hover point as the navigation goal.
        if poll_click() and hover_point is not None:
            state.update_goal(hover_point[0], hover_point[1])
            data.mocap_pos[marker_mocap_id] = np.array(
                [hover_point[0], hover_point[1], 0.01]
            )

        # Navigation: compute body-velocity command toward the goal.
        if state.have_goal:
            pos_xy, yaw = trunk_xy_yaw(data, trunk_body_id)
            v_target, yaw_target, range_to_goal = navigate_to_waypoint(
                pos_xy, yaw, state.goal_x, state.goal_y
            )
            if range_to_goal < WAYPOINT_REACHED_THRESHOLD:
                v_target, yaw_target = 0.0, 0.0
        else:
            v_target, yaw_target = 0.0, 0.0

        # Smooth the command.
        v_cmd_smooth = (1.0 - SMOOTH_ALPHA) * v_cmd_smooth + SMOOTH_ALPHA * v_target
        yaw_cmd_smooth = (1.0 - SMOOTH_ALPHA) * yaw_cmd_smooth + SMOOTH_ALPHA * yaw_target

        # Advance gait phase only while actually moving.
        dt = data.time - last_sim_time
        last_sim_time = data.time
        if abs(v_cmd_smooth) > 1e-3 or abs(yaw_cmd_smooth) > 1e-3:
            gait_phase = (gait_phase + 2.0 * math.pi * GAIT_FREQ_HZ * dt) % (2.0 * math.pi)

        targets = {
            leg: leg_targets(leg, gait_phase, v_cmd_smooth, yaw_cmd_smooth)
            for leg in LEG_ORDER
        }
        apply_pd_torques(model, data, targets)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
    if marker_body_id < 0:
        marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waypoint_marker")
    if marker_body_id < 0:
        raise RuntimeError("target_marker mocap body not found in scene")

    marker_mocap_id = model.body_mocapid[marker_body_id]
    state = WaypointState()

    def key_callback(keycode):
        if keycode == 32:  # SPACE
            state.clear()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        # Register GLFW mouse button callback for click detection.
        window = None
        try:
            window = viewer._platform._window
        except AttributeError:
            pass
        if window is None:
            try:
                window = viewer._platform.window
            except AttributeError:
                pass

        if window is not None:
            _original_mouse_cb = glfw.set_mouse_button_callback(window, None)

            def _mouse_button_cb(win, button, action, mods):
                if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and mods == 0:
                    _click_pending[0] = True
                if _original_mouse_cb is not None:
                    _original_mouse_cb(win, button, action, mods)

            glfw.set_mouse_button_callback(window, _mouse_button_cb)

        navigation_loop(
            model, data, viewer, state,
            trunk_body_id, floor_geom_id, marker_mocap_id,
        )


if __name__ == "__main__":
    main()
