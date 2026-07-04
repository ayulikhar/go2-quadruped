---
name: quadruped-forward-locomotion
drift_categories: [SIMULATION]
description: Use when generating or fixing a walking/trot gait for a legged quadruped (e.g. Go2) in MuJoCo — especially when the robot walks BACKWARD instead of forward. Supplies the forward/backward sign convention (direction is set by the phase offset between the foot-lift/knee waveform and the fore-aft/hip swing, NOT by flipping the hip sign, swapping the diagonal legs, or reversing the gait in time — those do nothing or tip the robot), the stance/swing rule keyed off the loaded model's actual actuator names, a closed-loop self-verify (run headless, measure torso displacement along +x, adjust the knee phase offset, re-run until forward AND upright), and a verified forward-trot reference controller.
applicability_triggers:
  - walking sequence
  - walk forward
  - moves forward
  - move forward
  - walks backward
  - walking backwards
  - quadruped
  - quadruped gait
  - gait
  - trot
  - trotting
  - locomotion
  - legged robot
  - go2
  - make it walk forward
---

## QUADRUPED WALKING DIRECTION — FORWARD vs BACKWARD (COMMON FAILURE)

A walking sequence (trot gait) that walks the WRONG direction — the quadruped moves backward when asked to walk forward — is almost always stable and upright: the bug is direction, not balance. Do NOT rewrite the whole controller or re-tune amplitude/frequency. It is a one-parameter phase fix.

### Root cause
Travel direction is set by the phase offset between the FOOT-LIFT (knee) waveform and the FORE-AFT (hip) swing WITHIN each leg. If hip and knee are driven at the same phase — e.g. `hip = A*sin(phi)` together with `knee = -B*(0.5 + 0.5*sin(phi))` — the foot is planted during the wrong half of the stroke and the robot walks backward.

### What does NOT fix direction (do not try these — they do nothing or make it worse)
- Swapping the diagonal leg pairs (relabeling group A/B): no change in direction.
- Time-reversing the gait / negating the angular frequency `omega`: no change (a 0/pi diagonal trot is symmetric under it).
- Negating the hip amplitude: does not cleanly reverse and tends to TIP the robot over.

### The rule (why it goes forward)
The foot must be PLANTED (knee extended, ctrl ~ 0) during the hip's rearward power-stroke, and LIFTED (knee bent) during the forward recovery swing. For a model with +x = forward, hip hinge `axis="0 1 0"`, and legs hanging along -z, a positive hip target sweeps the foot rearward (-x); so the stance (planted) half must drive the hip toward more-positive. Achieve this by OFFSETTING the knee foot-lift waveform relative to the hip by a phase `delta`:
`knee = -B*(0.5 + 0.5*sin(phi + delta))`. The SIGN of `delta` sets the travel direction.

### How to get it right — MEASURE, don't guess
Re-prompting "make it walk forward" fails because the model re-emits the same same-phase structure with nothing to correct against. Make direction a measured property:
1. Run the sim headless. Net travel = `data.qpos[0]` at the end minus at the start (the torso free-joint x; +x is forward).
2. If it is negative, the robot walked backward. Change the knee phase offset `delta` and re-run.
3. Sweep `delta` (e.g. over [-1.6, 1.6] rad) until net travel is POSITIVE and the robot stays upright (final `qpos[2]` near the standing height; body up-axis still positive). Confirm the positive number before declaring success.

### Verified forward reference (hip+knee-per-leg quadruped, +x forward)
```python
# actuator order [FR_hip, FR_knee, FL_hip, FL_knee, RR_hip, RR_knee, RL_hip, RL_knee]
phases = [0, 0, np.pi, np.pi, np.pi, np.pi, 0, 0]   # diagonal trot: FR,RL vs FL,RR
hip_amp, knee_amp, freq = 0.3, 0.6, 2.0
delta = -0.61          # rad: knee-vs-hip offset that yields FORWARD (+x)
omega = 2 * np.pi * freq
for i in range(8):
    phi = omega * t + phases[i]
    if i % 2 == 0:                                   # hip: fore-aft swing
        data.ctrl[i] = hip_amp * np.sin(phi)
    else:                                            # knee: foot lift, phase-shifted
        data.ctrl[i] = -knee_amp * (0.5 + 0.5 * np.sin(phi + delta))
```
This walks ~+0.6 m forward over 8 s, upright. `delta ~ 0` walks backward; the offset is the fix.

### Adapt to the actual model — do not assume indices
Resolve joints/actuators by NAME from the loaded model (`model.actuator(i).name`, `mj_name2id`), not hard-coded indices. A simplified quadruped has hip+knee per leg; the full 12-DOF Unitree Go2 uses hip/thigh/calf (abduction about x, thigh+calf pitch about y). Apply the same principle to the fore-aft (thigh) joint and keep `delta` as the direction lever, found by the measure-and-adjust loop above.

### Common code pitfalls in these controllers
- The keyframe enum is `mujoco.mjtObj.mjOBJ_KEY` (not `mj_KEY`).
- Do not `import mujoco.viewer` inside a function that also references `mujoco.*` — it shadows `mujoco` as a local and raises `UnboundLocalError`. Import it at module top.
