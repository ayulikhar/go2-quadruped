---
name: quadruped-recovery-and-getup
description: Use when authoring a self-righting, fall-recovery, or get-up sequence for a legged robot in MuJoCo — a quadruped (e.g. Unitree Go2) that has fallen on its side or back and needs to right itself and stand. Especially when the robot "extends its legs but won't rotate the hips / won't roll over" despite the maneuver being described in the prompt. Covers why recovery is a phased momentum maneuver rather than a single target pose, the tuck→abduct→roll→gather→stand phase recipe, and the ordered "hips won't rotate" structural ladder (actuator type, hip-abduction actually commanded, joint/ctrl range, control authority, foot friction) to check before tuning any joint targets.
---

## Recovery Is a Momentum Maneuver, Not a Target Pose

A fallen quadruped does NOT self-right by servoing to the standing pose. Feeding the final "standing" qpos to position servos makes all four legs push symmetrically into the ground — the body extends its legs but never rolls, so it stays stuck on its side (the exact "extends legs, hips don't rotate" signature). Self-righting requires **asymmetric, timed pushes that build angular momentum to roll the torso**, then a push-up once the feet are underneath.

Author it as a **phase sequence** (explicit keyframes / timed waypoints), holding each phase for a fixed duration before advancing:

1. **Tuck** — pull all legs in (flex thigh + calf) to shrink the footprint and get mass close to the roll axis. ~0.3–0.5 s.
2. **Sprawl / abduct** — drive the **hip abduction (HAA)** joints on the "up" side outward to plant those feet wide and shift the contact base past the CoM. This is the phase that actually initiates the roll — if HAA never moves, the robot cannot leave its side. ~0.3 s.
3. **Asymmetric roll push** — extend the down-side legs while the up-side legs pull, generating a torque about the body's roll axis to flip onto the belly. Momentum matters here — a slow quasi-static servo won't do it; command a decisive push. ~0.3–0.5 s.
4. **Gather feet under body** — once prone, abduct/flex all four legs to bring the feet beneath the torso.
5. **Push to stand** — extend all four symmetrically to the nominal standing pose, then hold to stabilize.

Interpolate ctrl targets between phase keyframes rather than snapping — a step change spikes qacc and can blow up the integrator.

## "Legs Extend But Hips Won't Rotate" — Structural Ladder

Check in THIS ORDER before touching joint-target magnitudes. Describing the maneuver in the prompt does nothing if the emitted control loop never drives the abduction DOF, or the model can't produce the torque.

1. **Actuator type vs. what you're writing.** MuJoCo Menagerie `unitree_go2` ships **torque `<motor>`** actuators — `data.ctrl[i]` is a *torque*, not a target angle. If the sequence writes joint *angles* into ctrl, gravity flops the legs into rough extension (looks like it "works") but hip abduction has to fight the whole body's weight and a small constant torque can't hold it → hips don't move. Fix: either run a PD loop each step (`ctrl = kp*(target − q) − kd*qvel`) or swap the hips to `<position kp=...>` actuators. This is the single most common cause of this exact symptom.

2. **Are the HAA actuators actually being commanded?** Go2 actuator order is `FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, …, RL_calf` — the abduction actuators are indices **0, 3, 6, 9**. A loop that only writes thigh/calf indices leaves hips at 0. Verify by printing `data.ctrl` for those indices, and confirm the actuator names you target resolve (a name typo silently no-ops via `mj_name2id → -1`).

3. **Joint range / ctrlrange clamp.** Go2 hip abduction range is ≈ ±1.05 rad (±60°). If the model's `<joint range>` or the actuator's `ctrlrange` is tighter than the pose needs, the target is clamped and the leg barely splays. Check both against the angle the roll phase actually requires.

4. **Control authority.** For torque motors compare peak torque to the gravitational load holding that joint down; for position servos check `kp`/`forcerange` aren't saturating (`ctrl` pinned at `forcerange`). Abduction must lift/roll far more mass than sagittal leg extension, so a gain that extends legs fine can be far too weak to rotate the hips. Raise kp/forcerange (or add roll momentum in phase 3 so you don't need a static hold).

5. **Foot friction.** Rolling converts leg push into torso rotation only if the feet grip. On a slick surface (low tangential friction / a ramp) the push slides instead of rolling. Put foot geoms on high-friction flat ground (`friction="1.5 0.1 0.01"`, `condim=6`) to isolate the maneuver before adding terrain back.

Only after 1–5 are confirmed sound is it worth tuning target angles or hold durations. Chasing joint targets while the abduction DOF is uncommanded or under-actuated is the trap that burns dozens of edits without moving the robot.
