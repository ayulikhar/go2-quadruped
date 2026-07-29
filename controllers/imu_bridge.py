"""Shared MuJoCo state bridge so read_imu.py can read sensor values from the
go2_walk.py simulation.

Usage:
  - go2_walk.py imports `publish_state(model, data)` and calls it once after
    creating its MjModel / MjData, then calls `tick()` inside its main loop.
  - read_imu.py imports `get_state()` to obtain the live (model, data) pair
    and reads sensors via `data.sensor('imu_accel')`, etc.

Both processes must run in the SAME Python process for this to work, because
MuJoCo's MjData is an in-memory C struct — it cannot be shared across OS
processes. To use across processes, run read_imu as a function called from
go2_walk (see `run_reader_in_thread`).
"""

import threading
import time

_state_lock = threading.Lock()
_state = {"model": None, "data": None, "step": 0, "sim_time": 0.0}


def publish_state(model, data):
    """Register the live MuJoCo model + data with the bridge.

    Call this once from go2_walk.py right after creating MjModel/MjData.
    """
    with _state_lock:
        _state["model"] = model
        _state["data"] = data


def tick():
    """Advance bookkeeping — call once per simulation step from go2_walk.py."""
    with _state_lock:
        _state["step"] += 1
        if _state["data"] is not None:
            _state["sim_time"] = float(_state["data"].time)


def get_state():
    """Return (model, data, step, sim_time) for reader-side consumption.

    Returns (None, None, 0, 0.0) if go2_walk.py has not published yet.
    """
    with _state_lock:
        return _state["model"], _state["data"], _state["step"], _state["sim_time"]


def read_sensor(name):
    """Convenience: return a copy of the named sensor's data array, or None."""
    with _state_lock:
        data = _state["data"]
        if data is None:
            return None
        return data.sensor(name).data.copy()


def run_reader_in_thread(reader_fn, interval=0.5):
    """Spawn a daemon thread that periodically calls `reader_fn(model, data)`.

    go2_walk.py can call this once after publish_state() so read_imu's
    printing runs alongside the simulation loop in the same process.
    """
    def _loop():
        next_t = 0.0
        while True:
            model, data, step, sim_time = get_state()
            if data is not None and sim_time >= next_t:
                try:
                    reader_fn(model, data, step)
                except Exception as e:
                    print(f"[imu_bridge] reader error: {e}")
                next_t = sim_time + interval
            time.sleep(0.01)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t
