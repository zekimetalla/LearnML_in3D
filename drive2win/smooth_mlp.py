"""MLP policy — steering-only network, hardcoded throttle, stuck recovery.

v12 changes:
  - 9 input features (added ground_friction back).
  - Removed nav blend — model steers entirely from its own learned weights.
  - No EMA smoothing (ALPHA=1.0), STEER_GAIN=1.4 for sharp turns.
  - Checkpoint homing: blends toward heading_error when within 25 m of gate.

Usage:
    python 03_benchmark.py --tag v12b --weights nav_v12b.npz --module drive2win.smooth_mlp
"""
from __future__ import annotations
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input

THROTTLE         = 0.8   # hardcoded forward throttle
ALPHA            = 1.0   # no EMA smoothing — raw model output
STEER_GAIN       = 1.4   # amplify steering so sharp turns reach ±1
STUCK_THRESHOLD  = 15    # frames wedged before triggering escape
REVERSE_FRAMES   = 10    # frames to hold reverse
STUCK_SPEED      = 0.3   # m/s — speed threshold
RAY_WEDGE        = 4.0   # m — front ray below this = near wall
PURE_STUCK_THR   = 50    # fallback: trigger if very slow for this many frames
PURE_STUCK_SPEED = 0.15  # m/s — fallback speed threshold

# Checkpoint homing — blends toward heading_error when close to a gate
CP_HOMING_DIST   = 25.0  # metres — start blending
CP_HOMING_MAX    = 0.65  # max blend fraction (0 = all model, 1 = all homing)
CP_HOMING_GAIN   = 1.0   # proportional gain on heading_error


def make_policy(weights_path: str):
    """Return the v12 steering-only policy."""
    w = nn.load(weights_path)
    prev          = np.zeros(1, dtype=np.float32)
    stuck_count   = 0
    reverse_count = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal prev, stuck_count, reverse_count

        sensors = state["sensors"]
        speed   = sensors.get("speed", 1.0)
        rays    = sensors.get("rays", [50.0] * 8)
        front   = rays[0] if rays else 50.0
        left    = rays[6] if len(rays) > 6 else 50.0
        right   = rays[2] if len(rays) > 2 else 50.0

        wedged     = (speed < STUCK_SPEED and front < RAY_WEDGE
                      and (left < RAY_WEDGE or right < RAY_WEDGE))
        pure_stuck = speed < PURE_STUCK_SPEED

        if wedged or pure_stuck:
            stuck_count += 1
        else:
            stuck_count = 0

        threshold = STUCK_THRESHOLD if wedged else PURE_STUCK_THR
        if stuck_count >= threshold:
            reverse_count = REVERSE_FRAMES
            stuck_count   = 0

        if reverse_count > 0:
            reverse_count -= 1
            steer = -0.8 if right < left else 0.8
            prev = np.array([steer], dtype=np.float32)
            return (-1.0, steer)

        x     = sensors_to_input(sensors)
        raw   = nn.forward(x, w)
        prev  = raw.copy()
        steer = float(np.clip(raw[0] * STEER_GAIN, -1.0, 1.0))

        # Checkpoint homing: blend toward gate direction when close
        cp_dist = sensors.get("checkpoint_distance", 100.0)
        if cp_dist < CP_HOMING_DIST:
            heading_norm = float(np.clip(
                sensors.get("heading_error", 0.0) / np.pi, -1.0, 1.0))
            homing_steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            blend = (CP_HOMING_DIST - cp_dist) / CP_HOMING_DIST * CP_HOMING_MAX
            steer = steer * (1.0 - blend) + homing_steer * blend

        return (THROTTLE, steer)

    return policy
