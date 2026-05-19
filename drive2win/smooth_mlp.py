"""MLP policy with EMA smoothing and stuck-recovery override.

Two layers on top of the raw MLP:

1. EMA smoothing (alpha=0.6): converts discrete WASD snap outputs into
   ramps so steering/throttle change gradually — reduces wall impacts.

2. Stuck detection: if speed < 0.5 m/s for STUCK_THRESHOLD consecutive
   frames the car is pinned to a wall. Override with full reverse +
   opposite steering for REVERSE_FRAMES frames, then resume normal policy.
   This fixes the "hit wall and freeze" failure mode from behavioral
   cloning — the model rarely sees truly stuck states in training data.

Usage:
    python 03_benchmark.py --tag v7-all3 --module drive2win.smooth_mlp
"""
from __future__ import annotations
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input, clip_action

ALPHA            = 0.6   # EMA weight for new prediction
STUCK_THRESHOLD  = 15    # frames wedged before triggering escape
REVERSE_FRAMES   = 20    # frames to hold reverse
STUCK_SPEED      = 0.3   # m/s — speed threshold
RAY_WEDGE        = 4.0   # m — front ray below this = near wall


def make_policy(weights_path: str):
    """Return a smoothed MLP policy with stuck-recovery override."""
    w = nn.load(weights_path)
    prev          = np.zeros(2, dtype=np.float32)
    stuck_count   = 0
    reverse_count = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal prev, stuck_count, reverse_count

        sensors = state["sensors"]
        speed   = sensors.get("speed", 1.0)
        rays    = sensors.get("rays", [50.0] * 8)
        front   = rays[0] if rays else 50.0
        left    = rays[6] if len(rays) > 6 else 50.0   # ray_6_-90
        right   = rays[2] if len(rays) > 2 else 50.0   # ray_2_+90

        # wedged = slow AND front wall close AND hemmed in on a side
        wedged = (speed < STUCK_SPEED and front < RAY_WEDGE
                  and (left < RAY_WEDGE or right < RAY_WEDGE))

        if wedged:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= STUCK_THRESHOLD:
            reverse_count = REVERSE_FRAMES
            stuck_count   = 0

        if reverse_count > 0:
            reverse_count -= 1
            # steer away from the closer wall
            steer = -0.8 if right < left else 0.8
            prev = np.array([-1.0, steer], dtype=np.float32)
            return (-1.0, steer)

        # --- normal smoothed MLP ---
        x        = sensors_to_input(sensors)
        raw      = nn.forward(x, w)
        smoothed = ALPHA * raw + (1.0 - ALPHA) * prev
        prev     = smoothed.copy()
        return clip_action(smoothed)

    return policy
