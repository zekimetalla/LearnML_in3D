"""MLP policy with EMA smoothing, stuck-recovery, and proportional nav blend.

Three layers on top of the raw MLP:

1. EMA smoothing (alpha=0.6): converts discrete WASD snap outputs into
   ramps so steering/throttle change gradually — reduces wall impacts.

2. Stuck detection: if speed < 0.5 m/s for STUCK_THRESHOLD consecutive
   frames the car is pinned to a wall. Override with full reverse +
   opposite steering for REVERSE_FRAMES frames, then resume normal policy.

3. Proportional nav blend: on open road (front ray > NAV_OPEN), blend
   MLP steering with a hardcoded proportional rule (steer = -heading_norm
   * NAV_GAIN). The MLP handles wall avoidance; the formula handles
   heading correction that WASD data can't teach reliably.

Usage:
    python 03_benchmark.py --tag v10 --module drive2win.smooth_mlp
"""
from __future__ import annotations
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input, clip_action

ALPHA            = 0.6   # EMA weight for new prediction
STUCK_THRESHOLD  = 15    # frames wedged before triggering escape
REVERSE_FRAMES   = 10    # frames to hold reverse (shorter = less overshoot)
STUCK_SPEED      = 0.3   # m/s — speed threshold
RAY_WEDGE        = 4.0   # m — front ray below this = near wall
PURE_STUCK_THR   = 50    # fallback: trigger escape after this many slow frames
PURE_STUCK_SPEED = 0.15  # m/s — speed for fallback stuck (catches ramp case)

NAV_BLEND        = 0.2   # fraction of steering from proportional nav
NAV_GAIN         = 0.8   # proportional nav gain
NAV_OPEN         = 6.0   # m — front ray must be > this to apply nav blend


def make_policy(weights_path: str):
    """Return a smoothed MLP policy with stuck-recovery and nav blend."""
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

        # primary: wedged = slow AND front wall close AND hemmed in on a side
        wedged = (speed < STUCK_SPEED and front < RAY_WEDGE
                  and (left < RAY_WEDGE or right < RAY_WEDGE))
        # fallback: very slow for a long time even without ray detection (ramp)
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
            prev = np.array([-1.0, steer], dtype=np.float32)
            return (-1.0, steer)

        # --- normal smoothed MLP ---
        x        = sensors_to_input(sensors)
        raw      = nn.forward(x, w)
        smoothed = ALPHA * raw + (1.0 - ALPHA) * prev

        # --- proportional nav blend (open road only) ---
        if front > NAV_OPEN:
            heading_norm = float(np.clip(
                sensors.get("heading_error", 0.0) / np.pi, -1.0, 1.0))
            nav_steer = float(np.clip(-heading_norm * NAV_GAIN, -1.0, 1.0))
            blended_steer = smoothed[1] * (1 - NAV_BLEND) + nav_steer * NAV_BLEND
            smoothed = np.array([smoothed[0], blended_steer], dtype=np.float32)

        prev = smoothed.copy()
        return clip_action(smoothed)

    return policy
