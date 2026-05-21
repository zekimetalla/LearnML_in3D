"""Input/output normalization for the navigation network.

Feature engineering (v12):
  - Dropped: ray_3_+135, ray_4_back, ray_5_-135 (rear rays, unhelpful on forward driving)
  - Kept: ground_friction (varies 0.4–1.2 across surface types, useful signal)
  - Output: steering only (throttle hardcoded in policy)

Kept features (9): speed, heading_error, checkpoint_distance,
                   ray_0_front, ray_1_+45, ray_2_+90, ray_6_-90, ray_7_-45,
                   ground_friction
"""
from __future__ import annotations
import numpy as np

SPD_MAX   = 20.0
DIST_MAX  = 100.0
RAY_MAX   = 50.0
FRIC_MAX  = 1.5

# Columns to keep from the raw 12-column state array
KEEP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 11]

FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
]
ACTION_NAMES = ["steering"]
N_FEATURES = 9
N_ACTIONS  = 1


def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Select 9 features from a raw (N, 12) array and normalize to [-1, 1].

    Returns float32 array of shape (N, 9).
    """
    s = np.asarray(states_raw, dtype=np.float32)[:, KEEP_COLS].copy()
    s[:, 0] = np.clip(s[:, 0] / SPD_MAX,   -1.0, 1.0)     # speed
    s[:, 1] = np.clip(s[:, 1] / np.pi,     -1.0, 1.0)     # heading_error
    s[:, 2] = np.clip(s[:, 2] / DIST_MAX,   0.0, 1.0)     # checkpoint_distance
    s[:, 3:8] = np.clip(s[:, 3:8] / RAY_MAX, 0.0, 1.0)    # 5 rays
    s[:, 8] = np.clip(s[:, 8] / FRIC_MAX,   0.0, 1.0)     # ground_friction
    return s


def sensors_to_input(sensors: dict) -> np.ndarray:
    """Convert live sensor dict to normalized 9-vector for the network.

    Returns shape (9,), float32.
    """
    rays = sensors["rays"]
    raw = np.array([
        sensors["speed"],
        sensors["heading_error"],
        sensors["checkpoint_distance"],
        rays[0],   # ray_0_front
        rays[1],   # ray_1_+45
        rays[2],   # ray_2_+90
        rays[6],   # ray_6_-90
        rays[7],   # ray_7_-45
        sensors.get("ground_friction", 1.0),
    ], dtype=np.float32)
    out = raw.copy()
    out[0] = np.clip(raw[0] / SPD_MAX,   -1.0, 1.0)
    out[1] = np.clip(raw[1] / np.pi,     -1.0, 1.0)
    out[2] = np.clip(raw[2] / DIST_MAX,   0.0, 1.0)
    out[3:8] = np.clip(raw[3:8] / RAY_MAX, 0.0, 1.0)
    out[8] = np.clip(raw[8] / FRIC_MAX,   0.0, 1.0)
    return out


def clip_action(a: np.ndarray, default_throttle: float = 0.9) -> tuple[float, float]:
    """Clamp network output to [-1, 1]. Handles both 1-output (steering-only)
    and 2-output (throttle+steering) networks."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if len(a) == 1:
        return default_throttle, float(np.clip(a[0], -1.0, 1.0))
    return float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
