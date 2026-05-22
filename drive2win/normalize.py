"""Input/output normalization for the navigation network.

Feature engineering (v13):
  - Dropped: ray_3_+135, ray_4_back, ray_5_-135 (rear rays)
  - Kept: ground_friction
  - Added: path_dir_x, path_dir_z, path_dist — direction/distance to
    a future point on the reference path (lookahead ~17 m)
  - Output: steering only (throttle hardcoded in policy)

Features (12): speed, heading_error, checkpoint_distance,
               ray_0_front, ray_1_+45, ray_2_+90, ray_6_-90, ray_7_-45,
               ground_friction, path_dir_x, path_dir_z, path_dist
"""
from __future__ import annotations
import numpy as np

SPD_MAX      = 20.0
DIST_MAX     = 100.0
RAY_MAX      = 50.0
FRIC_MAX     = 1.5
PATH_DIR_MAX = 50.0   # world-unit scale for direction components
PATH_DIST_MAX = 50.0  # world-unit scale for lookahead distance

# Columns to keep from the raw 12-column state array
KEEP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 11]

FEATURE_NAMES = [
    "speed", "heading_error", "checkpoint_distance",
    "ray_0_front", "ray_1_+45", "ray_2_+90", "ray_6_-90", "ray_7_-45",
    "ground_friction",
    "path_dir_x", "path_dir_z", "path_dist",
]
ACTION_NAMES = ["steering"]
N_FEATURES = 9
N_ACTIONS  = 1


def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Select 9 sensor features from a raw (N, 12) array and normalize.

    Returns float32 array of shape (N, 9). Path features are added
    separately via normalize_path_features().
    """
    s = np.asarray(states_raw, dtype=np.float32)[:, KEEP_COLS].copy()
    s[:, 0] = np.clip(s[:, 0] / SPD_MAX,     -1.0, 1.0)
    s[:, 1] = np.clip(s[:, 1] / np.pi,       -1.0, 1.0)
    s[:, 2] = np.clip(s[:, 2] / DIST_MAX,     0.0, 1.0)
    s[:, 3:8] = np.clip(s[:, 3:8] / RAY_MAX,  0.0, 1.0)
    s[:, 8] = np.clip(s[:, 8] / FRIC_MAX,     0.0, 1.0)
    return s


def normalize_path_features(path_feats: np.ndarray) -> np.ndarray:
    """Normalize (N, 3) or (3,) path features [dir_x, dir_z, dist].

    Returns float32 array same shape.
    """
    p = np.asarray(path_feats, dtype=np.float32).copy()
    single = p.ndim == 1
    if single:
        p = p[None, :]
    p[:, 0] = np.clip(p[:, 0] / PATH_DIR_MAX,  -1.0, 1.0)
    p[:, 1] = np.clip(p[:, 1] / PATH_DIR_MAX,  -1.0, 1.0)
    p[:, 2] = np.clip(p[:, 2] / PATH_DIST_MAX,  0.0, 1.0)
    return p[0] if single else p


def sensors_to_input(sensors: dict, path_feat: np.ndarray | None = None) -> np.ndarray:
    """Convert sensor dict + optional path feature to normalized input vector.

    Without path_feat: returns shape (9,)  — legacy / fallback
    With    path_feat: returns shape (12,) — full v13 model
    """
    rays = sensors["rays"]
    raw = np.array([
        sensors["speed"],
        sensors["heading_error"],
        sensors["checkpoint_distance"],
        rays[0], rays[1], rays[2], rays[6], rays[7],
        sensors.get("ground_friction", 1.0),
    ], dtype=np.float32)
    out = raw.copy()
    out[0] = np.clip(raw[0] / SPD_MAX,   -1.0, 1.0)
    out[1] = np.clip(raw[1] / np.pi,     -1.0, 1.0)
    out[2] = np.clip(raw[2] / DIST_MAX,   0.0, 1.0)
    out[3:8] = np.clip(raw[3:8] / RAY_MAX, 0.0, 1.0)
    out[8] = np.clip(raw[8] / FRIC_MAX,   0.0, 1.0)
    if path_feat is not None:
        out = np.concatenate([out, normalize_path_features(path_feat)])
    return out


def clip_action(a: np.ndarray, default_throttle: float = 0.9) -> tuple[float, float]:
    """Clamp network output to [-1, 1]. Handles both 1-output (steering-only)
    and 2-output (throttle+steering) networks."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if len(a) == 1:
        return default_throttle, float(np.clip(a[0], -1.0, 1.0))
    return float(np.clip(a[0], -1.0, 1.0)), float(np.clip(a[1], -1.0, 1.0))
