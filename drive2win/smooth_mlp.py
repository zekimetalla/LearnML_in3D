"""MLP policy — steering-only, stuck recovery, checkpoint homing + orbit escape.

v13 changes:
  - 12 input features (9 sensors + 3 path-relative features).
  - STEER_GAIN lowered to 1.4 — 1.9 caused violent overshoots near CPs.
  - Homing ramps from 25 m → 85% blend at gate.
  - Orbit escape: if moving AWAY from CP while within 15 m, crawl + full homing.
  - Throttle brakes on bad approach angles near CPs.

Usage:
    python 03_benchmark.py --tag v13 --weights nav_v13.npz --module drive2win.smooth_mlp
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input

THROTTLE         = 0.8
STEER_GAIN       = 1.4   # was 1.9 — lower to stop violent CP overshoots
STUCK_THRESHOLD  = 15
REVERSE_FRAMES   = 10
STUCK_SPEED      = 0.3
RAY_WEDGE        = 4.0
PURE_STUCK_THR   = 50
PURE_STUCK_SPEED = 0.15

CP_HOMING_DIST   = 50.0  # metres — start blending steering toward checkpoint
CP_HOMING_MAX    = 1.0   # max steering blend during approach — full override at gate
CP_HOMING_GAIN   = 5.0
CP_BRAKE_DIST    = 15.0  # metres — start braking toward checkpoint
CP_MIN_THROTTLE  = 0.5   # throttle at gate
CP_GATE_DIST     = 5.0   # within this: 100% heading correction to thread the gate
CP_ORBIT_DIST    = 8.0   # retreat detection zone (must be > CP_GATE_DIST)
CP_ORBIT_FRAMES  = 25    # frames stuck near CP before forcing a reverse (~1.25 s at 20 Hz)


def _load_path_ref(weights_path: str) -> np.ndarray | None:
    wp  = Path(weights_path)
    tag = wp.stem.replace("nav_", "")
    ref = wp.parent / f"path_ref_{tag}.npy"
    if ref.exists():
        path = np.load(ref).astype(np.float32)
        print(f"  loaded reference path: {ref.name} ({len(path)} points)")
        return path
    return None


def _path_features(pos_xz: np.ndarray, path_ref: np.ndarray,
                   lookahead: int = 5) -> np.ndarray:
    dists   = np.linalg.norm(path_ref - pos_xz, axis=1)
    nearest = int(np.argmin(dists))
    future  = path_ref[min(nearest + lookahead, len(path_ref) - 1)]
    delta   = future - pos_xz
    dist    = float(np.linalg.norm(delta))
    return np.array([delta[0], delta[1], dist], dtype=np.float32)


def make_policy(weights_path: str):
    w        = nn.load(weights_path)
    path_ref = _load_path_ref(weights_path)
    use_path = (path_ref is not None) and (w["W1"].shape[0] == 12)

    prev          = np.zeros(1, dtype=np.float32)
    stuck_count   = 0
    reverse_count = 0
    prev_cp_dist  = 100.0
    orbit_frames  = 0

    def policy(state: dict) -> tuple[float, float]:
        nonlocal prev, stuck_count, reverse_count, prev_cp_dist, orbit_frames

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

        if stuck_count >= (STUCK_THRESHOLD if wedged else PURE_STUCK_THR):
            reverse_count = REVERSE_FRAMES
            stuck_count   = 0

        # orbit detector: been near CP too long without passing → force reverse escape
        cp_near = sensors.get("checkpoint_distance", 100.0) < CP_ORBIT_DIST
        if cp_near:
            orbit_frames += 1
        else:
            orbit_frames = 0
        if orbit_frames >= CP_ORBIT_FRAMES and reverse_count == 0:
            reverse_count = REVERSE_FRAMES * 2
            orbit_frames  = 0

        if reverse_count > 0:
            reverse_count -= 1
            steer = -0.8 if right < left else 0.8
            prev  = np.array([steer], dtype=np.float32)
            return (-1.0, steer)

        # --- model steering ---
        path_feat = None
        if use_path:
            pos    = state.get("position") or {}
            pos_xz = np.array([pos.get("x", 0.0), pos.get("z", 0.0)], dtype=np.float32)
            path_feat = _path_features(pos_xz, path_ref)

        x     = sensors_to_input(sensors, path_feat)
        raw   = nn.forward(x, w)
        prev  = raw.copy()
        steer = float(np.clip(raw[0] * STEER_GAIN, -1.0, 1.0))

        # --- checkpoint homing ---
        cp_dist      = sensors.get("checkpoint_distance", 100.0)
        heading_err  = sensors.get("heading_error", 0.0)
        old_cp_dist  = prev_cp_dist
        approaching  = cp_dist < old_cp_dist
        prev_cp_dist = cp_dist
        throttle     = THROTTLE

        if cp_dist < CP_GATE_DIST:
            # Final gate commit: 100% heading correction, slow to thread precisely
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer    = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_ORBIT_DIST and not approaching:
            # Orbit escape: retreating from gate — crawl and re-aim
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            steer    = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            throttle = CP_MIN_THROTTLE

        elif cp_dist < CP_HOMING_DIST:
            heading_norm = float(np.clip(heading_err / np.pi, -1.0, 1.0))
            homing_steer = float(np.clip(-heading_norm * CP_HOMING_GAIN, -1.0, 1.0))
            t     = (CP_HOMING_DIST - cp_dist) / CP_HOMING_DIST
            blend = t * CP_HOMING_MAX
            steer = steer * (1.0 - blend) + homing_steer * blend
            # distance-based braking: full speed until CP_BRAKE_DIST, then ramp down
            if cp_dist < CP_BRAKE_DIST:
                brake_t  = (cp_dist - CP_GATE_DIST) / (CP_BRAKE_DIST - CP_GATE_DIST)
                brake_t  = float(np.clip(brake_t, 0.0, 1.0))
                throttle = CP_MIN_THROTTLE + (THROTTLE - CP_MIN_THROTTLE) * brake_t
            else:
                throttle = THROTTLE

        return (throttle, steer)

    return policy
