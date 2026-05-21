"""Shared evaluation utilities.

Used by:
- `drive2win/benchmark.py` to score a saved model on the seeded course
- your iteration logs (`benchmarks/<tag>.json`)
- whatever live agent you ship for the tournament (so live behavior matches
  what you benchmarked)
"""
from __future__ import annotations
import time
from typing import Callable
import numpy as np


# ── Run a policy ─────────────────────────────────────────────────────────
def run_policy(client, policy_fn: Callable, duration: float = 60.0,
               hz: float = 20.0, on_step: Callable | None = None) -> dict:
    """Run a policy at fixed Hz against a connected GameClient.

    Args:
        client: connected GameClient with WebSocket open.
        policy_fn: (state_dict) -> (throttle, steering)
        duration: seconds to run.
        hz: control frequency.
        on_step: optional callback (step, state, action) -> None for logging.

    Returns:
        Dict with:
            steps: number of control steps issued
            elapsed: actual wall time
            checkpoints_passed: max value seen during the run
            crashes: count of position resets we detected
            min_speed_streak: longest run of "stuck" frames (speed < 0.3)
            track: list of {t, position, speed} samples (1 Hz subsample)
    """
    interval = 1.0 / hz
    start = time.time()
    steps = 0
    checkpoints_passed = 0
    crashes = 0
    last_pos = None
    last_respawn = None
    stuck_streak = 0
    max_stuck = 0
    track = []
    next_log = start

    while time.time() - start < duration:
        state = client.get_latest_state()
        if not state or "sensors" not in state:
            time.sleep(interval); continue

        # checkpoints
        nav = state["sensors"].get("navigation") or {}
        cp = nav.get("checkpoints_completed", 0) or 0
        checkpoints_passed = max(checkpoints_passed, cp)

        # speed-based stuck / crash heuristics
        sp = state["sensors"].get("speed", 0.0)
        if sp < 0.3:
            stuck_streak += 1
        else:
            max_stuck = max(max_stuck, stuck_streak)
            stuck_streak = 0

        # crash detection — prefer the first-class respawn_count counter,
        # fall back to the position-teleport heuristic for older browser builds
        # that don't broadcast the counter yet.
        pos = state.get("position") or {}
        respawn = nav.get("respawn_count")
        if respawn is not None:
            if last_respawn is not None and respawn > last_respawn:
                crashes += respawn - last_respawn
            last_respawn = respawn
        elif last_pos is not None and pos:
            dx = pos.get("x", 0) - last_pos.get("x", 0)
            dz = pos.get("z", 0) - last_pos.get("z", 0)
            if (dx * dx + dz * dz) > 25.0:  # > 5 m in one frame
                crashes += 1
        last_pos = pos

        # policy step
        throttle, steering = policy_fn(state)
        client.send_control_ws(throttle, steering)
        steps += 1
        if on_step is not None:
            on_step(steps, state, (throttle, steering))

        # 1 Hz track sample
        now = time.time()
        if now >= next_log:
            track.append({"t": now - start, "position": pos, "speed": sp})
            next_log = now + 1.0

        time.sleep(interval)

    elapsed = time.time() - start
    return {
        "steps": steps,
        "elapsed": elapsed,
        "checkpoints_passed": checkpoints_passed,
        "crashes": crashes,
        "min_speed_streak": max(max_stuck, stuck_streak),
        "track": track,
    }


# ── Score a saved model on the benchmark course ─────────────────────────
def score_runs(runs: list[dict], target_checkpoints: int) -> dict:
    """Aggregate a list of run results into the headline metrics."""
    completed = [r for r in runs if r["checkpoints_passed"] >= target_checkpoints]
    times = [r["elapsed"] for r in completed]
    crashes_per_run = [r["crashes"] for r in runs]

    return {
        "n_runs": len(runs),
        "completion_rate": len(completed) / max(1, len(runs)),
        "median_lap_time": float(np.median(times)) if times else float("inf"),
        "mean_crashes": float(np.mean(crashes_per_run)) if crashes_per_run else 0.0,
        "max_checkpoints": max(r["checkpoints_passed"] for r in runs),
    }
