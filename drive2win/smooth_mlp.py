"""MLP policy with exponential moving average smoothing on outputs.

Keyboard-recorded data has discrete actions (-1, 0, +1) which makes the
raw MLP output snap between those values. Smoothing filters the output
over time so steering/throttle ramp rather than jump — critical for
stable lap driving.

Usage (benchmark):
    python 03_benchmark.py --tag v2-epochs800 --module drive2win.smooth_mlp

Alpha controls how much weight to give the new prediction vs the previous
smoothed value. Lower = smoother but slower to react.
    alpha=0.6  (default) — moderate smoothing, good for steering
    alpha=0.8  — light smoothing, faster reaction
    alpha=0.3  — heavy smoothing, very stable but sluggish turns
"""
from __future__ import annotations
import numpy as np
from drive2win import nn
from drive2win.normalize import sensors_to_input, clip_action

ALPHA = 0.6  # EMA weight for new prediction


def make_policy(weights_path: str):
    """Return a smoothed MLP policy function.

    Compatible with benchmark.py's --module flag interface.
    """
    w = nn.load(weights_path)
    prev = np.zeros(2, dtype=np.float32)

    def policy(state: dict) -> tuple[float, float]:
        nonlocal prev
        x = sensors_to_input(state["sensors"])
        raw = nn.forward(x, w)                      # shape (2,)
        smoothed = ALPHA * raw + (1.0 - ALPHA) * prev
        prev = smoothed.copy()
        return clip_action(smoothed)

    return policy
