"""MLP forward pass, backprop, and Adam — all in NumPy.

This module is the heart of the project. You should be able to read every
line in this file and explain what it does.

Default architecture: 12 -> 64 -> 32 -> 2 with ReLU/ReLU/tanh and MSE loss.
You will probably change H1/H2 (and re-derive backward()) at some point in
your iteration — that's the whole point.

The reference `backward()` below is for use by your *later* iterations and
by benchmarking. The version YOU implement (and submit for grading) lives
in `scripts/02_train.py` as `my_backward()`. Do not peek at this one before
you've tried — the whole point of the first iteration is to write it.
"""
from __future__ import annotations
import numpy as np

H1, H2 = 128, 64
N_IN, N_OUT = 12, 2


# ── Forward pass ────────────────────────────────────────────────────────
def forward(x: np.ndarray, w: dict) -> np.ndarray:
    """Compute the forward pass.

    Args:
        x: shape (N, 12) or (12,). Normalized inputs.
        w: dict with keys W1, b1, W2, b2, W3, b3.

    Returns:
        Action of shape (N, 2) or (2,) in [-1, 1].
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]
    z1 = x @ w["W1"] + w["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ w["W2"] + w["b2"]
    a2 = np.maximum(0, z2)
    z3 = a2 @ w["W3"] + w["b3"]
    y = np.tanh(z3)
    return y[0] if single else y


def forward_all(x: np.ndarray, w: dict) -> dict:
    """Same as forward(), but also returns every intermediate value so we can
    backprop through it.
    """
    z1 = x @ w["W1"] + w["b1"];   a1 = np.maximum(0, z1)
    z2 = a1 @ w["W2"] + w["b2"];  a2 = np.maximum(0, z2)
    z3 = a2 @ w["W3"] + w["b3"];  y = np.tanh(z3)
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "y": y}


# ── Loss ─────────────────────────────────────────────────────────────────
def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())


# ── Backward pass ────────────────────────────────────────────────────────
def backward(x: np.ndarray, y_target: np.ndarray, w: dict, cache: dict) -> dict:
    """Return gradients dW1, db1, ..., dW3, db3 for one mini-batch.

    Args:
        x: (N, 12)
        y_target: (N, 2)
        w: weight dict
        cache: forward_all() output for this batch

    Returns:
        Dict mirroring `w`, with gradients.
    """
    n = x.shape[0]
    y = cache["y"]
    # MSE → d/dy
    dy = 2.0 * (y - y_target) / (n * y.shape[1])
    # tanh derivative: 1 - tanh(z3)^2 = 1 - y^2
    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


# ── Optimizer (Adam) ─────────────────────────────────────────────────────
def init_adam(w: dict) -> dict:
    return {
        "m": {k: np.zeros_like(v) for k, v in w.items()},
        "v": {k: np.zeros_like(v) for k, v in w.items()},
        "t": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    }


def adam_step(w: dict, grads: dict, state: dict, lr: float = 1e-3) -> None:
    """In-place Adam update."""
    state["t"] += 1
    t = state["t"]
    b1, b2, eps = state["beta1"], state["beta2"], state["eps"]
    for k in w:
        g = grads[k]
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * (g * g)
        m_hat = state["m"][k] / (1 - b1 ** t)
        v_hat = state["v"][k] / (1 - b2 ** t)
        w[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── Initialization ───────────────────────────────────────────────────────
def init_weights(seed: int = 0) -> dict:
    """He-init for ReLU layers, Xavier-ish for the tanh output."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / N_IN), (N_IN, H1)).astype(np.float32),
        "b1": np.zeros(H1, dtype=np.float32),
        "W2": rng.normal(0, np.sqrt(2 / H1), (H1, H2)).astype(np.float32),
        "b2": np.zeros(H2, dtype=np.float32),
        "W3": rng.normal(0, np.sqrt(1 / H2), (H2, N_OUT)).astype(np.float32),
        "b3": np.zeros(N_OUT, dtype=np.float32),
    }


# ── Save / load ──────────────────────────────────────────────────────────
def save(weights: dict, path: str) -> None:
    np.savez(path, **weights)


def load(path: str) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


# ── Gradient check (Part 1, your safety net) ─────────────────────────────
def numerical_gradient(x: np.ndarray, y_target: np.ndarray, w: dict,
                       key: str, idx: tuple, h: float = 1e-4) -> float:
    """Two-sided finite difference for ONE entry of one weight matrix.

    Use this to verify your backward() against gold-standard numerical
    gradients. If your analytic gradient and the numerical one disagree
    by more than ~1e-4, your backprop has a bug.
    """
    w[key][idx] += h
    loss_p = mse_loss(forward(x, w), y_target)
    w[key][idx] -= 2 * h
    loss_m = mse_loss(forward(x, w), y_target)
    w[key][idx] += h  # restore
    return (loss_p - loss_m) / (2 * h)


def check_gradients(x: np.ndarray, y: np.ndarray, w: dict, n_samples: int = 5) -> dict:
    """Compare analytic and numerical gradients on a handful of random
    weights from each parameter. Returns a dict of max relative errors.
    """
    cache = forward_all(x, w)
    grads = backward(x, y, w, cache)
    rng = np.random.default_rng(0)
    report = {}
    for key in w:
        max_err = 0.0
        flat_size = w[key].size
        for _ in range(n_samples):
            flat_idx = rng.integers(0, flat_size)
            idx = np.unravel_index(flat_idx, w[key].shape)
            num = numerical_gradient(x, y, w, key, idx)
            ana = grads[key][idx]
            denom = max(1e-12, abs(num) + abs(ana))
            err = abs(num - ana) / denom
            max_err = max(max_err, err)
        report[key] = max_err
    return report
