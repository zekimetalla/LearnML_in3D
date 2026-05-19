"""Step 2 — Inspect data, write backprop, train, save weights.

Run:  python 02_train.py --data data_v1.npz --tag v1

It loads the dataset from `--data`, saves diagnostic figures, runs the
gradient check on YOUR `my_backward()`, trains for 300 epochs (Adam,
batch 64, lr 1e-3, 90/10 train/val), and saves nav_<tag>.npz.

The function `my_backward` near the top is yours to fill in. The script
asserts that your gradients agree with numerical_gradient before it lets
training start. If the assertion fires, fix the bug.

This script is the baseline. Once you've passed the gradient check and got
your first benchmark, the iteration loop is yours: change the architecture
in `drive2win/nn.py`, change the data, change the training
schedule, retrain, rebenchmark, commit, repeat.
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# TODO — write backward()
# =========================================================================
# Walk the chain rule outward from the loss:
#   y = tanh(z3),  loss = MSE(y, target)
#   z3 = a2 W3 + b3,   a2 = ReLU(z2)
#   z2 = a1 W2 + b2,   a1 = ReLU(z1)
#   z1 = x  W1 + b1
#
# Replace each `...` with the correct expression.
# =========================================================================
def my_backward(x, y_target, w, cache):
    n = x.shape[0]
    y = cache["y"]
    # --- output ---
    dy  = 2.0 * (y - y_target) / (n * y.shape[1])
    dz3 = dy * (1.0 - y * y)   # tanh derivative: 1 - tanh(z)^2 = 1 - y^2
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)
    # --- hidden 2 ---
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)  # ReLU mask
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    # --- hidden 1 ---
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)  # ReLU mask
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check():
    rng = np.random.default_rng(0)
    # Use float64 for the check — numpy 2.0 + Apple Accelerate BLAS reduces
    # float32 matmul precision enough to corrupt finite-difference comparisons.
    w = {k: v.astype(np.float64) for k, v in nn_mod.init_weights(seed=0).items()}
    x = rng.normal(size=(8, N_FEATURES))
    y = rng.uniform(-1, 1, size=(8, N_ACTIONS))
    cache = nn_mod.forward_all(x, w)
    grads = my_backward(x, y, w, cache)

    print("\ngradient check (max relative error per parameter):")
    for key in w:
        max_err = 0.0
        flat = w[key].size
        for _ in range(5):
            idx = np.unravel_index(rng.integers(0, flat), w[key].shape)
            num = nn_mod.numerical_gradient(x, y, w, key, idx)
            ana = grads[key][idx]
            denom = max(1e-12, abs(num) + abs(ana))
            max_err = max(max_err, abs(num - ana) / denom)
        flag = "OK" if max_err < 1e-4 else "BUG"
        print(f"  {key}: {max_err:.2e}   {flag}")
        assert max_err < 1e-4, (
            f"backward() gradient for {key} disagrees with numerical_gradient. "
            f"Fix it before training."
        )


def smooth_actions(actions: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian low-pass filter along the time axis.

    Keyboard (WASD) recording produces discrete {-1, 0, +1} steps. Smoothing
    converts those into ramps so the model learns proportional control instead
    of hard snaps. sigma=3 ≈ 0.15 s at 20 Hz — enough to remove the
    discontinuities without blurring the intent of each manoeuvre.
    """
    smoothed = gaussian_filter1d(actions.astype(np.float64), sigma=sigma, axis=0)
    return np.clip(smoothed, -1.0, 1.0).astype(np.float32)


def inject_nav_data(states_raw: np.ndarray, actions: np.ndarray,
                    n: int = 8000, gain: float = 0.8, seed: int = 1) -> tuple:
    """Inject synthetic proportional-navigation samples.

    WASD data has contradictory steering labels for the same heading_error
    (sometimes left, sometimes straight, sometimes right) so the model averages
    them to zero and ignores heading_error entirely.

    Fix: sample real sensor states, replace their actions with the analytic
    proportional rule  steering = -heading_error_norm * gain,  throttle = 0.8.
    Mixing these with real data gives the model an unambiguous navigation signal
    while keeping all recorded recovery/wall-avoidance behaviour.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states_raw), size=n, replace=True)
    syn_states  = states_raw[idx].copy()

    # heading_error is feature index 1, already in [-pi, pi]
    # normalize.py divides by pi → [-1, 1]; replicate that here for the rule
    heading_norm = np.clip(syn_states[:, 1] / np.pi, -1.0, 1.0)
    syn_steering  = np.clip(-heading_norm * gain, -1.0, 1.0).astype(np.float32)
    syn_throttle  = np.full(n, 0.8, dtype=np.float32)
    syn_actions   = np.stack([syn_throttle, syn_steering], axis=1)

    combined_states  = np.concatenate([states_raw, syn_states],  axis=0)
    combined_actions = np.concatenate([actions,    syn_actions],  axis=0)
    print(f"  injected {n:,} synthetic nav samples (gain={gain})")
    return combined_states, combined_actions


def inspect_dataset(states_raw, actions, tag: str):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")
    viz.plot_action_histograms(actions, out=f"figures/fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"figures/fig_heading_{tag}.png")


def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    N = len(X)
    perm = rng.permutation(N); n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    w = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)
    train_losses, val_losses = [], []
    best_val = float("inf"); best = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i+batch_size], Ys[i:i+batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb); n_b += 1
            grads = my_backward(xb, yb, w, cache)
            nn_mod.adam_step(w, grads, state, lr=lr)
        v = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        train_losses.append(ep_loss / max(1, n_b)); val_losses.append(v)
        if v < best_val:
            best_val = v; best = {k: w[k].copy() for k in w}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  val={v:.4f}  best={best_val:.4f}")

    return best, train_losses, val_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1.npz",
                    help="Dataset file from 01_collect.py")
    ap.add_argument("--tag", default="v1",
                    help="Output suffix (nav_<tag>.npz, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian sigma for action smoothing (0 = off). "
                         "Try 3.0 to fix discrete WASD targets.")
    ap.add_argument("--nav-inject", type=int, default=0,
                    help="Number of synthetic proportional-nav samples to inject. "
                         "Try 8000 to fix heading/steering correlation.")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    import os; os.makedirs("figures", exist_ok=True)
    inspect_dataset(states_raw, actions, tag=args.tag)

    X = normalize_states(states_raw)
    Y = actions.astype(np.float32)
    if args.smooth > 0.0:
        Y = smooth_actions(Y, sigma=args.smooth)
        print(f"actions smoothed (sigma={args.smooth}): "
              f"std {actions.std():.3f} → {Y.std():.3f}")
    if args.nav_inject > 0:
        states_raw, Y = inject_nav_data(states_raw, Y, n=args.nav_inject)
        X = normalize_states(states_raw)
    print(f"\nX range : [{X.min():+.2f}, {X.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")

    gradient_check()

    weights, tr_losses, va_losses = train(
        X, Y, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    viz.plot_loss_curves(tr_losses, va_losses, out=f"figures/fig_loss_{args.tag}.png")
    nn_mod.save(weights, f"nav_{args.tag}.npz")
    print(f"Saved nav_{args.tag}.npz")


if __name__ == "__main__":
    main()
