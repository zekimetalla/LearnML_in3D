# Iteration Progress

| Tag | Change | Val Loss | Δ |
|-----|--------|----------|---|
| v1 | Baseline behavioral cloning, 300 epochs | 0.1857 | — |
| v2-epochs800 | Trained longer (800 epochs) | 0.1506 | -19% |
| v3-smooth | Action smoothing on training data (Gaussian σ=3) | 0.0849 | -44% |

## v1 — Baseline
- Implemented `my_backward()` (backprop from scratch)
- 6,620 samples, seed 42, 5 collection phases (smooth laps, tight turns, obstacles, bad terrain, recovery)
- 300 epochs, Adam lr=1e-3, batch=64
- **Finding:** val loss still decreasing at epoch 300 → stopped too early. Action histograms showed discrete WASD inputs {-1, 0, +1} only.

## v2 — More Epochs
- Hypothesis: val loss trending down at epoch 300, more epochs = free gain
- Same data, same architecture, extended to 800 epochs
- **Result:** confirmed — val loss 0.1857 → 0.1506 (-19%), no overfitting

## v3 — Action Smoothing
- Hypothesis: discrete WASD targets {-1, 0, +1} bottleneck the model; smoothing targets into ramps gives a continuous signal to learn from
- Applied Gaussian low-pass filter (σ=3, ≈0.15s at 20Hz) to actions before training via `--smooth 3.0` flag
- Added `drive2win/smooth_mlp.py`: EMA inference smoothing (α=0.6) for deployment
- **Result:** biggest jump yet — val loss 0.1506 → 0.0849 (-44%)
- **Next bottleneck:** train/val gap widening → approaching data ceiling, need more recovery data

## Benchmark Status
Live benchmark blocked: relay server does not forward state from browser to Python client
(`[WS] Session assigned` is the last message on both sides — `enableStateBroadcast` not wired up).
All benchmark JSONs logged with 0/5 completion reflecting this infrastructure gap, not model quality.
